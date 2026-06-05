import numpy as np
import torch
import torch.nn as nn
import librosa
import json
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from scipy.io.wavfile import write
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE         = Path(r'C:\Users\TANISH VERMA\OneDrive\Desktop\Mindify')
TEXT_PATH    = BASE / 'models' / 'mindify-final-v3'
SPEECH_PATH  = BASE / 'models' / 'speech_model_v2'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Labels ────────────────────────────────────────────────────
TEXT_LABELS   = {0: 'Depression', 1: 'Anxiety', 2: 'Stress'}
EMOTION_MAP   = {
    'sad':      'Depression',
    'anxious':  'Anxiety',
    'stressed': 'Stress',
    'calm':     'Depression',
}

# ── Load text model ───────────────────────────────────────────
print("Loading models...")
tokenizer  = AutoTokenizer.from_pretrained(str(TEXT_PATH))
text_model = AutoModelForSequenceClassification.from_pretrained(str(TEXT_PATH))
text_model = text_model.to(DEVICE)
text_model.eval()

# ── Load speech config ────────────────────────────────────────
with open(SPEECH_PATH / 'speech_config.json') as f:
    cfg = json.load(f)

INPUT_SIZE   = cfg['input_size']
NUM_CLASSES  = cfg['num_classes']
CLASS_NAMES  = cfg['class_names']
SAMPLE_RATE  = cfg['sample_rate']
DURATION     = cfg['duration']
FEATURE_MEAN = np.load(SPEECH_PATH / 'feature_mean.npy')
FEATURE_STD  = np.load(SPEECH_PATH / 'feature_std.npy')

# ── Speech model ──────────────────────────────────────────────
class SpeechEmotionCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3)
        )
        conv_out = (input_size // 16) * 512
        self.classifier = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128),      nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

speech_model = SpeechEmotionCNN(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
speech_model.load_state_dict(
    torch.load(SPEECH_PATH / 'best_speech_model.pt', map_location=DEVICE)
)
speech_model.eval()
print("All models loaded.\n")

# ── Text prediction ───────────────────────────────────────────
def predict_text(text):
    inputs = tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(text_model(**inputs).logits, dim=1)[0]
    pred = probs.argmax().item()
    return TEXT_LABELS[pred], probs.cpu().numpy()

# ── Feature extraction ────────────────────────────────────────
def extract_features(audio, sr):
    target = int(sr * DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean   = np.mean(mfcc, axis=1)
    mfcc_std    = np.std(mfcc,  axis=1)
    chroma      = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    mel         = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_mean    = np.mean(mel, axis=1)
    zcr_mean    = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms_mean    = np.mean(librosa.feature.rms(y=audio))

    features = np.concatenate([
        mfcc_mean, mfcc_std, chroma_mean,
        mel_mean, [zcr_mean], [rms_mean]
    ]).astype(np.float32)

    return (features - FEATURE_MEAN) / FEATURE_STD

# ── Speech prediction ─────────────────────────────────────────
def predict_speech(audio, sr):
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    features = extract_features(audio, sr)
    tensor   = torch.tensor(features).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(speech_model(tensor), dim=1)[0].cpu().numpy()

    pred_idx      = probs.argmax()
    raw_emotion   = CLASS_NAMES[pred_idx]
    mindify_label = EMOTION_MAP[raw_emotion]
    return mindify_label, raw_emotion, probs

# ── Display result ────────────────────────────────────────────
def show_text_result(label, probs):
    print("\n" + "="*45)
    print(f"  PREDICTION : {label}")
    print("="*45)
    print("  Confidence breakdown:")
    for i, name in TEXT_LABELS.items():
        bar = '█' * int(probs[i] * 30)
        print(f"  {name:<12} {bar:<30} {probs[i]*100:.1f}%")
    print("="*45)

def show_speech_result(label, raw, probs):
    print("\n" + "="*45)
    print(f"  PREDICTION : {label}")
    print(f"  Raw emotion: {raw}")
    print("="*45)
    print("  Confidence breakdown:")
    for i, name in enumerate(CLASS_NAMES):
        mindify = EMOTION_MAP[name]
        bar = '█' * int(probs[i] * 30)
        print(f"  {name:<10} → {mindify:<12} {bar:<30} {probs[i]*100:.1f}%")
    print("="*45)

# ── Main ──────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════╗")
    print("║           MINDIFY — Mental Health AI     ║")
    print("║   Depression | Anxiety | Stress Detector ║")
    print("╚══════════════════════════════════════════╝")

    while True:
        print("\nWhat would you like to do?")
        print("  1. Write text")
        print("  2. Record voice")
        print("  3. Upload audio file")
        print("  4. Quit")
        print()

        choice = input("Enter choice (1/2/3/4): ").strip()

        # ── Option 1: Text ────────────────────────────────────
        if choice == '1':
            print("\nEnter your text below (press Enter twice to submit):")
            lines = []
            while True:
                line = input()
                if line == '' and lines:
                    break
                lines.append(line)
            text = ' '.join(lines).strip()

            if len(text) < 5:
                print("Text too short. Please write more.")
                continue

            print("\nAnalysing text...")
            label, probs = predict_text(text)
            show_text_result(label, probs)

        # ── Option 2: Record ──────────────────────────────────
        elif choice == '2':
            try:
                dur = int(input("Recording duration in seconds (default 5): ") or 5)
            except ValueError:
                dur = 5

            print(f"\nSpeak for {dur} seconds... starting now!")
            audio = sd.rec(
                int(dur * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            audio = audio.flatten()
            print("Recording done. Analysing...")

            label, raw, probs = predict_speech(audio, SAMPLE_RATE)
            show_speech_result(label, raw, probs)

        # ── Option 3: Upload file ─────────────────────────────
        elif choice == '3':
            path = input("\nEnter audio file path (.wav/.mp3): ").strip()
            path = path.strip('"')  # remove quotes if dragged in

            if not Path(path).exists():
                print(f"File not found: {path}")
                continue

            try:
                audio, sr = sf.read(path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
                print(f"Loaded: {Path(path).name} "
                      f"({sr}Hz, {len(audio)/sr:.1f}s)")
                print("Analysing...")

                label, raw, probs = predict_speech(audio, sr)
                show_speech_result(label, raw, probs)

            except Exception as e:
                print(f"Error loading file: {e}")

        # ── Option 4: Quit ────────────────────────────────────
        elif choice == '4':
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")

if __name__ == '__main__':
    main()