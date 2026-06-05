import numpy as np
import torch
import torch.nn as nn
import librosa
import json
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from scipy.io.wavfile import write

BASE         = Path(r'C:\Users\TANISH VERMA\OneDrive\Desktop\Mindify')
SPEECH_PATH  = BASE / 'models' / 'speech_model_v2'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load config ───────────────────────────────────────────────
with open(SPEECH_PATH / 'speech_config.json') as f:
    cfg = json.load(f)

INPUT_SIZE    = cfg['input_size']
NUM_CLASSES   = cfg['num_classes']
CLASS_NAMES   = cfg['class_names']
SAMPLE_RATE   = cfg['sample_rate']
DURATION      = cfg['duration']
FEATURE_MEAN  = np.load(SPEECH_PATH / 'feature_mean.npy')
FEATURE_STD   = np.load(SPEECH_PATH / 'feature_std.npy')

print(f"Classes: {CLASS_NAMES}")
print(f"Input size: {INPUT_SIZE}")
print(f"Sample rate: {SAMPLE_RATE}")

# ── Emotion mapping ───────────────────────────────────────────
EMOTION_MAP = {
    'sad':      'Depression',
    'anxious':  'Anxiety',
    'stressed': 'Stress',
    'calm':     'Depression',
}

# ── Model ─────────────────────────────────────────────────────
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

# Load model
speech_model = SpeechEmotionCNN(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
speech_model.load_state_dict(
    torch.load(SPEECH_PATH / 'best_speech_model.pt', map_location=DEVICE)
)
speech_model.eval()
print("Speech model loaded.")

# ── Feature extraction (must match training exactly) ──────────
def extract_features(audio, sr):
    # Pad or truncate to fixed duration
    target = int(sr * DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    # MFCC - 40 coefficients - mean + std = 80
    mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)   # 40
    mfcc_std  = np.std(mfcc,  axis=1)   # 40

    # Chroma - mean = 12
    chroma      = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # 12

    # Mel spectrogram - mean = 128
    mel      = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_mean = np.mean(mel, axis=1)  # 128

    # ZCR mean = 1, RMS mean = 1
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms_mean = np.mean(librosa.feature.rms(y=audio))

    # Concatenate: 40+40+12+128+1+1 = 222
    features = np.concatenate([
        mfcc_mean, mfcc_std, chroma_mean,
        mel_mean, [zcr_mean], [rms_mean]
    ]).astype(np.float32)

    # Normalize using training stats
    features = (features - FEATURE_MEAN) / FEATURE_STD
    return features

# ── Predict from audio array ──────────────────────────────────
def predict_speech(audio, sr):
    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    features = extract_features(audio, sr)
    tensor   = torch.tensor(features).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(speech_model(tensor), dim=1)[0].cpu().numpy()

    pred_idx     = probs.argmax()
    raw_emotion  = CLASS_NAMES[pred_idx]
    mindify_label = EMOTION_MAP[raw_emotion]
    confidence   = probs[pred_idx] * 100

    print(f"\nRaw emotion  : {raw_emotion} ({confidence:.1f}%)")
    print(f"Mindify label: {mindify_label}")
    print(f"All scores:")
    for i, name in enumerate(CLASS_NAMES):
        mindify = EMOTION_MAP[name]
        bar = '#' * int(probs[i] * 30)
        print(f"  {name:10s} → {mindify:12s} {bar:<30} {probs[i]*100:.1f}%")

    return mindify_label, raw_emotion, probs

# ── Record from microphone ────────────────────────────────────
def record_and_predict(duration=5):
    print(f"\nSpeak for {duration} seconds... starting now!")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    audio = audio.flatten()
    print("Recording done. Analysing...")
    return predict_speech(audio, SAMPLE_RATE)

# ── Predict from file ─────────────────────────────────────────
def predict_from_file(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    print(f"Loaded: {path} ({sr}Hz, {len(audio)/sr:.1f}s)")
    return predict_speech(audio, sr)

# ── Main loop ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("Mindify Speech Predictor")
    print("Commands: r=record | f=file | q=quit")
    print("="*50)

    while True:
        cmd = input("\nCommand (r/f/q): ").strip().lower()

        if cmd == 'q':
            print("Goodbye!")
            break
        elif cmd == 'r':
            try:
                dur = int(input("Duration in seconds (default 5): ") or 5)
            except:
                dur = 5
            record_and_predict(dur)
        elif cmd == 'f':
            path = input("Audio file path: ").strip()
            predict_from_file(path)
        else:
            print("Unknown command.")