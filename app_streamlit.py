import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import json
import soundfile as sf
import tempfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq

# ── Config ────────────────────────────────────────────────────
BASE        = Path(__file__).parent
TEXT_PATH   = BASE / 'models' / 'mindify-final-v3'
SPEECH_PATH = BASE / 'models' / 'speech_model_v2'
DEVICE      = torch.device('cpu')

import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

TEXT_LABELS = {0: 'Depression', 1: 'Anxiety', 2: 'Stress'}
EMOTION_MAP = {'sad': 'Depression', 'anxious': 'Anxiety', 'stressed': 'Stress', 'calm': 'Depression'}

FIRST_MSG_PROMPT = """You are MindSpace, a warm and emotionally intelligent companion.
The user may be experiencing: {emotion}
Your task:
1. Acknowledge what they shared — reflect it back warmly.
2. Show genuine understanding without being clinical.
3. Briefly normalise the feeling so they don't feel alone.
4. Ask ONE thoughtful follow-up question to continue the conversation.
Rules:
- Never say "the AI detected" or mention confidence scores.
- Never use labels like "you have anxiety". Speak to the feeling, not the label.
- Keep it 2–5 sentences. Warm, human, not therapist-like.
"""

FIRST_VOICE_PROMPT = """You are MindSpace, a warm and emotionally intelligent companion.
The user just shared a voice recording. Based on their voice, they may be experiencing: {emotion}
Your task:
1. Open gently — acknowledge they reached out.
2. Reflect what someone feeling {emotion} might be going through.
3. Invite them to share more in their own words.
4. Ask ONE open, caring question.
Rules:
- Never say "the model detected" or reference audio analysis.
- Sound like a caring friend, not a therapist.
- Keep it 2–4 sentences.
"""

FOLLOWUP_PROMPT = """You are MindSpace, a warm and emotionally intelligent companion.
Background: This user initially appeared to be experiencing {emotion}.
Do NOT keep mentioning or labelling this emotion. It is background context only.
Focus on:
- Understanding what they're sharing right now
- Helping them process their thoughts
- Asking thoughtful questions
- Offering gentle perspective when useful
- Sounding like a caring, grounded human friend
Rules:
- No clinical language. No therapist speak.
- 2–4 sentences per response.
- End with a question only if it feels natural.
"""

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MindSpace",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;1,400;1,500&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:        #f5f0e8;
    --bg2:       #ede6d6;
    --bg3:       #e8dfc8;
    --surface:   #faf7f2;
    --surface2:  #f0ebe0;
    --border:    rgba(139,115,85,0.12);
    --border2:   rgba(139,115,85,0.20);
    --border3:   rgba(139,115,85,0.32);
    --text:      #2c2416;
    --text2:     #5c4f3a;
    --muted:     #9c8a6e;
    --muted2:    #c4ae90;
    --accent:    #8b6f47;
    --accent2:   #a07c52;
    --gold:      #c4983a;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
}

[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 280px;
    background: linear-gradient(180deg, rgba(196,152,58,0.06) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }

.block-container {
    padding: 0 1.5rem 5rem 1.5rem !important;
    max-width: 740px !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--muted2); border-radius: 4px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    border: 1px solid var(--border2) !important;
    gap: 3px !important;
    box-shadow: 0 1px 4px rgba(139,115,85,0.08) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    box-shadow: 0 2px 8px rgba(139,115,85,0.12) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Chat input ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    background: var(--bg) !important;
    background-color: var(--bg) !important;
}

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 18px !important;
    box-shadow: 0 2px 16px rgba(139,115,85,0.08) !important;
}

[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: transparent !important;
    background-color: transparent !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    caret-color: var(--accent) !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted2) !important;
    -webkit-text-fill-color: var(--muted2) !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(139,111,71,0.5) !important;
    box-shadow: 0 0 0 3px rgba(139,111,71,0.08), 0 4px 20px rgba(139,115,85,0.12) !important;
}

[data-testid="stChatInput"] button,
[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #8b6f47, #a07c52) !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(139,111,71,0.25) !important;
}

/* ── Audio recorder timer fix ── */
[data-testid="stAudioInput"] {
    background: var(--surface) !important;
    border-radius: 18px !important;
}
[data-testid="stAudioInput"] > div {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border3) !important;
    border-radius: 18px !important;
    padding: 1.5rem !important;
}
/* The timer badge inside the audio recorder */
[data-testid="stAudioInput"] span,
[data-testid="stAudioInput"] div[class*="timer"],
[data-testid="stAudioInput"] div[class*="Timer"] {
    background: var(--bg2) !important;
    background-color: var(--bg2) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
    gap: 12px !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: var(--text2) !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 18px 4px 18px 18px !important;
    padding: 0.9rem 1.2rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 0.9rem 1.2rem !important;
    box-shadow: 0 2px 12px rgba(139,115,85,0.06) !important;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 34px !important; height: 34px !important;
    border-radius: 10px !important;
    font-size: 1rem !important; flex-shrink: 0 !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #8b6f47, #c4983a) !important;
    box-shadow: 0 4px 14px rgba(139,111,71,0.25) !important;
}
[data-testid="stChatMessageAvatarUser"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
}

/* ── Buttons ── */
.stButton button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 12px !important;
    font-size: 0.87rem !important;
    height: 40px !important;
    transition: all 0.2s ease !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #8b6f47 0%, #a07c52 100%) !important;
    border: none !important;
    color: #faf7f2 !important;
    box-shadow: 0 3px 16px rgba(139,111,71,0.3), inset 0 1px 0 rgba(255,255,255,0.12) !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(139,111,71,0.38) !important;
}
.stButton button[kind="secondary"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text2) !important;
}
.stButton button[kind="secondary"]:hover {
    border-color: var(--border3) !important;
    background: var(--bg2) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] > div {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border3) !important;
    border-radius: 18px !important;
    padding: 2.5rem 1rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover > div {
    border-color: rgba(139,111,71,0.45) !important;
    background: rgba(139,111,71,0.03) !important;
}
[data-testid="stFileUploader"] p {
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

[data-testid="stAlert"] {
    background: rgba(196,132,90,0.06) !important;
    border: 1px solid rgba(196,132,90,0.18) !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text2) !important;
}

audio {
    width: 100% !important;
    border-radius: 12px !important;
    margin: 0.5rem 0 !important;
}

[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px rgba(139,115,85,0.06) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--text2) !important;
    padding: 0.9rem 1.2rem !important;
}

h3 {
    font-family: 'Lora', serif !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    font-size: 1.05rem !important;
}
[data-testid="stHorizontalBlock"] { gap: 0.6rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained(str(TEXT_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(TEXT_PATH))
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_speech_model():
    with open(SPEECH_PATH / 'speech_config.json') as f:
        cfg = json.load(f)
    mean = np.load(SPEECH_PATH / 'feature_mean.npy')
    std  = np.load(SPEECH_PATH / 'feature_std.npy')

    class SpeechEmotionCNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.conv_block1 = nn.Sequential(
                nn.Conv1d(1, 64,  kernel_size=3, padding=1),
                nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2))
            self.conv_block2 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2))
            self.conv_block3 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3))
            self.conv_block4 = nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3))
            conv_out = (input_size // 16) * 512
            self.classifier = nn.Sequential(
                nn.Linear(conv_out, 512), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(512, 128),      nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, num_classes))

        def forward(self, x):
            x = x.unsqueeze(1)
            for block in [self.conv_block1, self.conv_block2,
                          self.conv_block3, self.conv_block4]:
                x = block(x)
            return self.classifier(x.flatten(start_dim=1))

    model = SpeechEmotionCNN(cfg['input_size'], cfg['num_classes'])
    model.load_state_dict(torch.load(
        SPEECH_PATH / 'best_speech_model.pt', map_location='cpu'))
    model.eval()
    return model, cfg, mean, std

@st.cache_resource
def load_groq():
    return Groq(api_key=GROQ_API_KEY)


# ── Prediction functions ──────────────────────────────────────
def predict_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    pred = probs.argmax().item()
    return TEXT_LABELS[pred], probs.cpu().numpy()

def extract_features(audio, sr, cfg, mean, std):
    target = int(sr * cfg['duration'])
    audio  = np.pad(audio, (0, max(0, target - len(audio))))[:target]
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    features = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(chroma, axis=1), np.mean(mel, axis=1),
        [np.mean(librosa.feature.zero_crossing_rate(audio))],
        [np.mean(librosa.feature.rms(y=audio))]
    ]).astype(np.float32)
    return (features - mean) / std

def predict_speech(audio, sr, model, cfg, mean, std):
    if sr != cfg['sample_rate']:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg['sample_rate'])
        sr = cfg['sample_rate']
    features = extract_features(audio, sr, cfg, mean, std)
    tensor = torch.tensor(features).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].numpy()
    pred_idx = probs.argmax()
    raw = cfg['class_names'][pred_idx]
    return EMOTION_MAP[raw], raw, probs


# ── LLM helpers ───────────────────────────────────────────────
def call_groq(groq_client, messages, max_tokens=180):
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.75,
        )
        return resp.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            return "I'm here — just hit a brief rate limit. Please try again in a few seconds."
        return "Something went wrong. Please try again."

def first_text_reply(groq_client, user_message, emotion):
    return call_groq(groq_client, [
        {"role": "system", "content": FIRST_MSG_PROMPT.format(emotion=emotion)},
        {"role": "user",   "content": user_message},
    ])

def first_voice_reply(groq_client, emotion):
    return call_groq(groq_client, [
        {"role": "system", "content": FIRST_VOICE_PROMPT.format(emotion=emotion)},
        {"role": "user",   "content": "[The user shared a voice recording]"},
    ])

def followup_reply(groq_client, history, emotion, user_message):
    messages = [{"role": "system", "content": FOLLOWUP_PROMPT.format(emotion=emotion)}]
    for msg in history[-12:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return call_groq(groq_client, messages)


# ── Emotion styles ────────────────────────────────────────────
EMOTION_STYLES = {
    'Depression': {
        'color': '#5a7fb5', 'light': '#dde8f5', 'border': '#b8cfe8',
        'icon': '💙', 'label': 'Low Mood',
        'info': "Your words carry a quietness — a weight that can make even small things feel heavy. That heaviness is real, and you don't have to carry it alone."
    },
    'Anxiety':    {
        'color': '#b5714a', 'light': '#f5e6db', 'border': '#e8c4aa',
        'icon': '🧡', 'label': 'Anxiety',
        'info': "There's a restlessness in your words — a mind that won't settle. That tightness is your nervous system speaking. It makes complete sense."
    },
    'Stress':     {
        'color': '#4a8c6e', 'light': '#ddf0e6', 'border': '#aad4be',
        'icon': '💚', 'label': 'Stress',
        'info': "Your words reveal real pressure — too much pulling at you at once. That feeling of being stretched thin is one of the most human experiences there is."
    },
}

def render_result_card(label, probs, labels_dict, is_speech=False, raw_emotion=None):
    import streamlit.components.v1 as components
    import html as _h

    s     = EMOTION_STYLES[label]
    color = s['color']
    light = s['light']
    bdr   = s['border']
    icon  = s['icon']
    info  = _h.escape(s['info'])

    raw_badge = ""
    if raw_emotion:
        raw_badge = (
            f"<span style='display:inline-block;margin-top:6px;"
            f"background:{light};border:1px solid {bdr};"
            f"border-radius:20px;padding:2px 10px;"
            f"font-size:0.72rem;color:{color};font-weight:500;'>"
            f"voice: {_h.escape(str(raw_emotion))}</span>"
        )

    if is_speech:
        items = [(cfg_name, EMOTION_MAP[cfg_name], float(probs[i]))
                 for i, cfg_name in enumerate(list(labels_dict))]
    else:
        items = [(name, name, float(probs[i])) for i, name in labels_dict.items()]

    bars_html = ""
    for (dname, mname, prob) in items:
        es  = EMOTION_STYLES[mname]
        c   = es['color']
        lt  = es['light']
        pct = prob * 100
        ltxt = _h.escape(f"{dname} → {mname}") if is_speech else _h.escape(dname)
        bars_html += f"""
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:0.82rem;color:#6b5a42;display:flex;align-items:center;gap:7px;">
              {es['icon']} <span>{ltxt}</span>
            </span>
            <span style="font-size:0.82rem;font-weight:600;color:{c};">{pct:.1f}%</span>
          </div>
          <div style="background:{lt};border-radius:8px;height:7px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{c}99,{c});border-radius:8px;transition:width 0.6s ease;"></div>
          </div>
        </div>"""

    html = f"""
    <!DOCTYPE html><html><head>
    <link href='https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,500&family=DM+Sans:wght@400;500;600&display=swap' rel='stylesheet'>
    <style>* {{ margin:0;padding:0;box-sizing:border-box; }} body {{ background:transparent;font-family:'DM Sans',sans-serif; }}</style>
    </head><body>
    <div style="background:{light};border:1px solid {bdr};border-radius:20px;padding:1.6rem 1.8rem;position:relative;overflow:hidden;margin-bottom:4px;">
      <div style="position:absolute;top:-15px;right:-15px;width:120px;height:120px;background:radial-gradient(circle,{color}15 0%,transparent 65%);pointer-events:none;"></div>
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:1.1rem;">
        <div style="width:54px;height:54px;background:white;border:1px solid {bdr};border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;box-shadow:0 2px 12px {color}20;">{icon}</div>
        <div>
          <div style="font-size:0.68rem;color:{color};text-transform:uppercase;letter-spacing:0.14em;font-weight:600;margin-bottom:3px;">Emotional state detected</div>
          <div style="font-size:1.7rem;font-weight:600;color:#2c2416;font-family:'Lora',serif;letter-spacing:-0.01em;line-height:1;">{_h.escape(label)}</div>
          {raw_badge}
        </div>
      </div>
      <div style="border-left:3px solid {color};padding-left:14px;font-family:'Lora',serif;font-style:italic;font-size:0.9rem;color:#5c4f3a;line-height:1.7;">{info}</div>
    </div>
    <div style="background:#faf7f2;border:1px solid rgba(139,115,85,0.16);border-radius:16px;padding:1.2rem 1.5rem;margin-top:10px;">
      <div style="font-size:0.68rem;color:#9c8a6e;text-transform:uppercase;letter-spacing:0.14em;font-weight:600;margin-bottom:14px;">Confidence breakdown</div>
      {bars_html}
    </div>
    </body></html>
    """
    total_height = 280 + len(items) * 62
    components.html(html, height=total_height, scrolling=False)


# ── Session state ─────────────────────────────────────────────
for key, val in {
    'text_chat_history': [], 'text_conversation': [],
    'text_emotion': None,    'text_probs': None,
    'voice_chat_history': [],'voice_conversation': [],
    'voice_emotion': None,   'voice_probs': None,
    'voice_raw': None,       'voice_labels_dict': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Load resources ────────────────────────────────────────────
with st.spinner("Starting MindSpace…"):
    tokenizer, text_model        = load_text_model()
    speech_model, cfg, mean, std = load_speech_model()
    groq_client                  = load_groq()


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@keyframes fade-up { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
@keyframes pulse-leaf { 0%,100% { opacity:0.7; transform:scale(1); } 50% { opacity:1; transform:scale(1.05); } }
</style>
<div style="padding:3rem 0 2.2rem 0;text-align:center;animation:fade-up 0.7s ease both;">
  <div style="display:inline-flex;align-items:center;gap:8px;background:#f0ebe0;border:1px solid rgba(139,115,85,0.25);border-radius:100px;padding:0.35rem 1.1rem;font-size:0.71rem;color:#8b6f47;letter-spacing:0.13em;text-transform:uppercase;font-family:'DM Sans',sans-serif;font-weight:600;margin-bottom:1.4rem;box-shadow:0 1px 6px rgba(139,115,85,0.1);">
    <span style="animation:pulse-leaf 2.5s ease-in-out infinite;display:inline-block;">🌿</span>
    Your mental wellness companion
  </div>
  <div style="margin-bottom:0.8rem;">
    <span style="font-family:'Lora',serif;font-size:3.2rem;font-weight:600;letter-spacing:-0.02em;color:#2c2416;">Mind</span><span style="font-family:'Lora',serif;font-size:3.2rem;font-weight:400;font-style:italic;letter-spacing:-0.02em;color:#8b6f47;">Space</span>
  </div>
  <p style="color:#9c8a6e;font-size:0.95rem;margin:0 auto;max-width:360px;line-height:1.7;font-family:'DM Sans',sans-serif;font-weight:400;">
    A quiet place to share what you're carrying —<br>through words or voice.
  </p>
  <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-top:2rem;opacity:0.45;">
    <div style="width:40px;height:1px;background:linear-gradient(90deg,transparent,#8b6f47);"></div>
    <div style="font-size:0.9rem;">🍃</div>
    <div style="width:40px;height:1px;background:linear-gradient(90deg,#8b6f47,transparent);"></div>
  </div>
  <div style="margin-top:1.5rem;font-family:'Lora',serif;font-style:italic;font-size:0.88rem;color:#c4ae90;letter-spacing:0.01em;">
    "It's okay to not be okay. Your feelings are valid."
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["💬  Text Chat", "🎤  Upload Voice", "🎙️  Record Voice"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — Text Chat
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([6, 1])
    with col_l:
        if st.session_state.text_emotion is None:
            st.markdown("""
            <div style="margin-bottom:1.2rem;">
              <div style="font-family:'Lora',serif;font-size:1.2rem;font-weight:500;color:#2c2416;margin-bottom:0.3rem;">What's on your mind?</div>
              <div style="font-size:0.83rem;color:#9c8a6e;line-height:1.6;">Share freely — there's no right way to begin.</div>
            </div>
            """, unsafe_allow_html=True)
    with col_r:
        if st.session_state.text_conversation:
            if st.button("Clear", key="text_clear"):
                for k in ['text_chat_history','text_conversation','text_emotion','text_probs']:
                    st.session_state[k] = [] if isinstance(st.session_state[k], list) else None
                st.rerun()

    if st.session_state.text_emotion is not None:
        with st.expander(f"{EMOTION_STYLES[st.session_state.text_emotion]['icon']}  Detected: {st.session_state.text_emotion}", expanded=False):
            render_result_card(st.session_state.text_emotion, st.session_state.text_probs, TEXT_LABELS)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    for turn in st.session_state.text_conversation:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant", avatar="🌿"):
            st.write(turn["reply"])

    user_message = st.chat_input("Share what you're feeling…", key="text_input")
    if user_message:
        is_first = len(st.session_state.text_conversation) == 0
        if is_first:
            with st.spinner("Reading your message…"):
                label, probs = predict_text(user_message, tokenizer, text_model)
            st.session_state.text_emotion = label
            st.session_state.text_probs   = probs
            with st.spinner("MindSpace is responding…"):
                reply = first_text_reply(groq_client, user_message, label)
        else:
            with st.spinner("MindSpace is responding…"):
                reply = followup_reply(groq_client, st.session_state.text_chat_history,
                                       st.session_state.text_emotion, user_message)
        st.session_state.text_chat_history += [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply}
        ]
        st.session_state.text_conversation.append({"user": user_message, "reply": reply})
        st.rerun()

    st.markdown("""
    <p style="color:#c4ae90;font-size:0.75rem;margin-top:2rem;text-align:center;font-family:'DM Sans',sans-serif;">
      🌱 MindSpace is not a clinical tool — please speak with a professional if needed.
    </p>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# Shared voice helpers
# ════════════════════════════════════════════════════════════════
def render_voice_results_and_chat(chat_input_key, clear_key):
    if st.session_state.voice_emotion is None:
        return
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    with st.expander(
        f"{EMOTION_STYLES[st.session_state.voice_emotion]['icon']}  Detected: {st.session_state.voice_emotion}",
        expanded=True
    ):
        render_result_card(
            st.session_state.voice_emotion, st.session_state.voice_probs,
            st.session_state.voice_labels_dict, is_speech=True,
            raw_emotion=st.session_state.voice_raw
        )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    for turn in st.session_state.voice_conversation:
        role = turn["role"]
        with st.chat_message("assistant" if role == "assistant" else "user",
                             avatar="🌿" if role == "assistant" else None):
            st.write(turn["content"])
    voice_msg = st.chat_input("Continue talking with MindSpace…", key=chat_input_key)
    if voice_msg:
        with st.spinner("MindSpace is responding…"):
            reply = followup_reply(groq_client, st.session_state.voice_chat_history,
                                   st.session_state.voice_emotion, voice_msg)
        for lst in ['voice_chat_history', 'voice_conversation']:
            st.session_state[lst].append({"role": "user", "content": voice_msg})
            st.session_state[lst].append({"role": "assistant", "content": reply})
        st.rerun()
    if st.button("↺  Start new conversation", key=clear_key):
        for k in ['voice_chat_history','voice_conversation','voice_emotion',
                  'voice_probs','voice_raw','voice_labels_dict']:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else None
        st.rerun()


def process_audio_and_respond(audio_array, sr_rate):
    label, raw, probs = predict_speech(audio_array, sr_rate, speech_model, cfg, mean, std)
    st.session_state.voice_emotion     = label
    st.session_state.voice_probs       = probs
    st.session_state.voice_raw         = raw
    st.session_state.voice_labels_dict = cfg['class_names']
    with st.spinner("MindSpace is responding…"):
        reply = first_voice_reply(groq_client, label)
    st.session_state.voice_chat_history = [{"role": "assistant", "content": reply}]
    st.session_state.voice_conversation = [{"role": "assistant", "content": reply}]


# ════════════════════════════════════════════════════════════════
# TAB 2 — Upload Voice
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
      <div style="font-family:'Lora',serif;font-size:1.2rem;font-weight:500;color:#2c2416;margin-bottom:0.3rem;">Upload a voice recording</div>
      <div style="font-size:0.83rem;color:#9c8a6e;">.wav · .mp3 · .ogg — speak naturally for the best results</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=['wav', 'mp3', 'ogg'], label_visibility="collapsed")
    if uploaded:
        st.audio(uploaded)
        col1, col2 = st.columns([1.3, 4])
        with col1:
            if st.button("Analyse →", type="primary", use_container_width=True, key="upload_btn"):
                st.session_state.voice_chat_history = []
                st.session_state.voice_conversation = []
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    audio, sr = sf.read(tmp_path)
                    if len(audio.shape) > 1: audio = audio.mean(axis=1)
                    audio = audio.astype(np.float32)
                    with st.spinner("Analysing your voice…"):
                        process_audio_and_respond(audio, sr)
                except Exception as e:
                    st.error(f"Error: {e}")
    if st.session_state.voice_emotion is not None:
        st.divider()
        render_voice_results_and_chat("voice_chat_input", "voice_clear")
    st.markdown("""
    <p style="color:#c4ae90;font-size:0.75rem;margin-top:2rem;text-align:center;">
      🌱 MindSpace is not a clinical tool — please speak with a professional if needed.
    </p>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — Record Voice
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
      <div style="font-family:'Lora',serif;font-size:1.2rem;font-weight:500;color:#2c2416;margin-bottom:0.3rem;">Record your voice</div>
      <div style="font-size:0.83rem;color:#9c8a6e;">Speak naturally — say how you're feeling in your own words</div>
    </div>
    """, unsafe_allow_html=True)
    recorded_audio = st.audio_input("", label_visibility="collapsed")
    if recorded_audio is not None:
        col1, col2 = st.columns([1.3, 4])
        with col1:
            if st.button("Analyse →", type="primary", use_container_width=True, key="record_btn"):
                st.session_state.voice_chat_history = []
                st.session_state.voice_conversation = []
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(recorded_audio.getvalue())
                        tmp_path = tmp.name
                    audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
                    audio = audio.astype(np.float32)
                    if len(audio) / sr < 1.0:
                        st.error("Recording too short — please record at least 2 seconds.")
                        st.stop()
                    if np.max(np.abs(audio)) < 0.001:
                        st.error("Recording appears silent — please check your microphone.")
                        st.stop()
                    with st.spinner("Analysing your voice…"):
                        process_audio_and_respond(audio, sr)
                except Exception as e:
                    st.exception(e)
    if st.session_state.voice_emotion is not None:
        st.divider()
        render_voice_results_and_chat("rec_chat_input", "rec_clear")
    st.markdown("""
    <p style="color:#c4ae90;font-size:0.75rem;margin-top:2rem;text-align:center;">
      🌱 MindSpace is not a clinical tool — please speak with a professional if needed.
    </p>""", unsafe_allow_html=True)