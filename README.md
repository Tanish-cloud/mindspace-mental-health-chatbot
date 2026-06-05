# 🌿 MindSpace - Mental Wellness Companion

MindSpace is an AI-powered mental health companion designed to provide a safe, warm, and non-clinical space for users to share their feelings. Whether through text or voice, MindSpace actively listens, detects your emotional state, and responds with empathy and understanding.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)
![Groq](https://img.shields.io/badge/Groq-Llama%203-blue?style=for-the-badge)

## ✨ Key Features
- **💬 Text Chat with Emotion Detection**: Uses a fine-tuned NLP model to detect emotional states (Stress, Anxiety, Depression) and responds compassionately.
- **🎙️ Voice Analysis (Real-time & Uploads)**: Leverages a custom-trained Convolutional Neural Network (CNN) to analyze speech patterns, pitch, and acoustic features to accurately gauge emotional states.
- **🧠 Empathetic LLM Integration**: Uses Groq's high-speed inference with LLaMA-3 (70B) to generate human-like, non-clinical, and supportive conversational responses.
- **🎨 Premium UI/UX**: Custom-built responsive design in Streamlit, utilizing dynamic CSS to create a calming, accessible, and distraction-free environment.

## 🛠️ Technology Stack
- **Frontend & App Logic**: Streamlit, Python
- **Machine Learning (Speech)**: PyTorch, Librosa (MFCC, Chroma, Mel Spectrogram extraction)
- **Machine Learning (Text)**: Hugging Face `transformers` (Sequence Classification)
- **Language Model**: Groq API (LLaMA-3)
- **Deployment**: Hugging Face Spaces (Docker)

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Tanish-cloud/mindspace-mental-health-chatbot.git
cd mindspace-mental-health-chatbot
```

### 2. Install dependencies
Make sure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
You will need a Groq API key to run the LLM backend. 
```bash
# On Windows
set GROQ_API_KEY=your_api_key_here

# On macOS/Linux
export GROQ_API_KEY="your_api_key_here"
```

### 4. Run the application
```bash
streamlit run app_streamlit.py
```

## 🧠 Architecture Highlights
- **Speech Emotion CNN**: A custom architecture featuring multiple 1D Convolutional blocks with Batch Normalization, MaxPooling, and Dropout, terminating in a dense classifier. 
- **Audio Processing Pipeline**: Handles dynamic resampling and pads/truncates audio to extract a standardized feature vector of MFCCs, Chroma, Mel spectrograms, Zero Crossing Rate, and RMS energy.
- **System Prompts**: Highly engineered prompts ensure the LLM strictly avoids clinical diagnoses or therapeutic language, remaining entirely in the scope of a "caring friend."

## ⚠️ Disclaimer
MindSpace is **not** a clinical tool, diagnostic instrument, or replacement for professional therapy. It is designed purely as a conversational companion.
