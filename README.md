# 🛡️ VoxSentinel: AI-Powered Anti-Fraud Defense Tower

**VoxSentinel** is an industrial-grade multi-modal forensic system designed for **FinQuest 2026**. It provides a robust defense against financial fraud by combining Deepfake detection, Emotion analysis, and LLM-driven pattern recognition to identify malicious intent in real-time voice communications.

---

## 🚀 Key Innovations

### 1. MSAP Forensic Head (Sentinel-Base)
Unlike traditional audio classifiers, our **Sentinel** engine utilizes **Multi-Scale Attentive Pooling (MSAP)**. By fusing weighted Mean ($\mu$) and weighted Standard Deviation ($\sigma$), it captures the rigid "neural textures" of AI vocoders, achieving a **99.56% Accuracy** in detecting voice clones.

### Hugging Face Repos for the two models:

For AI Detection:
https://huggingface.co/JesseHuang922/VoxSentinel-Base

For Emition Detection:
https://huggingface.co/JesseHuang922/VoxSentinel-Emotion-Base


### 2. Multi-Dimensional Risk Scoring
VoxSentinel doesn't just transcribe; it analyzes the "How" and "What":
* **Acoustic Fingerprinting:** Detects AI-synthesized speech artifacts.
* **Psychological Profiling:** Tracks user stress levels and emotional fluctuations (Fear, Anger, Neutral) via a temporal heatmap.
* **Cognitive Analysis:** Uses **Llama 3.2** to identify social engineering tactics (Authority Impersonation, Urgency, Unusual Payment Requests).

---

## 🛠️ System Architecture

The system is built on a decoupled, modular pipeline:

* **Sentinel Engine:** Fine-tuned `Wav2Vec2` with an MSAP head for Deepfake detection.
* **Emotion Engine:** `Wav2Vec2-Base` optimized for 7-core emotional state classification.
* **STT Engine:** `Faster-Whisper` (C2Translate) for high-speed, localized transcription.
* **Brain (LLM):** `Llama 3.2` running locally via **Ollama** for private, forensic text analysis.
* **Interface:** A high-impact **Gradio** Command Center with real-time Plotly visualizations.

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU (GTX 1660Ti or higher recommended)
- [Ollama](https://ollama.ai/) installed and running

### 1. Clone the Repository
```bash
git clone [https://github.com/JesseLau24/Finquest_Wav2Vec_Emo.git](https://github.com/JesseLau24/Finquest_Wav2Vec_Emo.git)
cd Finquest_Wav2Vec_Emo
```

### 2. Install Dependencies
```bash
pip install torch transformers librosa faster-whisper gradio plotly pandas requests
```

### 3. Pull the LLM
```bash
ollama pull llama3.2
```

### 🚦 UsageLaunch the Anti-Fraud Command Center:

```bash
python app.py
```

Once running, access the local dashboard at http://localhost:7860. 

Upload an audio file or use the microphone to start a full forensic scan.

Something like this would show:

![alt text](<Screenshot from 2026-04-23 22-57-48.png>)


Special thanks to the open-source community for providing the foundational models used in this defense taskforce.