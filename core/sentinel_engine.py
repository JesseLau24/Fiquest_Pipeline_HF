# core/sentinel_engine.py
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, AutoProcessor

# ==============================================================================
# 1. MSAP Architecture (Multi-Scale Attentive Pooling)
# ==============================================================================

class MultiScaleAttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128), 
            self.Tanh(), 
            nn.Linear(128, 1)
        )

    @staticmethod
    def Tanh():
        return nn.Tanh()

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        w = torch.softmax(self.attention(x), dim=1)
        mu = torch.sum(w * x, dim=1)
        
        # Robust variance calculation for "acoustic fingerprinting"
        delta = x - mu.unsqueeze(1)
        var = torch.sum(w * (delta**2), dim=1)
        std = torch.sqrt(torch.clamp(var, min=1e-9))
        return torch.cat([mu, std], dim=-1) # (batch, hidden_size * 2)

class SentinelForSyntheticDetection(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.pooling = MultiScaleAttentivePooling(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 512), 
            nn.ReLU(), 
            nn.BatchNorm1d(512), 
            nn.Dropout(0.3), 
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, 2)
        )
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled_output = self.pooling(outputs.last_hidden_state)
        return self.classifier(pooled_output)

# ==============================================================================
# 2. Sentinel Forensic Engine
# ==============================================================================

class SentinelEngine:
    def __init__(self, repo_id="JesseHuang922/VoxSentinel-Base", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing Sentinel (Deepfake Detection) on: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = SentinelForSyntheticDetection.from_pretrained(repo_id).to(self.device)
        self.model.eval()
        
        # Label mapping: 0 -> Synthetic/Fake, 1 -> Authentic/Real
        self.labels = {0: "Fake", 1: "Real"}

    def detect_voice_clone(self, audio_input):
        """
        Forensic analysis of audio. 
        Input: can be a file path (str) or a pre-loaded numpy array.
        """
        try:
            # 1. Load and Resample if it's a file path
            if isinstance(audio_input, str):
                wav, _ = librosa.load(audio_input, sr=16000, mono=True)
            else:
                wav = audio_input
            
            # 2. Voice Activity Detection / Trim silence
            wav, _ = librosa.effects.trim(wav, top_db=20)

            # 3. Preprocess
            inputs = self.processor(wav, return_tensors="pt", sampling_rate=16000)
            input_values = inputs.input_values.to(self.device)

            # 4. Inference
            with torch.no_grad():
                logits = self.model(input_values)
                probs = torch.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_idx].item()

            result = self.labels[pred_idx]
            
            return {
                "is_fake": True if pred_idx == 0 else False,
                "label": result,
                "confidence": confidence,
                "summary": f"{result} ({confidence:.2%} confidence)"
            }

        except Exception as e:
            return {"error": str(e)}

# ==============================================================================
# 3. Quick Test Logic
# ==============================================================================

if __name__ == "__main__":
    # Test initialization
    engine = SentinelEngine()
    print("[+] Sentinel Engine Ready!")
    
    # Usage Example:
    # report = engine.detect_voice_clone("path_to_audio.wav")
    # print(f"Deepfake Analysis: {report['summary']}")