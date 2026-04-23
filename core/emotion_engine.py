# core/emotion_engine.py
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, AutoProcessor

# ==============================================================================
# 1. 核心架构定义 (与训练时保持严丝合缝)
# ==============================================================================

class MultiScaleAttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        mu = torch.sum(attn_weights * x, dim=1)
        delta = x - mu.unsqueeze(1)
        var = torch.sum(attn_weights * (delta ** 2), dim=1)
        std = torch.sqrt(torch.clamp(var, min=1e-9))
        return torch.cat([mu, std], dim=-1)

class VoxSentinelForEmotion(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.pooling = MultiScaleAttentivePooling(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            # 推理模式下 Dropout 自动失效
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_labels)
        )
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled_output = self.pooling(outputs.last_hidden_state)
        logits = self.classifier(pooled_output)
        return logits

# ==============================================================================
# 2. 情绪分析引擎封装
# ==============================================================================

class EmotionEngine:
    def __init__(self, repo_id="JesseHuang922/VoxSentinel-Emotion-Base", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] 正在初始化情绪引擎，加载模型至: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = VoxSentinelForEmotion.from_pretrained(repo_id).to(self.device)
        self.model.eval()
        
        # 情绪到颜色的映射，方便前端直接使用
        self.color_map = {
            "Angry": "#FF4B4B",     # 红色
            "Disgust": "#800080",   # 紫色
            "Fear": "#FF8C00",      # 橙色 (焦虑/恐惧)
            "Happy": "#00FF00",     # 绿色
            "Neutral": "#808080",   # 灰色
            "Sad": "#0000FF",       # 蓝色
            "Surprise": "#FFFF00"   # 黄色
        }

    def predict_full(self, audio_array):
        """对整段音频进行预测"""
        inputs = self.processor(audio_array, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
            
        label = self.model.config.id2label[pred_idx]
        return label, confidence

    def predict_timeline(self, audio_path, window_sec=2.0, stride_sec=1.0):
        """
        核心功能：生成情绪时间轴
        window_sec: 采样窗口时长
        stride_sec: 步长（重叠采样）
        """
        speech, sr = librosa.load(audio_path, sr=16000)
        duration = len(speech) / sr
        
        timeline_data = []
        
        # 按滑动窗口切割
        for start_f in range(0, len(speech), int(stride_sec * sr)):
            end_f = start_f + int(window_sec * sr)
            if end_f > len(speech):
                break
            
            chunk = speech[start_f:end_f]
            label, conf = self.predict_full(chunk)
            
            timestamp = start_f / sr
            timeline_data.append({
                "time": timestamp,
                "emotion": label,
                "confidence": conf,
                "color": self.color_map.get(label, "#FFFFFF")
            })
            
        return timeline_data

# ==============================================================================
# 3. 单元测试 (直接运行此文件可验证)
# ==============================================================================

if __name__ == "__main__":
    # 模拟测试逻辑
    engine = EmotionEngine()
    print("[+] 引擎初始化成功！")
    
    # 这里可以放一个测试用的 wav 文件路径
    # results = engine.predict_timeline("test.wav")
    # for r in results:
    #     print(f"时间: {r['time']:.2f}s | 情绪: {r['emotion']} | 置信度: {r['confidence']:.2%}")