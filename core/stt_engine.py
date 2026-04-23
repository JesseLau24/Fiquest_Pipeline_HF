import os
from faster_whisper import WhisperModel

# ==============================================================================
# 3. Speech-to-Text Engine (Faster-Whisper)
# ==============================================================================

class STTEngine:
    def __init__(self, model_size="small", device=None, compute_type="float32"):
        """
        model_size: "tiny", "base", "small", "medium", "large-v3"
        compute_type: "float16", "int8_float16", "int8" (use int8 for CPU)
        """
        # Determine device automatically
        if device is None:
            self.device = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
        else:
            self.device = device
            
        # Optimization: Use int8 on CPU to prevent lag
        if self.device == "cpu":
            compute_type = "int8"
            
        print(f"[*] Initializing Faster-Whisper ({model_size}) on: {self.device} ({compute_type})")
        
        # This will download the model to ~/.cache/huggingface/hub/ by default 
        # or you can specify 'download_root' to a local folder in your project.
        self.model = WhisperModel(
            model_size, 
            device=self.device, 
            compute_type=compute_type
        )

    def transcribe(self, audio_path):
        """
        Transcribes audio to text with timestamps.
        Returns: Full text and a list of segment dictionaries.
        """
        try:
            # beam_size=5 is standard for good accuracy
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            
            full_text = ""
            segments_list = []
            
            print(f"[*] Detected language: '{info.language}' with probability {info.language_probability:.2f}")

            for segment in segments:
                full_text += f"{segment.text} "
                segments_list.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip()
                })
            
            return {
                "text": full_text.strip(),
                "segments": segments_list,
                "language": info.language
            }

        except Exception as e:
            return {"error": str(e)}

# ==============================================================================
# Quick Test Logic
# ==============================================================================

if __name__ == "__main__":
    # Test initialization
    engine = STTEngine(model_size="small")
    print("[+] STT Engine Ready!")
    
    # Usage Example:
    # result = engine.transcribe("path_to_audio.wav")
    # print(f"Transcribed Text: {result['text']}")