import requests
import json

# ==============================================================================
# CONFIGURATION - CHANGE YOUR MODEL HERE
# ==============================================================================
MODEL_NAME = "llama3.2"  # You can quickly change this to "llama3" or "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ==============================================================================
# 4. LLM Scam Analysis Engine (Ollama / Llama 3.2)
# ==============================================================================

class LLMAnalyzer:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        print(f"[*] Initializing LLM Analyzer with model: {self.model_name}")

    def analyze_scam_pattern(self, transcript_text):
        """
        Analyzes the transcript to identify potential scamming patterns.
        """
        if not transcript_text.strip():
            return {"risk_score": 0, "reasoning": "No text provided."}

        # System Prompt: Defining the AI's role as a Fraud Detection Expert
        system_context = (
            "You are a highly specialized Financial Fraud Detection AI. "
            "Analyze the following phone call transcript for scam patterns such as: "
            "1. Impersonation (Bank, Police, Government). "
            "2. High Urgency/Pressure (Threats of arrest, account freezing). "
            "3. Unusual Payment Requests (Crypto, Gift cards, Wire transfers). "
            "4. Personal Info Phishing (OTP, Password, Social Security).\n\n"
            "Return your analysis in JSON format ONLY with the following keys:\n"
            "- risk_score: (Integer 0-100)\n"
            "- detected_patterns: (List of specific red flags found)\n"
            "- warning_message: (A short, punchy warning for the user)\n"
            "- final_verdict: (Safe / Suspicious / High Risk)"
        )

        prompt = f"{system_context}\n\nTranscript: \"{transcript_text}\"\n\nJSON Output:"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"  # Forces Llama 3.2 to output valid JSON
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            
            # Parse the response from Ollama
            result_raw = response.json().get("response", "{}")
            analysis = json.loads(result_raw)
            
            return analysis

        except Exception as e:
            return {
                "risk_score": 50,
                "error": f"LLM Analysis failed: {str(e)}",
                "final_verdict": "Error"
            }

# ==============================================================================
# Quick Test Logic
# ==============================================================================

if __name__ == "__main__":
    # Test initialization
    analyzer = LLMAnalyzer()
    print("[+] LLM Engine Ready!")

    # Usage Example:
    test_text = "This is Officer John from the High Court. Your bank account is linked to money laundering. Please transfer your balance to our secure safety account immediately to avoid arrest."
    # result = analyzer.analyze_scam_pattern(test_text)
    # print(json.dumps(result, indent=4))