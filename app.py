import gradio as gr
import plotly.express as px
import pandas as pd
import numpy as np
import os

# Import your core modules
from core.emotion_engine import EmotionEngine
from core.sentinel_engine import SentinelEngine
from core.stt_engine import STTEngine
from core.llm_analyzer import LLMAnalyzer

# ==============================================================================
# INITIALIZATION
# ==============================================================================
print("[*] Booting VoxSentinel Defense System...")

emo_hw = EmotionEngine()
sentinel_hw = SentinelEngine()
stt_hw = STTEngine(model_size="small", device="cuda") 
llm_hw = LLMAnalyzer()

custom_css = """
@keyframes pulse-red {
    0% { background-color: rgba(255, 0, 0, 0.1); }
    50% { background-color: rgba(255, 0, 0, 0.4); }
    100% { background-color: rgba(255, 0, 0, 0.1); }
}
.danger-alert {
    animation: pulse-red 1.5s infinite;
    border: 3px solid #ff4b4b !important;
    border-radius: 15px !important;
    padding: 15px !important;
    text-align: center !important;
}
"""

# ==============================================================================
# PIPELINE LOGIC
# ==============================================================================

def process_audio(audio_path):
    if audio_path is None:
        return None, "<div>Waiting for Input...</div>", "N/A", {}, {}, "N/A", "Please provide audio."

    try:
        # 1. AI Clone Detection
        sentinel_report = sentinel_hw.detect_voice_clone(audio_path)
        is_ai = sentinel_report.get("is_fake", False)
        ai_conf = float(sentinel_report.get("confidence", 0.0))
        ai_label = {"AI Synthetic": ai_conf, "Real Human": 1.0 - ai_conf}

        # 2. Transcription
        stt_report = stt_hw.transcribe(audio_path)
        transcript = stt_report.get("text", "No speech detected.")

        # 3. Full Emotion Timeline Analysis
        # 这里调用 predict_timeline，它返回的是每一秒的最显著情绪
        timeline = emo_hw.predict_timeline(audio_path)
        
        if not timeline:
            empty_fig = px.line(title="Audio too short for timeline analysis")
            return empty_fig, "<div>Audio too short</div>", transcript, "0%", ai_label, "N/A", "Analysis incomplete."
            
        df = pd.DataFrame(timeline)
        
        # 为了看到“起伏”，我们绘制每段音频各情绪的置信度变化
        # 这里的 color_map 使用你在 emotion_engine.py 里定义的那个
        fig = px.line(df, x="time", y="confidence", color="emotion",
                      title="Raw Emotion Confidence Timeline",
                      markers=True,
                      color_discrete_map=emo_hw.color_map)
        
        fig.update_layout(
            template="plotly_dark", 
            yaxis_range=[0, 1.1], 
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend_title="Emotions"
        )

        # 获取主要情绪
        top_emotion = df['emotion'].mode()[0] if not df.empty else "Neutral"

        # 4. LLM Scam Analysis
        llm_report = llm_hw.analyze_scam_pattern(transcript)
        scam_score = int(llm_report.get("risk_score", 0))
        
        # 5. Global Risk
        total_risk = scam_score
        if is_ai and ai_conf > 0.8:
            total_risk = max(total_risk, 98)
        elif is_ai:
            total_risk = max(total_risk, 85)

        alert_status = "danger-alert" if total_risk > 70 else ""
        alert_html = f"""
        <div class='{alert_status}' style='padding: 10px; border-radius: 10px;'>
            <h1 style='color: {"#ff4b4b" if total_risk > 70 else "#00ff00"}; margin:0;'>
                GLOBAL RISK: {total_risk}%
            </h1>
            <p style='margin:0; color: white;'>{'CRITICAL: AI VOICE DETECTED' if is_ai else 'Status: Monitoring Content'}</p>
        </div>
        """

        warning_msg = f"### Verdict: {llm_report.get('final_verdict', 'Unknown')}\n\n"
        warning_msg += f"**Safety Message:** {llm_report.get('warning_message', 'N/A')}\n\n"
        warning_msg += "**Scam Patterns Found:**\n" + "\n".join([f"- {p}" for p in llm_report.get('detected_patterns', [])])

        return fig, alert_html, transcript, f"{scam_score}%", ai_label, top_emotion, warning_msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"<div>Error: {str(e)}</div>", "Error", {}, {}, "Error", str(e)

# ==============================================================================
# GRADIO INTERFACE
# ==============================================================================

my_theme = gr.themes.Soft(primary_hue="red")

with gr.Blocks(theme=my_theme, css=custom_css) as demo:
    gr.HTML("<h1 style='text-align: center; color: #ff4b4b;'>🛡️ VOX SENTINEL: COMMAND CENTER</h1>")
    
    with gr.Row():
        with gr.Column(scale=2):
            risk_html = gr.HTML("<div style='text-align:center;'>System Online. Awaiting input...</div>")
        with gr.Column(scale=1):
            top_emo_out = gr.Label(label="Predominant Emotion")
        with gr.Column(scale=1):
            ai_status_out = gr.Label(label="AI Detection (Sentinel)")
        with gr.Column(scale=1):
            scam_score_out = gr.Label(label="Scam Risk Score")

    gr.HTML("<hr>") 

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Audio Acquisition")
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath")
            run_btn = gr.Button("🚀 RUN FULL DIAGNOSTICS", variant="primary")
            transcript_out = gr.Textbox(placeholder="Waiting for audio transcript...", lines=10, label="Transcription")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📈 Raw Emotion Timeline"):
                    plot_out = gr.Plot()
                with gr.TabItem("🧠 Fraud Intelligence"):
                    llm_final_out = gr.Markdown("Intelligence report...")

    run_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[plot_out, risk_html, transcript_out, scam_score_out, ai_status_out, top_emo_out, llm_final_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)