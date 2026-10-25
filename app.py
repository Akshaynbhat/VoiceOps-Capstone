"""Gradio demo: speak (or type) -> Gnani STT -> agent -> Gnani TTS. Use this to record the 2-minute demo.

    python app.py --dataset <extracted_dataset_folder>
"""
import argparse
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from voiceops import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=None)
ap.add_argument("--no-share", action="store_true")
args, _ = ap.parse_known_args()
if args.dataset:
    config.set_dataset_dir(args.dataset)

from voiceops.agent import Agent, ConversationState  # noqa: E402
from voiceops.speech import synthesize_reply, transcribe_audio  # noqa: E402

agent = Agent()


def load_wav(path):
    """WAV file -> (sample_rate, numpy array) for gr.Audio(type='numpy')."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    data = np.frombuffer(frames, dtype=dtype)
    if ch > 1:
        data = data.reshape(-1, ch)
    return sr, data


def respond(audio, typed, stt_lang, chat, state):
    state = state or ConversationState()
    msg, note = (typed or "").strip(), ""
    if audio is not None:
        try:
            msg = transcribe_audio(audio, language_code=stt_lang)
        except Exception as e:
            note = f"STT error: {e}"
    if not msg:
        reply = "Sorry, I didn't catch that. Could you please say it again?"
        res = {"retrieved_sections": [], "tool_calls": [], "decision": "clarify"}
    else:
        res = agent.handle(msg, state)
        reply = res["reply"]
    chat = list(chat or []) + [{"role": "user", "content": msg or "(unclear audio)"},
                               {"role": "assistant", "content": reply}]
    out = None
    try:
        path = synthesize_reply(reply, ROOT / "outputs" / f"reply_{int(time.time())}.wav")
        out = load_wav(path)   # send audio as data, so it plays through proxies/share links
    except Exception as e:
        note += f" TTS error: {e}"
    info = (f"Decision: {res['decision']}\nSections used: {', '.join(res['retrieved_sections'])}\n"
            f"Tool calls: {res['tool_calls']}\n{note}").strip()
    return chat, state, None, "", out, info


with gr.Blocks(title="ZentroMart VoiceOps") as demo:
    gr.Markdown("# ZentroMart VoiceOps\nAI support assistant · Gnani STT/TTS · grounded in ZentroMart policies")
    state = gr.State(None)
    try:
        chat = gr.Chatbot(type="messages", label="Conversation", height=380)
    except TypeError:  # Gradio 6 removed `type` (messages format is the default)
        chat = gr.Chatbot(label="Conversation", height=380)
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak")
        audio_out = gr.Audio(label="Spoken reply (press ▶ if it doesn't auto-play)", type="numpy", autoplay=True)
    typed = gr.Textbox(label="…or type", placeholder="My order hasn't arrived. Can I cancel it?")
    stt_lang = gr.Dropdown(["en-IN", "hi-IN", "kn-IN", "ta-IN", "te-IN", "mr-IN", "bn-IN", "gu-IN"],
                           value=config.STT_LANGUAGE, label="Speech language")
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        reset = gr.Button("New conversation")
    info = gr.Textbox(label="Behind the scenes", lines=4)
    send.click(respond, [audio_in, typed, stt_lang, chat, state], [chat, state, audio_in, typed, audio_out, info])
    typed.submit(respond, [audio_in, typed, stt_lang, chat, state], [chat, state, audio_in, typed, audio_out, info])
    reset.click(lambda: ([], None, None, "", None, ""), outputs=[chat, state, audio_in, typed, audio_out, info])

if __name__ == "__main__":
    demo.launch(share=not args.no_share)
