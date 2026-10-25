
import os, sys
from pathlib import Path
import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
DATASET = Path(os.environ.get(
    "VOICEOPS_DATASET_DIR",
    "/content/voiceops_dataset/Capstone_Project_Dataset_FTTT_2026"
))

from voiceops.speech import transcribe_audio, synthesize_reply
from voiceops.retrieval import load_kb, build_retriever, retrieve_kb
from voiceops.agent import ConversationState, handle_message
from voiceops.refund_handler import answer_refund_timeline

chunks = load_kb(str(DATASET))
vectorizer, matrix = build_retriever(chunks)

def load_csv(name):
    path = DATASET / "data" / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

orders = load_csv("orders.csv")
customers = load_csv("customers.csv")
holidays_df = load_csv("holidays.csv")
holidays = (
    holidays_df.iloc[:, 0].dropna().astype(str).tolist()
    if not holidays_df.empty else []
)
state = ConversationState()

def answer_question(message):
    results = retrieve_kb(message, chunks, vectorizer, matrix, top_k=5)
    evidence = "\n\n".join(
        str(item.get("text", "")) for item in results
    )
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GROQ_API_KEY") or ""
        except Exception:
            pass

    if api_key and evidence.strip():
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content":
                 "You are ZentroMart customer support. Answer using only "
                 "the supplied knowledge-base evidence. If evidence is "
                 "insufficient, say so and ask a focused question. Never "
                 "invent policy details.\n\nEVIDENCE:\n" + evidence},
                {"role": "user", "content": message}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content, evidence

    return (
        "I found these relevant policy details. Please configure GROQ_API_KEY "
        "in Colab Secrets for grounded natural-language answers.\n\n" + evidence,
        evidence
    )

def respond(audio, typed_message, language, history):
    global state
    message = (typed_message or "").strip()
    if audio is not None:
        message = transcribe_audio(audio, language_code=language)
    if not message:
        return history, "", None, "Please record audio or type a message."

    # Try the deterministic refund handler first; otherwise use the agent/RAG.
    try:
        refund = answer_refund_timeline(message, orders, holidays)
    except Exception:
        refund = None

    if refund:
        reply = str(refund)
        evidence = "Refund timeline handler"
    else:
        reply, evidence = answer_question(message)
        try:
            result = handle_message(
                message, state, orders_df=orders, customers_df=customers,
                rag_fn=lambda q: answer_question(q)[0]
            )
            if isinstance(result, str) and result.strip():
                reply = result
            elif isinstance(result, dict):
                reply = result.get("response") or result.get("reply") or reply
        except Exception:
            pass

    history = list(history or [])
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ])

    output_audio = None
    try:
        output_path = ROOT / "outputs" / "latest_reply.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_audio = synthesize_reply(
            reply, str(output_path), language_code=language
        )
    except Exception as exc:
        evidence += f"\n\nSpeech output unavailable: {exc}"

    return history, "", output_audio, evidence

with gr.Blocks(title="ZentroMart VoiceOps") as demo:
    gr.Markdown("# ZentroMart VoiceOps Support\nSpeak or type your customer-support question.")
    history = gr.State([])
    chat = gr.Chatbot(label="Conversation")
    audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath",
                        label="Speak or upload audio")
    message = gr.Textbox(label="Or type your message",
                         placeholder="Where is my order?")
    language = gr.Dropdown(
        choices=["en-IN", "hi-IN", "kn-IN", "ta-IN", "te-IN"],
        value="en-IN", label="Speech language"
    )
    submit = gr.Button("Send", variant="primary")
    clear = gr.Button("Clear conversation")
    audio_out = gr.Audio(label="Spoken reply", type="filepath")
    evidence = gr.Textbox(label="Retrieved evidence / notes", lines=5)

    submit.click(
        respond,
        inputs=[audio_in, message, language, history],
        outputs=[chat, message, audio_out, evidence]
    ).then(lambda h: h, inputs=[chat], outputs=[history])

    clear.click(
        lambda: ([], [], None, ""),
        outputs=[chat, history, audio_out, evidence]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=False)
