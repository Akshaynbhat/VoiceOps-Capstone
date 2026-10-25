"""Build report.pdf (<= 4 pages) from reports/eval/summary.json.

    python make_report.py [--name "Your Name"] [--repo https://github.com/you/VoiceOps-Capstone]

Run the evaluation first (python -m voiceops.evaluation ...). Edit the text in
SECTIONS below if you want to add your own observations.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent


def pct(v):
    return "n/a" if v is None else f"{v}%"


def build(name, repo, out):
    sp = ROOT / "reports" / "eval" / "summary.json"
    s = json.loads(sp.read_text()) if sp.exists() else {}
    q1, q2 = s.get("questions_step1", {}), s.get("questions_step2", {})
    a1, a2 = s.get("questions_asr_step1", {}), s.get("questions_asr_step2", {})
    cv = s.get("conversations", {})

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12.2, spaceAfter=4)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=7, spaceAfter=3,
                        textColor=colors.HexColor("#1f3a5f"))
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
    P = lambda t, st=body: Paragraph(t.replace("₹", "Rs. "), st)  # noqa: E731

    def table(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.4), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d4")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4)]))
        return t

    el = [P("VoiceOps 2026: ZentroMart AI Support Agent", h1),
          P(f"{name} · Capstone report · {repo.replace('<', '&lt;').replace('>', '&gt;')}", small), Spacer(1, 6)]

    el += [P("1. What I built", h2),
           P("<b>Pipeline:</b> customer audio → Gnani STT (en-IN, ITN on) → agent (router → controller → retrieval + "
             "business rules → LLM reply) → Gnani TTS (timbre-v2.5). The LLM is Groq-hosted "
             "(<i>openai/gpt-oss-20b</i> for replies, <i>llama-3.1-8b-instant</i> as a JSON router). "
             "<font face='Courier'>python run.py &lt;audio_folder&gt; --dataset &lt;dataset&gt;</font> processes a folder "
             "and saves reply text, reply audio and the section IDs used for every file."),
           P("<b>Step 1 (prompting):</b> a system prompt setting the role (ZentroMart's AI assistant), limits (no actions, "
             "no invented policy, say when unsure) and style. No documents, so it has to guess."),
           P("<b>Step 2 (RAG):</b> every <font face='Courier'>## [SECTION-ID]</font> heading is one chunk (82 current "
             "sections), so the ID travels with the text and is reported back. Hybrid TF-IDF (word 1-2 grams + "
             "character 3-5 grams) with a synonym map from customer words to policy words (\"send it back\" → return "
             "window, \"nobody came\" → reverse pickup), plus an English search query written by the router, which "
             "lets Hindi/Hinglish questions find English policies. The superseded returns policy is excluded from "
             "search and only fetched when RET-3 says it still applies (electronics delivered before 1 August 2026). "
             "Products named in a question are matched to <font face='Courier'>products.csv</font> rows and linked "
             "to the policy for their return category. The prompt states the precedence rule (policy &gt; agent "
             "guide &gt; help &gt; FAQ &gt; marketing) so the FAQ's vague \"5 to 7 days\" never overrides REF-2."),
           P("<b>Deterministic business rules:</b> LLMs get date arithmetic wrong, so business days (Mon-Sat, skipping "
             "holidays.csv), refund ETAs (read from the REF-2 table), return deadlines (RET-2/3/8, calendar days), "
             "overdue refunds and Plus pickup-fee waivers are computed in Python and handed to the LLM as facts. "
             "The numbers are parsed from the KB tables at runtime; nothing is hard-coded per question."),
           P("<b>Step 3 (conversation):</b> a small state machine owns the parts that must never go wrong: order IDs "
             "parsed even when spoken in groups (\"8978 2837\", \"double four\"); verification with phone last-4 "
             "<i>or</i> pincode, one retry, then no details; tool calls (<font face='Courier'>lookup_order</font>, "
             "<font face='Courier'>create_ticket</font>); ESC-1 handovers (human request, legal threat, unauthorised "
             "activity, safety, repeat contact, refund &gt;3 business days overdue, damage &gt; ₹10,000). A ticket is "
             "mentioned only if <font face='Courier'>created=True</font>, and a post-filter removes any sentence "
             "claiming an action the code didn't take (\"I have cancelled…\"). Full chat history and the pending "
             "request are kept, so \"how long will the refund take?\" resolves to the same order and item."),
           P("<b>Step 4 (voice):</b> any input format is converted to 16 kHz mono WAV with ffmpeg and split into "
             "≤55 s clips (the REST limit is 60 s). Empty transcripts get \"Sorry, I didn't catch that\". Replies are "
             "written for the ear: 2-3 sentences, dates with month and year, order IDs and RRNs read in groups of "
             "four, ₹ read as rupees; Devanagari replies are spoken with hi-IN.")]

    el += [P("2. Step 1 vs Step 2 results", h2),
           P("Scored automatically on all 52 practice questions: a reply passes only if it contains every "
             "<i>must_include</i> fact (any listed alternative; dates and spoken numbers normalised) and none of the "
             "<i>must_not_include</i> phrases. Retrieval recall = share of gold section IDs that were retrieved.")]
    rows = [["", "Step 1: prompt only", "Step 2: RAG agent"],
            ["Content accuracy (typed question)", pct(q1.get("content_accuracy")), pct(q2.get("content_accuracy"))],
            ["Content accuracy (asr_text, simulated voice)", pct(a1.get("content_accuracy")), pct(a2.get("content_accuracy"))],
            ["Decision accuracy (answer/clarify/escalate/refuse)", "-", pct(q2.get("decision_accuracy"))],
            ["Retrieval recall / hit rate", "-", f"{pct(q2.get('retrieval_recall'))} / {pct(q2.get('retrieval_hit'))}"]]
    for lang in ("en", "hi-en", "hi"):
        rows.append([f"Content accuracy, language = {lang}", pct(q1.get("by_language", {}).get(lang)),
                     pct(q2.get("by_language", {}).get(lang))])
    el += [table(rows, [7.6 * cm, 4.4 * cm, 4.4 * cm]), Spacer(1, 5)]
    el += [P(f"<b>Conversations (Step 3):</b> {cv.get('turns', 'n/a')} turns in {cv.get('n_conversations', 'n/a')} "
             f"conversations: content {pct(cv.get('content_accuracy'))}, decision {pct(cv.get('decision_accuracy'))}, "
             f"tool calls {pct(cv.get('tool_call_accuracy'))}; {cv.get('conversations_fully_correct', 'n/a')} conversations "
             "fully correct on every turn.")]

    el += [P("3. What I learned from the comparison", h2),
           P("Prompt-only answers sound confident but are usually wrong on anything ZentroMart-specific: return "
             "windows per category, the ₹49 pickup fee and the Plus waiver, wallet refunds being instant, the "
             "festive-sale 30-day window. The model fills gaps with generic e-commerce norms (\"5-7 business days\", "
             "\"30-day returns\"), which is exactly what the <i>must_not_include</i> lists catch. Grounding fixed most of "
             "these; the remaining gains came from things retrieval alone doesn't solve: (a) date arithmetic, fixed by "
             "computing dates in Python; (b) conflicting documents, fixed by passing doc type/status with each chunk "
             "and stating the precedence rule; (c) Hindi questions against English documents, fixed by having the "
             "router write an English search query.")]

    fails = q2.get("failures", [])
    if fails:
        by_intent = Counter(f["intent"] for f in fails)
        ex = "; ".join(f"{f['id']} ({f['intent']}: missing {', '.join(f['missing'][:2]) or '-'}"
                       f"{'; forbidden ' + ', '.join(f['forbidden']) if f['forbidden'] else ''})" for f in fails[:6])
        el += [P("4. What went wrong", h2),
               P(f"Step 2 still failed {len(fails)} of {q2.get('n')} questions, by intent: "
                 + ", ".join(f"{k} ({v})" for k, v in by_intent.most_common()) + f". Examples: {ex}.")]
    else:
        el += [P("4. What went wrong", h2)]
    el += [P("Problems hit while building: the first notebook version chunked by fixed length, which split policy "
             "tables and lost section IDs; the superseded returns policy leaked old rules (10-day electronics, free "
             "pickup) into answers; dates like \"17 October 2026\" weren't parsed; pandas without "
             "<font face='Courier'>dtype=str</font> stripped leading zeros from IDs and pincodes; the demo UI hid "
             "agent errors behind a bare <font face='Courier'>except: pass</font>; and the router sometimes asked for "
             "an order ID for questions policy alone can answer (\"my order is on its way, can I still stop it?\"). "
             "Free-tier Groq rate limits made full evaluation runs slow, so the evaluator saves each result and resumes.")]

    el += [P("5. What I would fix next", h2),
           P("• Replace TF-IDF with multilingual embeddings (e.g. a bge-m3-style model) plus a reranker, so Hindi and "
             "other Indian languages retrieve directly instead of relying on the router's translation.<br/>"
             "• Use Gnani streaming STT/TTS over WebSocket for lower latency in live calls, and barge-in.<br/>"
             "• Confidence-based recovery for unclear audio: when the transcript has no order ID where one is expected, "
             "or STT confidence is low, read back what was heard and confirm before acting.<br/>"
             "• A cancel tool and grocery slot data (CAN-3) so the agent can act, not just explain.<br/>"
             "• An LLM-as-judge pass alongside the string-match scorer to catch correct paraphrases it misses.")]

    el += [P("6. Bonus work", h2),
           P("Hindi/Hinglish replies in the customer's script; spoken-number normalisation for ASR output "
             "(\"eight nine seven eight\", \"double four\"); evaluation on the noisy <i>asr_text</i> column to "
             "estimate voice robustness; long-audio chunking; offline pytest suite with the LLM stubbed (verification, "
             "retry lockout, tickets, business days, retrieval)."),
           Spacer(1, 4),
           P("All answers are generated by the agent; nothing was typed or edited by hand. Only Gnani is used for "
             "speech. Only the synthetic dataset was used.", small)]

    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                      topMargin=1.4 * cm, bottomMargin=1.4 * cm, title="VoiceOps 2026 report").build(el)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Your Name")
    ap.add_argument("--repo", default="https://github.com/<your-username>/VoiceOps-Capstone")
    ap.add_argument("--out", default=str(ROOT / "report.pdf"))
    a = ap.parse_args()
    build(a.name, a.repo, a.out)
