"""Batch runner: folder of customer audio -> Gnani STT -> agent -> Gnani TTS.

    python run.py <audio_folder> --dataset <extracted_dataset_folder> [--output outputs]

For every audio file it writes, into the output folder:
    <name>.txt   the agent's reply text
    <name>.wav   the spoken reply (Gnani TTS)
    <name>.json  transcript, reply, retrieved section IDs, tool calls, decision, timings
plus results.csv / results.jsonl summarising all files.

Each top-level audio file is treated as its own call. Audio files inside a
sub-folder are treated as consecutive turns of ONE conversation (sorted by name).
"""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from voiceops import config  # noqa: E402

UNCLEAR_REPLY = "Sorry, I didn't catch that. Could you please say it again?"
ERROR_REPLY = "Sorry, I'm having trouble right now. Could you please try again in a moment?"


def collect(folder):
    from voiceops.speech import AUDIO_EXTS
    folder = Path(folder)
    sessions = [[p] for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    for sub in sorted(d for d in folder.iterdir() if d.is_dir()):
        turns = sorted(p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
        if turns:
            sessions.append(turns)
    return sessions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio_folder")
    ap.add_argument("--dataset", required=True, help="extracted Capstone_Project_Dataset_FTTT_2026 folder")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--stt-language", default=None, help="Gnani STT language code (default en-IN)")
    args = ap.parse_args()

    config.set_dataset_dir(args.dataset)
    from voiceops.agent import Agent, ConversationState
    from voiceops.speech import synthesize_reply, transcribe_audio

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    agent = Agent()
    sessions = collect(args.audio_folder)
    if not sessions:
        sys.exit(f"No audio files found in {args.audio_folder}")
    print(f"{sum(map(len, sessions))} audio file(s) in {len(sessions)} session(s)")

    rows = []
    for turns in sessions:
        state = ConversationState()
        for audio in turns:
            rel = audio.relative_to(args.audio_folder).with_suffix("")
            name = "__".join(rel.parts)
            rec = {"file": str(audio.relative_to(args.audio_folder)), "transcript": "", "reply": "",
                   "retrieved_sections": [], "tool_calls": [], "decision": "", "error": ""}
            t0 = time.time()
            try:
                rec["transcript"] = transcribe_audio(str(audio), language_code=args.stt_language)
                t1 = time.time()
                if not rec["transcript"].strip():
                    rec["reply"], rec["decision"] = UNCLEAR_REPLY, "clarify"
                else:
                    res = agent.handle(rec["transcript"], state)
                    rec.update({k: res[k] for k in ("reply", "retrieved_sections", "tool_calls", "decision", "ticket_id")})
                rec["stt_seconds"], rec["agent_seconds"] = round(t1 - t0, 2), round(time.time() - t1, 2)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["reply"] = rec["reply"] or ERROR_REPLY
                traceback.print_exc()
            try:
                t2 = time.time()
                synthesize_reply(rec["reply"], out / f"{name}.wav")
                rec["tts_seconds"] = round(time.time() - t2, 2)
                rec["reply_audio"] = f"{name}.wav"
            except Exception as e:
                rec["error"] += f" | TTS {type(e).__name__}: {e}"
                traceback.print_exc()
            (out / f"{name}.txt").write_text(rec["reply"], encoding="utf-8")
            (out / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(rec)
            print(f"[{rec['decision'] or 'error'}] {rec['file']}\n  heard: {rec['transcript']}\n  reply: {rec['reply']}\n"
                  f"  sections: {', '.join(rec['retrieved_sections'])}")

    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "transcript", "reply", "retrieved_sections", "decision", "reply_audio", "error"])
        for r in rows:
            w.writerow([r["file"], r["transcript"], r["reply"], " ; ".join(r["retrieved_sections"]),
                        r["decision"], r.get("reply_audio", ""), r["error"]])
    errors = sum(1 for r in rows if r["error"])
    print(f"\nDone: {len(rows)} file(s), {errors} with errors. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
