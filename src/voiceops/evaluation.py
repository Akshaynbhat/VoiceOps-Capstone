"""Evaluation: Step 1 (prompt-only) vs Step 2 (RAG agent) on the 52 practice
questions, plus the 23 practice conversations (96 turns) for Step 3.

    python -m voiceops.evaluation --dataset <folder> [--asr] [--limit N] [--fresh]

Results are appended row by row to reports/eval/*.jsonl, so if you hit a Groq
rate limit you can simply re-run the same command and it resumes.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import business_rules as br
from . import config, llm

MONTH_RE = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|अक्टूबर|नवंबर|सितंबर|दिसंबर|अगस्त|जुलाई|जून)"
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "thirty": 30,
             "forty-nine": 49, "seventy-nine": 79, "ek": 1, "do": 2, "teen": 3, "saat": 7, "das": 10}


def normalise(text):
    """Lower-case and add canonical forms so equivalent facts match:
    native digits -> ASCII, any written date -> also YYYY-MM-DD, number words -> digits."""
    t = str(text).translate(str.maketrans("०१२३४५६७८९", "0123456789")).lower()
    t = t.replace("’", "'").replace("‘", "'")
    extra = []
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH_RE}\.?(?:\s*,?\s*(20\d\d))?", t):
        d = br.parse_date_text(m.group(0))
        if d:
            extra.append(d.isoformat())
    for m in re.finditer(rf"{MONTH_RE}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d\d))?", t):
        d = br.parse_date_text(m.group(0))
        if d:
            extra.append(d.isoformat())
    for w, n in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", t):
            extra.append(str(n))
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 10,000 -> 10000
    return t + " || " + " ".join(extra)


def evaluate_answer_text(answer, row):
    """Score one reply against must_include / must_not_include.
    Returns dict(content_ok, missing, forbidden)."""
    t = normalise(answer)
    missing, forbidden = [], []
    for fact in [f.strip() for f in str(row.get("must_include") or "").split(" ; ") if f.strip()]:
        alts = [a.strip().lower() for a in fact.split(" / ") if a.strip()]
        if not any(normalise(a).split(" || ")[0] in t for a in alts):
            missing.append(fact)
    for bad in [f.strip() for f in str(row.get("must_not_include") or "").split(" ; ") if f.strip()]:
        if bad.lower() in t.split(" || ")[0]:
            forbidden.append(bad)
    return {"content_ok": not missing and not forbidden, "missing": missing, "forbidden": forbidden}


def gold_ids(row):
    return [g.strip() for g in str(row.get("gold_evidence") or "").split(" ; ") if g.strip()]


def retrieval_scores(retrieved, gold):
    if not gold:
        return None, None
    hit = [g for g in gold if g in retrieved]
    return len(hit) / len(gold), bool(hit)


STEP1_SYSTEM = ("You are ZentroMart's AI customer support assistant. ZentroMart is an Indian online store selling "
                "groceries, electronics, fashion and home goods. Today is 20 October 2026. In your first reply, say you "
                "are an AI assistant. Help with orders, deliveries, returns, refunds and cancellations. Be polite and "
                "concise (2-3 sentences). Reply in the customer's language. You cannot take actions such as cancelling "
                "orders or issuing refunds. If you are not sure, say so and offer to connect a human colleague.")


def step1_answer(question):
    return llm.chat([{"role": "system", "content": STEP1_SYSTEM}, {"role": "user", "content": question}])


def _load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    return done


def _append(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_questions(agent, qs, outdir, use_asr, skip_step1):
    path = outdir / f"questions{'_asr' if use_asr else ''}.jsonl"
    done = _load_done(path)
    from .agent import ConversationState
    for i, (_, row) in enumerate(qs.iterrows(), 1):
        text = row["asr_text"] if use_asr else row["question"]
        acceptable = [d.strip() for d in str(row.get("acceptable_decisions") or row["expected_decision"]).split(" ; ")]
        for step in ("step1", "step2"):
            key = f"{row['id']}|{step}"
            if key in done or (step == "step1" and skip_step1):
                continue
            try:
                if step == "step1":
                    reply, rec = step1_answer(text), {"retrieved_sections": [], "decision": ""}
                else:
                    rec = agent.handle(text, ConversationState())
                    reply = rec["reply"]
            except Exception as e:
                print(f"  ! {key}: {e}\n  Stopping; re-run the same command later to resume.")
                return path
            score = evaluate_answer_text(reply, row)
            recall, hit = retrieval_scores(rec["retrieved_sections"], gold_ids(row))
            out = {"key": key, "id": row["id"], "step": step, "language": row["language"], "intent": row["intent"],
                   "question": text, "reply": reply, **score, "decision": rec.get("decision", ""),
                   "decision_ok": rec.get("decision", "") in acceptable if step == "step2" else None,
                   "acceptable": acceptable, "retrieved": rec["retrieved_sections"], "gold": gold_ids(row),
                   "recall": recall, "hit": hit}
            _append(path, out)
            done[key] = out
            print(f"[{i}/{len(qs)}] {key}: {'PASS' if score['content_ok'] else 'FAIL'} {score['missing'] or ''} {score['forbidden'] or ''}")
    return path


def run_conversations(agent, conv, outdir):
    path = outdir / "conversations.jsonl"
    done = _load_done(path)
    from .agent import ConversationState
    for cid, g in conv.groupby("conversation_id"):
        g = g.sort_values("turn", key=lambda s: s.astype(int))
        if all(f"{cid}|{t}" in done for t in g["turn"]):
            continue
        # replay the whole conversation (state must be rebuilt from turn 1)
        state, rows = ConversationState(), []
        try:
            for _, row in g.iterrows():
                rec = agent.handle(row["customer_message"], state)
                rows.append((row, rec))
        except Exception as e:
            print(f"  ! {cid}: {e}\n  Stopping; re-run the same command later to resume.")
            return path
        for row, rec in rows:
            score = evaluate_answer_text(rec["reply"], row)
            exp_tools = sorted(re.sub(r"\(.*", "", x.strip()) + ("+verify" if "verification_value" in x else "")
                               for x in str(row["expected_tool_calls"]).split(" ; ") if x.strip())
            got_tools = sorted(t["name"] + ("+verify" if "verification_value" in t["args"] else "")
                               for t in rec["tool_calls"])
            recall, hit = retrieval_scores(rec["retrieved_sections"], gold_ids(row))
            out = {"key": f"{cid}|{row['turn']}", "conversation_id": cid, "turn": int(row["turn"]),
                   "scenario": row["scenario"], "language": row["language"], "message": row["customer_message"],
                   "reply": rec["reply"], **score, "decision": rec["decision"],
                   "expected_decision": row["expected_decision"], "decision_ok": rec["decision"] == row["expected_decision"],
                   "tools_ok": exp_tools == got_tools, "expected_tools": exp_tools, "tools": got_tools,
                   "retrieved": rec["retrieved_sections"], "recall": recall, "hit": hit}
            _append(path, out)
            print(f"{out['key']}: content {'ok' if score['content_ok'] else 'FAIL'} | decision {'ok' if out['decision_ok'] else 'FAIL'}"
                  f" | tools {'ok' if out['tools_ok'] else 'FAIL'}")
    return path


def _pct(xs):
    xs = [x for x in xs if x is not None]
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def summarise(outdir):
    s = {}
    for name in ("questions", "questions_asr"):
        p = outdir / f"{name}.jsonl"
        if not p.exists():
            continue
        rows = list(_load_done(p).values())
        for step in ("step1", "step2"):
            r = [x for x in rows if x["step"] == step]
            if not r:
                continue
            d = {"n": len(r), "content_accuracy": _pct([x["content_ok"] for x in r])}
            if step == "step2":
                d.update(decision_accuracy=_pct([x["decision_ok"] for x in r]),
                         retrieval_recall=_pct([x["recall"] for x in r]), retrieval_hit=_pct([x["hit"] for x in r]))
            by_lang = defaultdict(list)
            for x in r:
                by_lang[x["language"]].append(x["content_ok"])
            d["by_language"] = {k: _pct(v) for k, v in by_lang.items()}
            d["failures"] = [{"id": x["id"], "intent": x["intent"], "missing": x["missing"], "forbidden": x["forbidden"]}
                             for x in r if not x["content_ok"]]
            s[f"{name}_{step}"] = d
    p = outdir / "conversations.jsonl"
    if p.exists():
        r = list(_load_done(p).values())
        s["conversations"] = {"turns": len(r), "content_accuracy": _pct([x["content_ok"] for x in r]),
                              "decision_accuracy": _pct([x["decision_ok"] for x in r]),
                              "tool_call_accuracy": _pct([x["tools_ok"] for x in r]),
                              "retrieval_hit": _pct([x["hit"] for x in r]),
                              "conversations_fully_correct": sum(
                                  all(x["content_ok"] and x["decision_ok"] and x["tools_ok"] for x in r if x["conversation_id"] == c)
                                  for c in {x["conversation_id"] for x in r}),
                              "n_conversations": len({x["conversation_id"] for x in r}),
                              "failures": [{"key": x["key"], "scenario": x["scenario"], "missing": x["missing"],
                                            "forbidden": x["forbidden"], "decision": x["decision"],
                                            "expected": x["expected_decision"], "tools_ok": x["tools_ok"]}
                                           for x in r if not (x["content_ok"] and x["decision_ok"] and x["tools_ok"])]}
    (outdir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    return s


def print_summary(s):
    print("\n================ SUMMARY ================")
    for k, v in s.items():
        if k == "conversations":
            print(f"Conversations ({v['turns']} turns): content {v['content_accuracy']}% | decision {v['decision_accuracy']}% | "
                  f"tool calls {v['tool_call_accuracy']}% | fully correct {v['conversations_fully_correct']}/{v['n_conversations']}")
        else:
            extra = (f" | decision {v.get('decision_accuracy')}% | retrieval recall {v.get('retrieval_recall')}% "
                     f"hit {v.get('retrieval_hit')}%") if "step2" in k else ""
            print(f"{k} (n={v['n']}): content {v['content_accuracy']}%{extra}  by language {v['by_language']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=str(config.ROOT / "reports" / "eval"))
    ap.add_argument("--asr", action="store_true", help="use asr_text instead of question (simulated voice input)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-step1", action="store_true")
    ap.add_argument("--skip-conversations", action="store_true")
    ap.add_argument("--only-conversations", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete previous results first")
    a = ap.parse_args(argv)
    if a.dataset:
        config.set_dataset_dir(a.dataset)
    from .agent import Agent
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.fresh:
        for p in outdir.glob("*.jsonl"):
            p.unlink()
    agent = Agent()
    qs = pd.read_csv(config.DATASET_DIR / "practice" / "questions.csv", dtype=str).fillna("")
    conv = pd.read_csv(config.DATASET_DIR / "practice" / "conversations.csv", dtype=str).fillna("")
    if a.limit:
        qs = qs.head(a.limit)
        conv = conv[conv.conversation_id.isin(conv.conversation_id.unique()[: max(1, a.limit // 4)])]
    if not a.only_conversations:
        run_questions(agent, qs, outdir, a.asr, a.skip_step1)
    if not a.skip_conversations:
        run_conversations(agent, conv, outdir)
    s = summarise(outdir)
    print_summary(s)
    print(f"\nDetailed results: {outdir}")


if __name__ == "__main__":
    sys.exit(main())
