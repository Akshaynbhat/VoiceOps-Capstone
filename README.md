# VoiceOps: ZentroMart AI Support Agent

A voice customer-support agent for ZentroMart (the fictional store in the VoiceOps 2026 capstone).
It answers from company documents (RAG), runs an identity-checked support conversation with tools, and takes
calls through **Gnani STT/TTS**.

```
customer audio → Gnani STT → agent (router → controller → retrieval + business rules → LLM) → Gnani TTS → spoken reply
```

| Step | Where |
| --- | --- |
| 1 · Prompt-only LLM | `src/voiceops/evaluation.py` (`STEP1_SYSTEM`) |
| 2 · RAG over the KB, section IDs kept | `src/voiceops/retrieval.py` |
| Deterministic dates and timelines (business days, refunds, return windows) | `src/voiceops/business_rules.py` |
| 3 · Conversation, verification, tools, handover | `src/voiceops/agent.py` |
| 4 · Gnani speech | `src/voiceops/speech.py` |
| Batch run command | `run.py` |
| Demo UI | `app.py` |

## Setup

```bash
git clone https://github.com/your-akshaynbhat/VoiceOps-Capstone.git
cd VoiceOps-Capstone
pip install -r requirements.txt          # needs ffmpeg on PATH for webm/mp3/m4a input
cp .env.example .env                     # then fill in GROQ_API_KEY and GNANI_API_KEY
```

- **Keys:** `GROQ_API_KEY` (LLM) and `GNANI_API_KEY` (speech) are read from the environment, from `.env`, or from Colab Secrets. No keys are in the repo.
- **Dataset:** not committed. Download `Capstone_Project_Dataset_FTTT_2026.zip`, extract it, and pass the folder with `--dataset`.
- **Colab:** open `VoiceOps_Colab.ipynb`, which runs every step below in order.

## Run command (batch, used for testing)

```bash
python run.py <audio_folder> --dataset <path/to/Capstone_Project_Dataset_FTTT_2026> --output outputs
```

For each audio file (`.wav .mp3 .m4a .webm .ogg .flac`), it saves the following in `outputs/`:

- `<name>.txt`: the agent's reply text
- `<name>.wav`: the spoken reply (Gnani TTS, `timbre-v2.5`)
- `<name>.json`: transcript, reply, **retrieved section IDs**, tool calls, decision and timings
- `results.csv` and `results.jsonl`: a summary of all files

Each audio file at the top level of the folder is its own call. Audio files inside a sub-folder are treated as the
turns of one conversation, in filename order. Empty or unclear audio gets the reply "Sorry, I didn't catch that".
An error on one file never stops the batch.

## Evaluation (Step 1 vs Step 2, plus conversations)

```bash
PYTHONPATH=src python -m voiceops.evaluation --dataset <dataset>          # 52 questions × 2 steps + 96 turns
PYTHONPATH=src python -m voiceops.evaluation --dataset <dataset> --asr    # same questions using the noisy asr_text
python make_report.py --name "Your Name" --repo <github-url>              # writes report.pdf
```

Each result is saved as soon as it's computed, so if Groq rate-limits you, re-run the same command and it resumes.
Results go to `reports/eval/` (`summary.json` plus per-row `.jsonl` files).

## Tests (offline, no keys)

```bash
VOICEOPS_DATASET_DIR=<dataset> pytest -q
```

## Demo UI

```bash
python app.py --dataset <dataset>        # Gradio: speak or type; shows the sections and tool calls behind each reply
```

## Design notes

- **Grounding:** one chunk per `## [SECTION-ID]` section. Hybrid TF-IDF (word + character n-grams), a synonym map
  (customer words → policy words), and an English search query from the router, so Hindi and Hinglish questions find
  English policies. The superseded returns policy is not searched; it is used only where RET-3 says it still applies.
  Document precedence: policy > agent guide > help > FAQ > marketing.
- **No LLM date arithmetic:** refund ETAs, return deadlines and overdue checks are computed in Python from the KB
  tables and `holidays.csv` (reference date 20 October 2026), then given to the LLM as facts.
- **Safety by construction:** verification (phone last 4 or pincode, one retry), tool calls and ESC-1 handovers
  are decided in code, not by the LLM. A ticket is mentioned only when `create_ticket` returns `created=True`.
  Claims of actions that never happened are filtered out of replies.
- **Speech:** Gnani only, with no fallback service. Clips over 55 s are split. Replies are written for the ear:
  dates include the month name and year, and IDs are read in groups of 4.

## Submission contents

`README.md`, `.env.example`, `requirements.txt`, `run.py`, `src/`, `report.pdf`, `demo.mp4`, `sample_outputs/`


🎥 **Demo Video:** The demo video is included in this repository as [`demo.mp4`](./demo.mp4).  
🔗 **Video Link:** [Watch the VoiceOps Demo](https://github.com/Akshaynbhat/VoiceOps-Capstone/blob/main/demo.mp4) — a separate link is also provided for convenient access.
