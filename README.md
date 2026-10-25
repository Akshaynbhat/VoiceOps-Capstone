# VoiceOps — ZentroMart AI Support Assistant

A grounded customer-support agent with retrieval, conversation handling,
business rules, and Gnani speech-to-text/text-to-speech.

## Setup
Install requirements, configure environment variables, then run:
python run.py <audio_folder> --dataset <extracted_dataset_path>

The supplied capstone dataset is external and must not be committed.

## Acceptance checklist
- [ ] Canonical KB section IDs preserved
- [ ] Prompt-only vs RAG evaluated on all 52 practice questions
- [ ] Conversation tests run against the 96 supplied turns
- [ ] Gnani used for both STT and TTS
- [ ] Batch runner saves reply text, audio, and used chunk IDs
- [ ] Four-page report and two-minute demo completed
