"""Step 4: Gnani-only speech (STT + TTS). No other speech service is used, even as a fallback."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config

MAX_CLIP_SECONDS = 55  # Gnani REST STT accepts clips up to 60 s
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".aac", ".opus", ".mp4"}

_stt = _tts = None


def _clients():
    global _stt, _tts
    if _stt is None:
        if not config.secret("GNANI_API_KEY"):
            raise RuntimeError("GNANI_API_KEY is not set (env, .env or Colab Secrets).")
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        _stt, _tts = GnaniSTTClient(), GnaniTTSClient()
    return _stt, _tts


# ------------------------------------------------------------------ audio prep
def _duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def to_wav_chunks(path, workdir):
    """Convert any input to 16 kHz mono WAV, split into <=55 s pieces."""
    path = Path(path)
    if not shutil.which("ffmpeg"):
        return [path]  # assume the file is already a supported format
    dur = _duration(path)
    n = max(1, int(dur // MAX_CLIP_SECONDS) + (1 if dur % MAX_CLIP_SECONDS > 0.5 else 0)) if dur else 1
    chunks = []
    for i in range(n):
        out = Path(workdir) / f"{path.stem}_{i}.wav"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ss", str(i * MAX_CLIP_SECONDS),
               "-t", str(MAX_CLIP_SECONDS), "-ac", "1", "-ar", "16000", str(out)]
        subprocess.run(cmd, check=True)
        if out.exists() and out.stat().st_size > 1000:
            chunks.append(out)
    return chunks or [path]


# ------------------------------------------------------------------ STT
def transcribe_audio(audio_path, language_code=None):
    """Gnani STT. Returns the transcript text ('' if nothing was heard)."""
    stt, _ = _clients()
    lang = language_code or config.STT_LANGUAGE
    texts = []
    with tempfile.TemporaryDirectory() as tmp:
        for chunk in to_wav_chunks(audio_path, tmp):
            try:  # ITN writes numbers as digits ("8978 2837"), which helps order IDs
                res = stt.transcribe(str(chunk), language_code=lang, format="transcribe")
            except Exception:
                res = stt.transcribe(str(chunk), language_code=lang)
            texts.append((res or {}).get("transcript", "") or "")
    return " ".join(t.strip() for t in texts if t and t.strip())


# ------------------------------------------------------------------ TTS
def text_for_speech(text):
    """Write replies for the ear: group long digit strings, read ₹ as rupees."""
    t = re.sub(r"\b(\d{4})(\d{4})(\d{4})?\b",
               lambda m: " ".join(g for g in m.groups() if g), text)   # order IDs / RRNs in 4s
    t = re.sub(r"₹\s?([\d,]+)", lambda m: f"{m.group(1)} rupees", t)
    t = t.replace("Rs.", "rupees").replace("&", " and ")
    return re.sub(r"\s+", " ", t).strip()


def tts_language(text):
    return "hi-IN" if re.search(r"[ऀ-ॿ]", text) else "en-IN"


def synthesize_reply(text, output_path, language_code=None, voice=None):
    """Gnani TTS -> WAV file. Returns the output path."""
    _, tts = _clients()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lang = language_code if language_code and language_code != "en-IN" else tts_language(text)
    tts.synthesize(text_for_speech(text), voice=voice or config.TTS_VOICE, model=config.TTS_MODEL,
                   language=lang, output_file=str(output_path))
    return str(output_path)
