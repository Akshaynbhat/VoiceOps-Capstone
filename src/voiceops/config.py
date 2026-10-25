"""Central configuration. Secrets come from environment variables, a .env file,
or Colab Secrets (GROQ_API_KEY / GNANI_API_KEY). Never hard-code keys."""
import os
import sys
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]


def secret(name, default=""):
    """Read a secret from the environment, falling back to Colab Secrets."""
    val = os.getenv(name, "")
    if val:
        return val
    try:
        from google.colab import userdata  # type: ignore
        val = userdata.get(name) or ""
        if val:
            os.environ[name] = val  # SDKs (e.g. Gnani) read from env
        return val
    except Exception:
        return default


# The dataset is NOT committed. Point at the extracted ZIP folder.
DATASET_DIR = Path(os.getenv("VOICEOPS_DATASET_DIR",
                             os.getenv("DATASET_DIR", ROOT / "Capstone_Project_Dataset_FTTT_2026")))

# Dataset reference date ("today"), kb/00_about_this_knowledge_base.md [KB-00.2]
TODAY = date(2026, 10, 20)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# router runs on a separate Groq model so it uses a separate rate-limit quota
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant")

STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-IN")
TTS_MODEL = os.getenv("TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.getenv("TTS_VOICE", "Pranav")

TOP_K = int(os.getenv("TOP_K", "5"))


def set_dataset_dir(path):
    global DATASET_DIR
    p = Path(path).resolve()
    # accept either the extracted folder or its parent
    if not (p / "kb").is_dir() and (p / "Capstone_Project_Dataset_FTTT_2026" / "kb").is_dir():
        p = p / "Capstone_Project_Dataset_FTTT_2026"
    if not (p / "kb").is_dir():
        raise FileNotFoundError(f"No kb/ folder under {p}. Pass the extracted dataset folder.")
    DATASET_DIR = p
    os.environ["VOICEOPS_DATASET_DIR"] = str(p)
    return p


def import_tools(dataset_dir=None):
    """Import lookup_order / create_ticket from the dataset's tools package."""
    ds = str(dataset_dir or DATASET_DIR)
    if ds not in sys.path:
        sys.path.insert(0, ds)
    from tools import create_ticket, lookup_order  # noqa: E402
    return lookup_order, create_ticket
