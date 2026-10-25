"""Offline tests (no API keys needed). The LLM is stubbed; tools, rules and
retrieval run for real against the dataset.

    VOICEOPS_DATASET_DIR=<dataset folder> pytest -q
"""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceops import business_rules as br  # noqa: E402
from voiceops import config, llm  # noqa: E402

pytestmark = pytest.mark.skipif(not (config.DATASET_DIR / "kb").is_dir(),
                                reason="set VOICEOPS_DATASET_DIR to the extracted dataset")


@pytest.fixture(scope="module")
def agent():
    llm.chat = lambda messages, **k: "Stub reply."

    def no_router(*a, **k):
        raise RuntimeError("offline")
    llm.chat_json = no_router
    from voiceops.agent import Agent
    return Agent(use_llm_router=False)


def test_business_days_skip_sunday_and_holiday():
    # 17 Oct 2026 is a Saturday; 18 Oct Sunday and 19 Oct holiday are skipped
    assert br.add_business_days(date(2026, 10, 17), 1) == date(2026, 10, 20)
    assert br.refund_eta("2026-10-17", "card")["latest"] == date(2026, 10, 27)


def test_refund_table_read_from_kb():
    t = br.refund_table()
    assert t["upi"] == (1, 3) and t["wallet"] == (0, 0) and t["card"] == (5, 7)


def test_return_windows():
    assert br.return_window_days("fashion", "2026-10-14", "2026-10-06T10:00") == 30   # RET-8 festive
    assert br.return_window_days("fashion", "2026-10-12", "2026-09-30T10:00") == 15
    assert br.return_window_days("electronics", "2026-07-20") == 10                   # RET-3 old window
    assert br.return_window_days("electronics", "2026-10-10") == 7
    assert br.return_window_days("grocery", "2026-10-10") == 0


def test_date_parsing():
    assert br.parse_date_text("refund 17 October 2026 ko shuru hua") == date(2026, 10, 17)
    assert br.parse_date_text("started on 2026-10-14") == date(2026, 10, 14)


def test_order_id_parsing():
    from voiceops.agent import extract_numbers, normalize_spoken_numbers
    assert extract_numbers("Order number hai 8978 2837.") == ("89782837", None)
    assert extract_numbers(normalize_spoken_numbers("eight nine seven eight two eight three seven")) == ("89782837", None)
    assert extract_numbers("Last four digits 4533 hain.") == (None, "4533")
    assert extract_numbers(normalize_spoken_numbers("double four five three")) == (None, "4453")


def test_retrieval_finds_refund_table():
    from voiceops.retrieval import Retriever
    r = Retriever(config.DATASET_DIR)
    ids = [c.section_id for c, _ in r.search("how long does a UPI refund take", 5)]
    assert "REF-2" in ids
    assert all("v3" not in i for i in ids)  # superseded policy is not searched


def test_verification_flow_and_ticket(agent):
    from voiceops.agent import ConversationState
    st = ConversationState()
    r1 = agent.handle("Hello, mujhe order ke baare mein help chahiye.", st)
    assert r1["decision"] == "clarify" and "AI" in r1["reply"]
    r2 = agent.handle("Order number hai 8978 2837.", st)
    assert r2["action"] == "ask_verification" and r2["tool_calls"][0]["name"] == "lookup_order"
    r3 = agent.handle("Last four digits 4533 hain.", st)
    assert st.verified and r3["tool_calls"][0]["verified"] is True
    r4 = agent.handle("TV damage report kiya tha. Abhi replacement approve kar do na?", st)
    assert r4["decision"] == "escalate" and r4["ticket_id"]


def test_failed_verification_shares_nothing(agent):
    from voiceops.agent import ConversationState
    st = ConversationState()
    agent.handle("Hi, I need help with my order.", st)
    agent.handle("Order 7746 1439", st)
    assert agent.handle("1266", st)["action"] == "retry_verification"
    r = agent.handle("1266", st)
    assert r["action"] == "verification_failed" and not st.verified and st.order is None


def test_human_request_creates_ticket(agent):
    from voiceops.agent import ConversationState
    r = agent.handle("Can I just speak to a person?", ConversationState())
    assert r["decision"] == "escalate" and r["ticket_id"].startswith("TKT")
