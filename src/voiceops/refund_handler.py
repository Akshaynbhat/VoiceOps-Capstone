
import re
from datetime import datetime
from pathlib import Path
import importlib.util

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "voiceops_business_rules", _here / "business_rules.py"
)
_business_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_business_rules)

def answer_refund_timeline(question, orders_df=None, holidays=()):
    q = str(question).lower()

    if not any(word in q for word in
               ["refund", "money back", "credited", "credit back"]):
        return None

    payment = next(
        (method for method in [
            "upi", "credit card", "debit card", "net banking",
            "wallet", "neft", "bank transfer", "card"
        ] if method in q),
        None
    )

    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    initiated_date = date_match.group(1) if date_match else None

    if not payment or not initiated_date:
        return (
            "I can help check the refund timeline. Please provide the "
            "payment method and refund initiation date (YYYY-MM-DD)."
        )

    try:
        result = _business_rules.calculate_refund_deadline(
            initiated_date=initiated_date,
            payment_method=payment,
            holidays=holidays
        )
        return str(result)
    except Exception:
        return (
            "I couldn't calculate the refund timeline from those details. "
            "Please confirm the payment method and initiation date."
        )
