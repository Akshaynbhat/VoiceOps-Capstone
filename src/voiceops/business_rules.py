"""Deterministic policy calculations.

The LLM is bad at date arithmetic, so dates are computed here and handed to
it as facts. The *numbers* (days per payment method, return windows) are read
from the knowledge-base tables at runtime (REF-2, RET-2, RETv3-2), not
hard-coded, so a policy edit flows through automatically.
"""
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS.update({"sept": 9, "अक्टूबर": 10, "नवंबर": 11, "सितंबर": 9, "दिसंबर": 12, "अगस्त": 8,
               "जुलाई": 7, "जून": 6})

METHOD_LABELS = {  # REF-2 row label -> canonical payment key
    "upi": "upi", "credit or debit card": "card", "net banking": "netbanking",
    "zentromart wallet for cash-on-delivery orders": "cod_wallet",
    "bank transfer (neft) for cash-on-delivery orders": "cod_bank",
    "zentromart wallet": "wallet",
}
FESTIVE_START, FESTIVE_END = date(2026, 10, 1), date(2026, 10, 15)   # RET-8
ELECTRONICS_CHANGE = date(2026, 8, 1)                               # RET-3


# ---------------------------------------------------------------- dates
def to_date(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    return datetime.fromisoformat(str(x)[:10]).date()


def fmt_date(d):
    """'27 October 2026' - month name and year, as STY-2 / the Gnani guide ask."""
    return f"{d.day} {d.strftime('%B')} {d.year}" if d else ""


def parse_date_text(text, default_year=2026):
    """Find a date written as 2026-10-17, 17/10/2026, '17 October 2026', 'October 17'."""
    t = text.lower()
    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"\b(\d{1,2})[/.](\d{1,2})[/.](20\d{2})\b", t)
    if m:
        return date(int(m[3]), int(m[2]), int(m[1]))
    month_pat = "|".join(sorted(map(re.escape, MONTHS), key=len, reverse=True))
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_pat})\.?(?:\s*,?\s*(20\d{{2}}))?", t)
    if m:
        return date(int(m[3] or default_year), MONTHS[m[2]], int(m[1]))
    m = re.search(rf"\b({month_pat})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d{{2}}))?", t)
    if m:
        return date(int(m[3] or default_year), MONTHS[m[1]], int(m[2]))
    return None


# ---------------------------------------------------------------- business days
_HOLIDAYS = None


def holidays(dataset_dir=None):
    global _HOLIDAYS
    if _HOLIDAYS is None or dataset_dir:
        df = pd.read_csv(Path(dataset_dir or config.DATASET_DIR) / "data" / "holidays.csv", dtype=str)
        _HOLIDAYS = {to_date(d) for d in df["date"]}
    return _HOLIDAYS


def is_business_day(d):
    return d.weekday() != 6 and d not in holidays()  # Mon-Sat, minus holidays (KB-00.3)


def add_business_days(start, n):
    """The n-th business day after `start` (the start day itself is day 0)."""
    d, count = to_date(start), 0
    while count < n:
        d += timedelta(days=1)
        if is_business_day(d):
            count += 1
    return d


def business_days_between(a, b):
    """Business days strictly after a, up to and including b (0 if b <= a)."""
    a, b = to_date(a), to_date(b)
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if is_business_day(d):
            n += 1
    return n


calculate_deadline = add_business_days  # name kept from the original scaffold


# ---------------------------------------------------------------- policy tables
def _kb_section(section_id, dataset_dir=None):
    for p in (Path(dataset_dir or config.DATASET_DIR) / "kb").glob("*.md"):
        txt = p.read_text(encoding="utf-8")
        m = re.search(rf"^## \[{re.escape(section_id)}\].*?(?=^## \[|\Z)", txt, flags=re.M | re.S)
        if m:
            return m.group(0)
    raise KeyError(section_id)


_REFUND_TABLE = None


def refund_table():
    """{method: (min_days, max_days)} from REF-2; (0, 0) means instant."""
    global _REFUND_TABLE
    if _REFUND_TABLE is None:
        table = {}
        for line in _kb_section("REF-2").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 2:
                continue
            key = METHOD_LABELS.get(cells[0].lower())
            if not key:
                continue
            nums = [int(x) for x in re.findall(r"\d+", cells[1])]
            table[key] = (0, 0) if "instant" in cells[1].lower() else (nums[0], nums[-1])
        _REFUND_TABLE = table
    return _REFUND_TABLE


def refund_eta(initiated_on, method):
    """Expected arrival window for a refund (REF-2, business days)."""
    method = normalise_method(method)
    lo, hi = refund_table()[method]
    start = to_date(initiated_on)
    if hi == 0:
        return {"method": method, "instant": True, "earliest": start, "latest": start}
    return {"method": method, "instant": False, "min_days": lo, "max_days": hi,
            "earliest": add_business_days(start, lo), "latest": add_business_days(start, hi)}


calculate_refund_deadline = refund_eta


def normalise_method(m):
    m = (m or "").lower().strip()
    aliases = {"credit card": "card", "debit card": "card", "card": "card", "upi": "upi",
               "netbanking": "netbanking", "net banking": "netbanking", "wallet": "wallet",
               "cod_wallet": "cod_wallet", "cod_bank": "cod_bank", "neft": "cod_bank",
               "bank transfer": "cod_bank"}
    return aliases.get(m, m)


def method_in_text(text):
    t = text.lower()
    for pat, key in [(r"cod|cash on delivery", None), (r"credit|debit|card", "card"), (r"\bupi\b", "upi"),
                     (r"net ?banking", "netbanking"), (r"neft|bank transfer", "cod_bank"),
                     (r"wallet|वॉलेट", "wallet")]:
        if re.search(pat, t) and key:
            return key
    return None


def _window_table(section_id):
    out = {}
    for line in _kb_section(section_id).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0] or cells[0].startswith("-") or cells[0] == "Category":
            continue
        days = re.search(r"(\d+)\s*days", cells[1])
        val = int(days.group(1)) if days else 0  # 0 = not returnable
        if len(cells) >= 3 and "`" in cells[2]:
            out[cells[2].strip("` ")] = val
        else:
            out[cells[0].lower()] = val
    return out


def return_window_days(category, delivered_on, placed_at=None):
    """Return window in calendar days, applying RET-2, RET-3 and RET-8."""
    cur = _window_table("RET-2")
    days = cur.get(category, 0)
    delivered, placed = to_date(delivered_on), to_date(placed_at)
    if category == "electronics" and delivered and delivered < ELECTRONICS_CHANGE:
        old = _window_table("RETv3-2")
        days = next((v for k, v in old.items() if k.startswith("electronics")), days)
    if category == "fashion" and placed and FESTIVE_START <= placed <= FESTIVE_END:
        days = max(days, 30)
    return days


# ---------------------------------------------------------------- facts for the LLM
def _g(o, k):
    v = o.get(k)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" else v


def order_facts(order, today=None):
    """Human-readable, pre-computed facts about a VERIFIED order (no phone/address)."""
    today = today or config.TODAY
    f, flags = [], {}
    f.append(f"Order {order['order_id']}: status {order.get('status')}, placed {fmt_date(to_date(order.get('placed_at')))}, "
             f"payment {order.get('payment_method')}, delivery type {order.get('delivery_type')}, total ₹{order.get('total_inr')}.")
    f.append(f"Customer first name {order.get('customer_first_name')}; ZentroMart Plus member: {order.get('is_plus')}.")
    for k, label in [("promised_date", "Promised delivery date"), ("shipped_on", "Shipped on"),
                     ("delivered_on", "Delivered on")]:
        if _g(order, k):
            f.append(f"{label}: {fmt_date(to_date(order[k]))}.")
    if order.get("status") not in ("delivered", "cancelled") and _g(order, "promised_date"):
        late = to_date(order["promised_date"]) < today
        f.append(f"As of today ({fmt_date(today)}) the order is {'LATE (past promised date)' if late else 'on time'}; "
                 f"delivery attempts so far: {order.get('delivery_attempts')}.")
    if _g(order, "delivered_on") and _g(order, "promised_date") and to_date(order["delivered_on"]) > to_date(order["promised_date"]):
        f.append("It was delivered after the promised date (late delivery).")
    if _g(order, "express_late_credit_inr"):
        f.append(f"Express late-delivery credit of ₹{order['express_late_credit_inr']} was credited to ZentroMart Wallet "
                 f"on {fmt_date(to_date(order['express_late_credit_on']))}.")
    if _g(order, "coupon_code"):
        f.append(f"Coupon {order['coupon_code']} gave ₹{order.get('coupon_discount_inr')} off (min order ₹{order.get('coupon_min_order_inr')}).")
    f.append(f"Earlier support contacts about this order in the last 7 days: {order.get('contacts_last_7_days') or 0}.")
    flags["repeat_contact"] = int(order.get("contacts_last_7_days") or 0) >= 2

    # items + return eligibility
    max_item = 0
    for it in order.get("items") or []:
        price = int(it.get("unit_price_inr") or 0)
        max_item = max(max_item, price)
        line = f"Item {it.get('line_no')}: {it.get('title')} (SKU {it.get('sku')}, category {it.get('return_category')}, ₹{price} x {it.get('qty')})."
        if _g(order, "delivered_on"):
            w = return_window_days(it.get("return_category"), order["delivered_on"], order.get("placed_at"))
            if w == 0:
                line += " Not returnable for change of mind (RET-5)."
            else:
                dl = to_date(order["delivered_on"]) + timedelta(days=w)
                line += (f" Return window {w} calendar days from delivery -> last day to return {fmt_date(dl)} "
                         f"({'still open' if today <= dl else 'CLOSED'} as of {fmt_date(today)}).")
                if it.get("return_category") == "fashion":
                    line += (" Change-of-mind reverse pickup fee: "
                             + ("waived (Plus member)." if order.get("is_plus") else "₹49 (waived if damaged/defective/wrong)."))
        f.append(line)

    # refunds
    flags["refund_overdue"] = False
    for kind, label in [("cancellation", "Cancellation refund"), ("rto", "Return-to-origin refund"),
                        ("return", "Return refund")]:
        init = _g(order, f"{kind}_refund_initiated_on")
        if not init:
            continue
        method = order.get(f"{kind}_refund_method")
        eta = refund_eta(init, method)
        s = (f"{label} of ₹{order.get(f'{kind}_refund_amount_inr')} initiated on {fmt_date(to_date(init))} "
             f"to {method}; RRN {order.get(f'{kind}_rrn')}. ")
        done = _g(order, f"{kind}_refund_completed_on")
        if done:
            s += f"Completed on {fmt_date(to_date(done))}."
        elif eta["instant"]:
            s += "Wallet refunds are instant (within 2 hours)."
        else:
            s += (f"Expected within {eta['min_days']} to {eta['max_days']} business days, i.e. by "
                  f"{fmt_date(eta['latest'])}.")
            over = business_days_between(eta["latest"], today)
            if over > 0:
                s += f" It is now {over} business days past that maximum."
            if over > 3:
                flags["refund_overdue"] = True
                s += " That is more than 3 business days overdue -> hand over to a human (ESC-1 rule 5)."
        f.append(s)

    # return milestones
    if _g(order, "return_status"):
        s = f"Return of {order.get('return_sku')} ({order.get('return_reason')}): status {order['return_status']}"
        for k, lab in [("return_requested_on", "requested"), ("return_pickup_on", "picked up"),
                       ("return_qc_passed_on", "QC passed")]:
            if _g(order, k):
                s += f", {lab} {fmt_date(to_date(order[k]))}"
        if _g(order, "return_pickup_on") and not _g(order, "return_qc_passed_on") and not _g(order, "return_refund_initiated_on"):
            if order.get("is_plus"):
                s += ". Plus member: refund should be initiated at pickup."
            else:
                s += (". Quality check (QC) not yet passed: refund is initiated after QC passes, within 2 business days "
                      f"of pickup (by {fmt_date(add_business_days(order['return_pickup_on'], 2))}).")
        if _g(order, "return_reverse_pickup_fee_inr"):
            s += f". Pickup fee deducted ₹{order['return_reverse_pickup_fee_inr']}"
        f.append(s + ".")

    # damage
    flags["high_value_damage"] = False
    if _g(order, "damage_sku"):
        dprice = next((int(i.get("unit_price_inr") or 0) for i in order.get("items") or [] if i.get("sku") == order["damage_sku"]), 0)
        f.append(f"Damage report: {order.get('damage_issue')} on {order['damage_sku']} (item value ₹{dprice}), reported "
                 f"{order.get('damage_hours_after_delivery')} hours after delivery, photos attached: "
                 f"{order.get('damage_photos_attached')}, status {order.get('damage_status')}.")
        flags["high_value_damage"] = dprice > 10000
    flags["max_item_value"] = max_item
    return f, flags


def message_facts(text, today=None):
    """Facts computable from a general question, e.g. refund started on a date via a method."""
    out = []
    d, m = parse_date_text(text), method_in_text(text)
    if d and m and re.search(r"refund|रिफ़ंड|रिफंड|paisa|money|credit|shuru|initiat|start", text, re.I):
        eta = refund_eta(d, m)
        if eta["instant"]:
            out.append(f"A {m} refund initiated on {fmt_date(d)} is instant (within 2 hours).")
        else:
            out.append(f"A {m} refund initiated on {fmt_date(d)} takes {eta['min_days']} to {eta['max_days']} business days "
                       f"(Mon-Sat, excluding holidays): earliest {fmt_date(eta['earliest'])}, latest {fmt_date(eta['latest'])}.")
    out.append(f"Today's date is {fmt_date(today or config.TODAY)}.")
    return out
