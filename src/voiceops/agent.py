"""Step 3: the conversational agent.

Design: a small deterministic controller owns everything that must never go
wrong (verification, one retry, when to call tools, ticket creation, never
claiming an action that didn't happen). The LLM is used for two things only:
  1. a router that labels the customer's message (JSON), and
  2. the reply itself, written ONLY from retrieved KB sections, verified order
     facts and dates computed in business_rules.py.
"""
import re
from dataclasses import dataclass, field

from . import business_rules as br
from . import config, llm
from .retrieval import Retriever, render_product

# ------------------------------------------------------------------ text utils
_NUM_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
              "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
              "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4", "paanch": "5", "panch": "5",
              "chhe": "6", "che": "6", "saat": "7", "aath": "8", "nau": "9", "shunya": "0"}
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DIGIT_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|\d)"


def normalize_spoken_numbers(text):
    """'eight nine seven eight double two' -> '8 9 7 8 2 2'. Only converts runs of
    3+ number words so normal words ("do", "one") are left alone."""
    t = text.translate(_DEV_DIGITS)

    def rep_multi(m):
        mult = {"double": 2, "triple": 3}[m.group(1).lower()]
        w = m.group(2).lower()
        return " ".join([_NUM_WORDS.get(w, w)] * mult)
    t = re.sub(rf"\b(double|triple)\s+({_DIGIT_WORD}|ek|do|teen|char|chaar|paanch|saat|aath|nau)\b",
               rep_multi, t, flags=re.I)
    words = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
    run = re.compile(rf"\b(?:(?:{words}|\d+)[\s,-]+){{2,}}(?:{words}|\d+)\b", re.I)

    def rep_run(m):
        toks = re.split(r"[\s,-]+", m.group(0))
        spelled = [tk for tk in toks if not tk.isdigit()]
        if len(spelled) < 2 and not all(tk.isdigit() for tk in toks):
            return m.group(0)
        return " ".join(_NUM_WORDS.get(tk.lower(), tk) for tk in toks)
    return run.sub(rep_run, t)


def _digit_runs(text):
    """Groups of digit tokens separated only by spaces/hyphens."""
    runs, cur, last_end = [], [], None
    for m in re.finditer(r"\d+", text):
        if cur and re.fullmatch(r"[\s-]*", text[last_end:m.start()]):
            cur.append(m.group(0))
        else:
            if cur:
                runs.append(cur)
            cur = [m.group(0)]
        last_end = m.end()
    if cur:
        runs.append(cur)
    return runs


def extract_numbers(text):
    """Return (order_id or None, verification candidate or None)."""
    oid, ver = None, None
    for run in _digit_runs(text):
        joined = "".join(run)
        if oid is None:
            acc = ""
            for i, g in enumerate(run):
                acc += g
                if len(acc) == 8:
                    oid = acc
                    rest = "".join(run[i + 1:])
                    if len(rest) in (4, 6):
                        ver = rest
                    break
                if len(acc) > 8:
                    break
            if oid:
                continue
        if ver is None and len(joined) in (4, 6):
            ver = joined
    return oid, ver


def detect_language(text):
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    if re.search(r"\b(hai|hain|kya|mera|meri|mujhe|kar|karna|nahi|kitne|chahiye|aur|kab|kahan|kaise|"
                 r"wala|wapas|bhai|ji|haan|theek|achha|sakta|sakti|hoon|raha|rahi|baar|ko|ka|ki)\b", text, re.I):
        return "hi-en"
    return "en"


LANG_NAME = {"en": "English", "hi": "Hindi in Devanagari script",
             "hi-en": "Hinglish (Hindi in Roman/Latin script, mixed with English)"}

# keyword safety net: these OR with the LLM router so a router miss never skips a handover
KW = {
    "wants_human": r"\b(human|real person|live agent|real agent|manager|supervisor|insaan|insan|kisi se baat)\b|"
                   r"(speak|talk|connect).{0,25}(person|someone|agent|executive)|इंसान|व्यक्ति से बात",
    "legal_threat": r"consumer (court|forum)|\blegal\b|lawyer|advocate|police|\bfir\b|sue\b|court|वकील|पुलिस|अदालत|कोर्ट",
    "unauthorized_activity": r"did ?n[o']?t (place|make|order)|never (placed|ordered|made)|not (placed|made) by me|unauthori[sz]ed|fraud|hack|"
                             r"maine nahi kiya|maine order nahi|मैंने नहीं",
    "safety_issue": r"swell|burning|smoke|spark|fire|caught fire|shock|injur|overheat|phool gay|dhuan|aag|जल|धुआं|चिंगारी",
    "repeat_contact": r"third time|3rd time|teesri baar|tisri baar|तीसरी बार|again and again|baar baar|bar bar|"
                      r"called (twice|two times) already",
    "compensation_demand": r"compensation|voucher|muavza|muaavza|मुआवज़ा|मुआवजा|वाउचर|give me ₹?\s?\d+|"
                           r"\d+\s*(rupaye|rupees|rs)\b.*(chahiye|want|give)",
    "other_person_data": r"(phone|number|address|email|details|naam|name).{0,50}(other|another|someone|person who|neighbou?r|building|"
                         r"customer who)|someone else'?s|dusre ka|किसी और का",
    "damage_claim": r"damag|broken|crack|defect|not working|wrong item|missing|toot|kharab|टूट|खराब|डैमेज",
    "refund_status": r"(where|kahan|kahaan|status).{0,20}refund|refund.{0,25}(where|kahan|kahaan|not (yet )?(received|come|arrived)|nahi aaya)|रिफ़ंड कहाँ|रिफंड कहाँ",
}

ROUTER_PROMPT = """You label one customer message for ZentroMart (an Indian online store) support.
Return ONLY a JSON object with these keys:
"intent": short snake_case label (e.g. where_is_my_order, return_eligibility, refund_timeline, refund_status, cancel_order, damage, warranty, coupon, delivery, payment, plus_membership, human_request, greeting, other),
"needs_order_details": true ONLY if the answer requires looking up THIS customer's order record: tracking/status ("where is my order"), their refund status ("where is my refund"), acting on a specific order ("cancel it"), a damage/replacement claim, or a vague "I need help with my order". false whenever policy alone can answer, even if the customer mentions their order (e.g. "how fast are UPI refunds", "is SKU X assured", "can I return the swimsuit / how many days do I get", "my order is on its way, can I still stop it?", "refund started on 17 Oct by card, when will it arrive").
"wants_human": customer asks for a human/person/agent,
"legal_threat": mentions consumer court, lawyer, police, legal action,
"unauthorized_activity": reports an order/payment/account activity they did not make,
"safety_issue": swelling battery, burning smell, smoke, sparks, overheating, shock or injury,
"repeat_contact": customer says they've contacted support about this several times / third time,
"compensation_demand": asks for money, voucher, discount or compensation beyond policy,
"damage_claim": item arrived damaged, defective, wrong or missing, or asks to approve a replacement for that,
"refund_status": asks where their refund is / why it hasn't arrived,
"search_query": an ENGLISH search query (6-15 words) using policy vocabulary that would find the answer; resolve follow-ups like "how long will that take?" using the recent conversation.
"""


@dataclass
class ConversationState:
    history: list = field(default_factory=list)
    current_order_id: str = None
    verified: bool = False
    order: dict = None
    verification_attempts: int = 0
    awaiting_verification: bool = False
    verification_locked: bool = False
    pending_request: str = None
    last_intent: str = None
    turns: int = 0
    tickets: list = field(default_factory=list)
    order_flags: dict = field(default_factory=dict)


SYSTEM_PROMPT = """You are ZentroMart's AI customer support assistant. ZentroMart is an Indian online store selling groceries, electronics, fashion and home goods. Today's date is 20 October 2026.

Rules:
1. Answer ONLY from the KNOWLEDGE SECTIONS, ORDER FACTS and COMPUTED FACTS given in this turn. Never use general knowledge or invent a policy, date, amount, fee, feature or timeline. The sections are search results and may be only loosely related: if none of them EXPLICITLY answers the customer's specific question (e.g. a feature, service, discount or option that is not mentioned), reply exactly in this spirit: "I'm sorry, I don't have that information" and offer to connect them with a human colleague. Do not guess and do not answer a different question.
2. When sources disagree, follow: current policy > agent guide > help article > FAQ > marketing page. FAQ or marketing wording never overrides a policy (e.g. use the REF-2 refund table, not the FAQ's "5 to 7 days").
3. For dates use the COMPUTED FACTS; do not do your own date arithmetic. Write dates with the month name and year, e.g. "27 October 2026". Mention concrete numbers (days, rupee amounts) when relevant.
4. Your reply will be spoken aloud: 2-3 short sentences, plain text, no markdown, no lists, no section IDs. Ask at most one question.
5. Reply in the language specified for this turn.
6. Never claim an action you did not perform. You cannot cancel orders, change addresses, issue refunds, approve claims or replacements, or give compensation, vouchers or credits. The customer cancels unshipped orders themselves, free, in My Orders in the app. Mention a ticket only if TICKET says one was created.
7. Never read out phone numbers, emails, street addresses or the customer's verification digits.
8. Be warm and polite; when refusing, apologise briefly and explain the policy."""

ACTIONS = {
    "answer": "Answer the customer's latest message.",
    "ask_order_id": ("This needs the customer's own order record. If part of the question can be answered from general "
                     "policy, give that in one short sentence. Then ask for their order ID (one question only)."),
    "ask_verification": ("The order ID is noted. Before sharing ANY order details, ask for ONE of: the last 4 digits of the "
                         "registered phone number, or the delivery pincode of the order. Share no order details."),
    "retry_verification": ("Those details did not match our records. Politely ask them to try once more with the last 4 digits "
                           "of the registered phone number or the delivery pincode. Do not share any order details or hint "
                           "at the correct value."),
    "verification_failed": ("Verification failed twice, so you cannot share any details of this order. Say so politely, suggest "
                            "contacting support from the registered phone number, and offer help with general policy questions. "
                            "Do not mention any date, status or pincode of the order."),
    "verified_answer": ("Verification succeeded. Answer the customer's earlier request: \"{pending}\" using the ORDER FACTS. "
                        "If there was no specific request, briefly give the order status and ask how you can help."),
    "escalate": ("This must go to a human colleague (reason: {reason}). TICKET {ticket} was created. Tell the customer a human "
                 "colleague will contact them, mention the ticket. If they asked for something you cannot do (approve, "
                 "refund, compensate), politely say you can't do it yourself. Do not promise any outcome, amount or time. {extra}"),
    "escalate_failed": ("This needs a human colleague (reason: {reason}) but the ticket could NOT be created. Apologise, say a "
                        "human colleague needs to handle it and ask them to contact support again. Do not say a ticket exists."),
    "refuse_privacy": ("The customer is asking for another person's personal details. Politely refuse: you never share "
                       "another customer's phone number, address or other details. Offer help with their own order instead."),
    "refuse_compensation": ("The customer is asking for compensation/voucher/money that the policy does not allow. Apologise for "
                            "the inconvenience, explain what the policy says (from the knowledge sections and order facts), "
                            "clearly say you cannot offer it, and offer to connect them with a human colleague."),
}

ESC_REASON = {
    "wants_human": "customer asked for a human agent (ESC-1 rule 6)",
    "legal_threat": "customer mentioned legal action / consumer court / police (ESC-1 rule 3)",
    "unauthorized_activity": "customer reports activity they did not make (ESC-1 rule 2)",
    "safety_issue": "product safety risk; customer must stop using the product immediately (ESC-1 rule 7, DWM-5)",
    "repeat_contact": "third or later contact about the same issue within 7 days (ESC-1 rule 4)",
    "high_value_damage": "damaged/defective/wrong item worth more than ₹10,000: Priority Resolution team decides (ESC-1 rule 1, DWM-3)",
    "refund_overdue": "refund more than 3 business days past its maximum timeline (ESC-1 rule 5)",
}
FORCED_SECTIONS = {
    "ask_verification": ["IDV-2"], "retry_verification": ["IDV-3"], "verification_failed": ["IDV-3"],
    "escalate": ["ESC-1", "ESC-2"], "refuse_compensation": ["DEL-4", "ESC-4"], "refuse_privacy": ["IDV-4", "ESC-4"],
}
REASON_SECTIONS = {"high_value_damage": ["DWM-3"], "safety_issue": ["DWM-5"], "refund_overdue": ["REF-6"]}

FORBIDDEN_CLAIMS = re.compile(
    r"[^.!?।]*(I have cancelled|I've cancelled|cancelled it for you|cancelled your order|cancel kar diya|कैंसिल कर दिया|"
    r"रद्द कर दिया|has been credited|voucher (has been )?added|I have approved|I've approved|has been approved|"
    r"approve kar diya|refund (has been )?issued)[^.!?।]*[.!?।]?", re.I)
TICKET_CLAIM = re.compile(r"[^.!?।]*(ticket|टिकट)[^.!?।]*[.!?।]?", re.I)


class Agent:
    def __init__(self, dataset_dir=None, retriever=None, use_llm_router=True):
        if dataset_dir:
            config.set_dataset_dir(dataset_dir)
        self.retriever = retriever or Retriever(config.DATASET_DIR)
        self.lookup_order, self.create_ticket = config.import_tools(config.DATASET_DIR)
        self.use_llm_router = use_llm_router

    # ------------------------------------------------------------ routing
    def route(self, text, state):
        flags = {k: bool(re.search(p, text, re.I)) for k, p in KW.items()}
        r = {"intent": "other", "needs_order_details": False, "search_query": text}
        if self.use_llm_router:
            recent = "\n".join(f"{m['role']}: {m['content']}" for m in state.history[-4:])
            try:
                out = llm.chat_json([
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": f"Recent conversation:\n{recent or '(none)'}\n\nCustomer message: {text}"}])
                r.update({k: v for k, v in out.items() if v is not None})
            except Exception as e:  # router failure must not break the turn
                r["router_error"] = str(e)[:200]
        if not self.use_llm_router or "router_error" in r:
            r["needs_order_details"] = bool(re.search(
                r"where is my order|order status|has ?n.t arrived|not arrived|nahi aaya|nahin aaya|where is it|when will my refund|"
                r"my refund (come|arrive)|mera refund|track (my|the) order|help with (my|an|the|this) order|"
                r"order ke baare|order (kahan|kahaan)|cancel it|ise cancel|isko cancel|मेरा ऑर्डर कहाँ|ऑर्डर के बारे",
                text, re.I))
        for k, v in flags.items():
            r[k] = bool(r.get(k)) or v
        # an order ID in the message always means "this is about my order"
        if extract_numbers(text)[0]:
            r["needs_order_details"] = True
        if r.get("refund_status") or r.get("repeat_contact"):
            r["needs_order_details"] = True
        if len(br.message_facts(text)) > 1:  # customer already gave the date + payment method
            r["needs_order_details"] = False
        return r

    # ------------------------------------------------------------ tools
    def _ticket(self, state, reason_key, text, route):
        o = state.order or {}
        checked = (f"Verified caller; order status {o.get('status')}; " if state.verified else "Caller not verified; ")
        summary = (f"Issue: {text.strip()[:300]} | Intent: {route.get('intent')} | Order: {state.current_order_id or 'n/a'} | "
                   f"Customer said earlier: {(state.pending_request or '')[:200]} | Checked: {checked}"
                   f"{'; '.join(br.order_facts(o)[0][:2]) if state.verified else ''} | Escalation reason: {ESC_REASON[reason_key]}")
        res = self.create_ticket(summary=summary, order_id=state.current_order_id, reason=reason_key,
                                 intent=route.get("intent"))
        if res.get("created"):
            state.tickets.append(res["ticket_id"])
        return res

    # ------------------------------------------------------------ main turn
    def handle(self, text, state):
        text = (text or "").strip()
        state.turns += 1
        lang = detect_language(text)
        norm = normalize_spoken_numbers(text)
        route = self.route(norm, state)
        state.last_intent = route.get("intent")
        tool_calls, ticket, action, fmt, reason = [], None, "answer", {}, None
        oid, ver = extract_numbers(norm)

        # 1. hand over at once, no verification needed (ESC-1 rules 2, 3, 6, 7)
        for key in ("safety_issue", "legal_threat", "unauthorized_activity", "wants_human"):
            if route.get(key) and not route.get("other_person_data"):
                reason = key
                break

        # catalog products mentioned (used for damage value and return-category rules)
        products = self.retriever.find_products(norm + " " + (state.pending_request or ""))
        if not reason and route.get("damage_claim") and not state.verified and \
                any(int(p["price_inr"] or 0) > 10000 for p in products):
            reason = "high_value_damage"  # DWM-3: catalog price already shows it is over ₹10,000

        # 2. order ID / verification bookkeeping
        if not reason and route.get("other_person_data"):
            action = "refuse_privacy"
        elif not reason:
            if oid and oid != state.current_order_id:
                state.current_order_id, state.verified, state.order = oid, False, None
                state.verification_attempts, state.verification_locked = 0, False
                res = self.lookup_order(oid)
                tool_calls.append({"name": "lookup_order", "args": {"order_id": oid}, "verified": None})
                state.awaiting_verification = True
                if not state.pending_request and route.get("intent") not in ("greeting", "other"):
                    state.pending_request = text
            if state.awaiting_verification and ver and not state.verification_locked:
                res = self.lookup_order(state.current_order_id, verification_value=ver)
                ok = bool(res.get("verified"))
                tool_calls.append({"name": "lookup_order", "args": {"order_id": state.current_order_id,
                                                                    "verification_value": "****"}, "verified": ok})
                if ok:
                    state.verified, state.order, state.awaiting_verification = True, res, False
                    _, state.order_flags = br.order_facts(res)
                    action, fmt = "verified_answer", {"pending": state.pending_request or "(none)"}
                else:
                    state.verification_attempts += 1
                    if state.verification_attempts >= 2:
                        state.verification_locked, state.awaiting_verification = True, False
                        action = "verification_failed"
                    else:
                        action = "retry_verification"
            elif state.awaiting_verification and not state.verified:
                if route.get("needs_order_details") or oid:
                    action = "ask_verification"
                    if not oid and not state.pending_request:
                        state.pending_request = text
            elif route.get("needs_order_details") and not state.verified:
                if state.verification_locked:
                    action = "verification_failed"
                elif not state.current_order_id:
                    action = "ask_order_id"
                    state.pending_request = state.pending_request or text

        # 3. order-based escalations (only once verified)
        if not reason and state.verified and action in ("answer", "verified_answer"):
            fl = state.order_flags
            if route.get("damage_claim") and fl.get("high_value_damage"):
                reason = "high_value_damage"
            elif route.get("repeat_contact") and action == "answer":
                reason = "repeat_contact"
            elif route.get("refund_status") and fl.get("refund_overdue"):
                reason = "refund_overdue"
            elif fl.get("repeat_contact") and action == "answer" and route.get("needs_order_details") \
                    and route.get("intent") not in ("greeting",):
                reason = "repeat_contact"

        if reason:
            res = self._ticket(state, reason, text, route)
            tool_calls.append({"name": "create_ticket", "args": {"reason": reason}, "created": bool(res.get("created"))})
            if res.get("created"):
                ticket = res["ticket_id"]
                extra = ("Share the refund reference number (RRN) from the ORDER FACTS so their bank can trace it."
                         if reason == "refund_overdue" else
                         "Tell them to stop using the product immediately." if reason == "safety_issue" else "")
                action, fmt = "escalate", {"reason": ESC_REASON[reason], "ticket": ticket, "extra": extra}
            else:
                action, fmt = "escalate_failed", {"reason": ESC_REASON[reason]}
        elif route.get("compensation_demand") and action == "answer":
            action = "refuse_compensation"

        # 4. retrieval
        query_bits = [route.get("search_query") or "", norm]
        if action == "verified_answer" and state.pending_request:
            query_bits.append(state.pending_request)
        hits = self.retriever.search(" ".join(query_bits), k=config.TOP_K)
        ids = [c.section_id for c, _ in hits]
        extra_ids = FORCED_SECTIONS.get(action, []) + REASON_SECTIONS.get(reason or "", [])
        mfacts = br.message_facts(norm)
        if len(mfacts) > 1:
            extra_ids.append("KB-00.3")
        if any(i.startswith(("FAQ-", "MKT-")) or i == "REF-2" for i in ids):  # FAQ/marketing conflict with REF-2
            extra_ids.append("KB-00.1")
        for sid in extra_ids:
            if sid not in ids and self.retriever.get(sid):
                ids.append(sid)
        sections = [self.retriever.get(i).render() for i in ids]
        cat_sections = {"electronics": ["RET-2", "RET-3"], "fashion": ["RET-2", "RET-8"], "grocery": ["RET-2", "RET-5"],
                        "innerwear": ["RET-2", "RET-5"], "digital": ["RET-2", "RET-5"], "beauty_sealed": ["RET-2", "RET-4"]}
        for p in products:
            if re.search(r"warrant|assured|guarantee|claim", norm, re.I):
                more = ["WAR-2"]
            elif route.get("damage_claim"):
                more = ["DWM-1"]
            else:
                more = list(cat_sections.get(p["return_category"], ["RET-2"]))
                d = br.parse_date_text(norm)
                if p["return_category"] == "electronics" and d and d < br.ELECTRONICS_CHANGE:
                    more.append("RETv3-2")  # superseded table still applies (RET-3)
            for sid in more:
                if sid not in ids and self.retriever.get(sid):
                    ids.append(sid)
                    sections.append(self.retriever.get(sid).render())
        for p in products:
            ids.append(f"products.csv:{p['sku']}")
            sections.append(render_product(p))

        # 5. reply
        facts = br.order_facts(state.order)[0] if state.verified else []
        ctx = ["KNOWLEDGE SECTIONS:\n" + ("\n\n".join(sections) or "(none found)"),
               "ORDER FACTS (verified):\n" + ("\n".join(facts) if facts else "(no verified order)"),
               "COMPUTED FACTS:\n" + "\n".join(mfacts),
               f"TICKET: {ticket + ' created' if ticket else 'none created this turn'}",
               f"REPLY LANGUAGE: {LANG_NAME[lang]}",
               "TASK: " + ACTIONS[action].format(**fmt)]
        if state.turns == 1:
            ctx.append("This is your FIRST reply: start by saying you are ZentroMart's AI support assistant.")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state.history[-8:] + [
            {"role": "user", "content": "\n\n".join(ctx) + f"\n\nCUSTOMER MESSAGE: {text}"}]
        try:
            reply = llm.chat(messages)
        except Exception as e:
            reply = ("Sorry, I'm having trouble right now. Please try again in a moment, or I can connect you "
                     "with a human colleague.")
            route["llm_error"] = str(e)[:200]
        reply = self._postprocess(reply, state, ticket, lang)

        state.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        if action in ("verified_answer", "escalate"):
            state.pending_request = None
        decision = {"escalate": "escalate", "escalate_failed": "escalate", "ask_order_id": "clarify",
                    "ask_verification": "clarify", "retry_verification": "clarify",
                    "verification_failed": "refuse", "refuse_compensation": "refuse",
                    "refuse_privacy": "refuse"}.get(action, "answer")
        if decision == "answer" and state.verified and re.search(r"cancel|रद्द|कैंसिल", norm, re.I) \
                and (state.order or {}).get("status") not in ("placed", "packed", "confirmed", "processing"):
            decision = "refuse"  # CAN-2: a shipped order cannot be cancelled
        return {"reply": reply, "decision": decision, "action": action, "retrieved_sections": ids,
                "tool_calls": tool_calls, "ticket_id": ticket, "language": lang, "route": route}

    def _postprocess(self, reply, state, ticket, lang):
        reply = re.sub(r"\[[A-Z]{2,5}v?\d*-[\d.]+\]", "", reply)            # no section IDs
        reply = re.sub(r"[*#`_]{1,3}", "", reply).strip()                    # no markdown
        reply = FORBIDDEN_CLAIMS.sub("", reply).strip()
        if not ticket:
            reply = TICKET_CLAIM.sub(lambda m: "" if re.search(r"creat|rais|bana|logged|generated", m.group(0), re.I)
                                     else m.group(0), reply).strip()
        if state.turns == 1 and not re.search(r"\bAI\b|एआई", reply):
            intro = {"hi": "नमस्ते, मैं ZentroMart का AI सहायक हूँ। ",
                     "hi-en": "Namaste, main ZentroMart ka AI assistant hoon. "}.get(lang, "Hi, I'm ZentroMart's AI support assistant. ")
            reply = intro + reply
        return reply or "Sorry, could you say that again?"


# Backwards-compatible functional API used by the original scaffold / app.py
_default_agent = None


def handle_message(message, state, agent=None, **_):
    global _default_agent
    if agent is None:
        _default_agent = _default_agent or Agent()
        agent = _default_agent
    return agent.handle(message, state)
