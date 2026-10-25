"""Step 2: section-level retrieval over kb/*.md (plus catalog rows).

Each chunk is one `## [SECTION-ID] Title` section, so the section ID travels
with the text and can be reported for retrieval scoring. Superseded documents
are excluded from search (their only live rule - pre-August electronics keep a
10-day window - is applied in business_rules.py from RET-3).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRECEDENCE = {"policy": 1, "agent_guide": 2, "help": 3, "faq": 4, "marketing": 5, "meta": 0}

# Customers don't use the documents' words. Expand common phrasings
# (English, Hinglish, Hindi) into the vocabulary the policies use.
SYNONYMS = {
    r"send (it )?back|give (it )?back|wapas|वापस|lauta|लौटा|\breturn": "return window days category returnable",
    r"how long|kitne din|कितने दिन|how many days|till when|kab tak": "window days business days timeline",
    r"collect|pick ?up|nobody came|koi nahi aaya": "reverse pickup missed pickup",
    r"money back|paisa|पैसे|refund|रिफ़ंड|रिफंड": "refund initiated business days payment method",
    r"cancel|रद्द|कैंसिल|stop it|stop the order|on its way|rok": "cancel cancellation after the order ships refuse the parcel",
    r"never placed|did ?n.t (place|order)|not me|fraud|hack": "order payment account activity they did not make hand over human",
    r"(phone|number|address|email).{0,40}(person|someone|customer|neighbour|building)": "never read out phone share another customer's details",
    r"not (at )?home|ghar pe nahi|missed delivery|reschedul": "delivery attempts reschedule SMS",
    r"late|der se|delay|देर": "late deliveries promised date compensation",
    r"voucher|compensation|muavza|मुआवज़ा|credit": "compensation late deliveries wallet credit",
    r"broken|damaged|toota|टूटा|crack|defect|wrong item|missing": "damaged defective wrong missing item high-value Priority Resolution",
    r"human|person|insaan|इंसान|real agent|manager": "hand over human colleague",
    r"warranty|guarantee|assured": "warranty assured claim My Orders",
    r"coupon|promo|code|discount": "coupon offers one coupon per order",
    r"\bcod\b|cash on delivery|cash": "cash on delivery COD fee refund wallet bank transfer",
    r"plus|membership": "ZentroMart Plus membership benefits",
    r"track|where is|kahan hai|कहाँ": "track order status promised date",
    r"open(ed)?|seal|khol|खोल": "condition seal intact unopened",
    r"battery|smoke|spark|burning|garam|heat": "safety risk stop using hand over",
    r"deliver(y)? fee|shipping (fee|charge)": "delivery fees free 499",
}


@dataclass
class Chunk:
    section_id: str
    title: str
    text: str
    doc_id: str
    doc_type: str
    status: str
    effective_from: str
    source: str
    meta: dict = field(default_factory=dict)

    def render(self):
        return (f"[{self.section_id}] {self.title}  (doc {self.doc_id}, type {self.doc_type}, "
                f"status {self.status}, effective {self.effective_from})\n{self.text.strip()}")


def _parse_header(raw):
    meta = {}
    if raw.startswith("---"):
        _, head, body = raw.split("---", 2)
        for line in head.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta, body
    return meta, raw


def load_kb(kb_dir, include_superseded=False):
    chunks = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        meta, body = _parse_header(path.read_text(encoding="utf-8"))
        status = meta.get("status", "current")
        if status == "superseded" and not include_superseded:
            continue
        parts = re.split(r"^## \[([^\]]+)\]\s*(.*)$", body, flags=re.M)
        # parts = [preamble, id1, title1, text1, id2, title2, text2, ...]
        for i in range(1, len(parts) - 2, 3):
            sid, title, text = parts[i].strip(), parts[i + 1].strip(), parts[i + 2]
            chunks.append(Chunk(sid, title, text, meta.get("doc_id", path.stem),
                                meta.get("doc_type", ""), status,
                                meta.get("effective_from", ""), path.name, meta))
    return chunks


def expand_query(q):
    extra = [v for pat, v in SYNONYMS.items() if re.search(pat, q, flags=re.I)]
    return q + " " + " ".join(extra)


class Retriever:
    def __init__(self, dataset_dir, include_superseded=False):
        dataset_dir = Path(dataset_dir)
        self.chunks = load_kb(dataset_dir / "kb", include_superseded)
        # superseded sections are not searched, but can be fetched by ID when a
        # current policy says they still apply (RET-3 -> RETv3-2)
        self.by_id = {c.section_id: c for c in load_kb(dataset_dir / "kb", include_superseded=True)}
        docs = [f"{c.title} {c.title} {c.text}" for c in self.chunks]
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.Xw = self.word.fit_transform(docs)
        self.Xc = self.char.fit_transform(docs)
        self.products = pd.read_csv(dataset_dir / "data" / "products.csv", dtype=str).fillna("")
        self.products["base_title"] = self.products["title"].str.split(",").str[0].str.lower()

    def search(self, query, k=5):
        q = expand_query(query)
        sw = cosine_similarity(self.word.transform([q]), self.Xw)[0]
        sc = cosine_similarity(self.char.transform([q]), self.Xc)[0]
        scores = 0.6 * sw + 0.4 * sc
        order = scores.argsort()[::-1]
        hits = []
        for i in order[:k]:
            if scores[i] <= 0:
                break
            hits.append((self.chunks[i], float(scores[i])))
        return hits

    def get(self, section_id):
        return self.by_id.get(section_id)

    def find_products(self, text, limit=3):
        """Catalog rows mentioned by SKU or by product name."""
        t = text.lower()
        found = []
        for sku in re.findall(r"\b([a-z]{3})[-\s]?(\d{4})\b", t):
            code = f"{sku[0].upper()}-{sku[1]}"
            rows = self.products[self.products.sku == code]
            if len(rows):
                found.append(rows.iloc[0].to_dict())
        if not found:
            # longest product names first so "X2 Pro" beats "X2"
            cand = self.products[self.products.base_title.apply(lambda b: len(b) > 6 and b in t)]
            for _, r in cand.sort_values("base_title", key=lambda s: s.str.len(), ascending=False).iterrows():
                found.append(r.to_dict())
        seen, out = set(), []
        for r in found:
            if r["sku"] not in seen:
                seen.add(r["sku"])
                out.append(r)
        return out[:limit]


def render_product(p):
    return (f"[products.csv:{p['sku']}] {p['title']} | brand {p['brand']} | return_category "
            f"{p['return_category']} | price ₹{p['price_inr']} | fulfilled_by {p['fulfilled_by']} | "
            f"assured {p['assured']} | warranty_months {p['warranty_months']}")
