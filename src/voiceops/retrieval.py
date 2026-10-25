from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_kb(kb_dir):
    from pathlib import Path
    import re

    root = Path(kb_dir)
    if (root / "kb").is_dir():
        root = root / "kb"

    chunks = []
    pattern = re.compile(
        r"(?m)^##\s+\[([A-Z][A-Z0-9]*-\d+(?:\.\d+)?)\]\s*(.*?)\s*$"
    )

    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        matches = list(pattern.finditer(text))

        for i, match in enumerate(matches):
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunks.append({
                "section_id": match.group(1),
                "title": match.group(2).strip(),
                "source_file": path.name,
                "text": text[match.start():end_pos].strip(),
                "superseded_source": "superseded" in path.name.lower(),
            })

    if not chunks:
        raise ValueError(
            f"No canonical sections found. Searched Markdown files under: {root}"
        )

    ids = [item["section_id"] for item in chunks]
    if len(ids) != len(set(ids)):
        duplicates = sorted({x for x in ids if ids.count(x) > 1})
        print("Warning: repeated section IDs:", duplicates)

    print(f"Loaded {len(chunks)} knowledge-base sections from {root}")
    return chunks

def build_retriever(chunks):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode"
    )
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return vectorizer, matrix

def retrieve_kb(question, chunks, vectorizer, matrix, top_k=6):
    query = vectorizer.transform([str(question)])
    scores = cosine_similarity(query, matrix).ravel()
    ranked = scores.argsort()[::-1]

    results = []
    for idx in ranked:
        item = dict(chunks[int(idx)])
        item["score"] = float(scores[idx])
        results.append(item)
        if len(results) >= top_k:
            break
    return results
