import json, pathlib
from rag_engine import RAGEngine

base = pathlib.Path(__file__).parent.absolute()
engine = RAGEngine(base / "data" / "knowledge_base.jsonl")

query = "How does the EU AI Act define high-risk systems?"
print(f"\n{'='*70}\nQUERY: {query}\n{'='*70}\n")

results = engine.search(query, top_k=3)

for i, r in enumerate(results):
    print(f"--- CHUNK {i+1} ---")
    print(f"celex: {r.get('celex')}")
    print(f"article: {r.get('article')}")
    print(f"citation: {r.get('citation', '')[:80]}")
    print(f"score: {r.get('_score', 0):.4f}")
    print(f"text: {r['text'][:600]}...")
    print()

combined = "\n\n".join(r["text"] for r in results)
cit = " | ".join(r.get("citation", "") for r in results)
print(f"{'='*70}\nLLM ANSWER:\n{'='*70}")
print(engine.generate_answer(query, combined, cit, ""))
print()

# Check for empty chunk hashes
print(f"{'='*70}\nEMPTY CHUNK HASH CHECK:\n{'='*70}")
from collections import Counter
corpus = [json.loads(l) for l in open(base / "data" / "knowledge_base.jsonl", encoding="utf-8") if l.strip()]
c = Counter(d.get("provenance", {}).get("chunk_hash", "") for d in corpus)
print(f"Empty hashes (''): {c.get('', 0)}")
print(f"Top 3 hashes: {c.most_common(3)}")