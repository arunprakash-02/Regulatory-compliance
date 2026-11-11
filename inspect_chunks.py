import json, pathlib, random
from collections import Counter

BASE_DIR = pathlib.Path(r"D:\MyLLM\rag\data")
KB_FILE = BASE_DIR / "knowledge_base.jsonl"
OFSI_FILE = BASE_DIR / "ofsi_sanctions.jsonl"

def load_jsonl(path):
    if not path.exists():
        print(f"❌ {path.name} not found.")
        return []
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs

def inspect_knowledge_base(docs):
    print(f"\n{'='*50}")
    print(f"🔍 INSPECTING: knowledge_base.jsonl")
    print(f"{'='*50}")
    print(f"✅ Total chunks: {len(docs)}")

    # 1. Source Type Distribution
    types = Counter(d.get("source_type", "unknown") for d in docs)
    print("\n--- Source Type Distribution ---")
    for t, count in types.items():
        print(f"  {t}: {count}")

    # 2. Check for Missing Provenance
    missing_prov = sum(1 for d in docs if not d.get("provenance", {}).get("source_url"))
    print(f"\n--- Missing Provenance (source_url) ---")
    print(f"  Chunks without source_url: {missing_prov}")

    # 3. Check Chunk Lengths
    lengths = [len(d.get("text", "")) for d in docs]
    if lengths:
        print(f"\n--- Chunk Length Stats ---")
        print(f"  Min: {min(lengths)} chars")
        print(f"  Max: {max(lengths)} chars")
        print(f"  Avg: {sum(lengths) // len(lengths)} chars")
        
        too_short = sum(1 for l in lengths if l < 100)
        if too_short:
            print(f"  ⚠️  Chunks < 100 chars: {too_short}")

    # 4. Check CELEX coverage
    celex_counts = Counter(d.get("celex", "") for d in docs if d.get("celex"))
    print("\n--- CELEX Coverage ---")
    if not celex_counts:
        print("  ⚠️ No CELEX numbers found!")
    for celex, count in celex_counts.most_common():
        print(f"  {celex}: {count}")

    # 5. Check article coverage
    article_missing = sum(1 for d in docs if not d.get("article"))
    print(f"\n--- Missing Article ---")
    print(f"  Chunks without article: {article_missing}")

    # 6. Check duplicate chunk hashes
    hashes = [d.get("provenance", {}).get("chunk_hash") for d in docs]
    duplicate_hashes = len(hashes) - len(set(hashes))
    print(f"\n--- Duplicate Chunk Hashes ---")
    print(f"  Duplicates: {duplicate_hashes}")

    # 7. Check source URL coverage
    source_urls = Counter(
        d.get("provenance", {}).get("source_url", "") for d in docs
    )
    print("\n--- Source URL Coverage (Top 10) ---")
    for url, count in source_urls.most_common(10):
        print(f"  {count}: {url if url else '[EMPTY]'}")

    # 8. Random Sample
    print("\n--- Random Sample (2 chunks) ---")
    for doc in random.sample(docs, min(2, len(docs))):
        print(f"\n  ID: {doc.get('id')}")
        print(f"  Citation: {doc.get('citation')}")
        print(f"  Article: {doc.get('article')}")
        print(f"  Source URL: {doc.get('provenance', {}).get('source_url')}")
        print(f"  Text (first 300 chars): {doc.get('text', '')[:300]}...")

def inspect_ofsi(docs):
    print(f"\n{'='*50}")
    print(f"🔍 INSPECTING: ofsi_sanctions.jsonl")
    print(f"{'='*50}")
    print(f"✅ Total OFSI chunks: {len(docs)}")

    if not docs:
        return

    # Check duplicates
    hashes = [d.get("provenance", {}).get("chunk_hash") for d in docs]
    duplicate_hashes = len(hashes) - len(set(hashes))
    print(f"\n--- Duplicate Chunk Hashes ---")
    print(f"  Duplicates: {duplicate_hashes}")

    # Random Sample
    print("\n--- Random Sample (1 chunk) ---")
    doc = random.choice(docs)
    print(f"\n  ID: {doc.get('id')}")
    print(f"  Citation: {doc.get('citation')}")
    print(f"  Source URL: {doc.get('provenance', {}).get('source_url')}")
    print(f"  Text (first 300 chars): {doc.get('text', '')[:300]}...")

if __name__ == "__main__":
    kb_docs = load_jsonl(KB_FILE)
    if kb_docs:
        inspect_knowledge_base(kb_docs)
        
    ofsi_docs = load_jsonl(OFSI_FILE)
    if ofsi_docs:
        inspect_ofsi(ofsi_docs)
        
    print(f"\n{'='*50}")
    print("Inspection Complete.")