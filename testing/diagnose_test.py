import json, pathlib
from rag_engine import RAGEngine

def diagnose():
    base = pathlib.Path(__file__).parent.absolute()
    engine = RAGEngine(base / "data" / "knowledge_base.jsonl")
    
    tests = [json.loads(l) for l in open(base / "data" / "master_test_suite.jsonl", encoding="utf-8") if l.strip()]
    positives = [t for t in tests if t.get("type") == "positive"]
    
    print(f"\n📋 Total tests: {len(tests)} | Positives: {len(positives)}\n")
    
    # Show FIRST positive test in full detail
    if positives:
        t = positives[0]
        print("=" * 70)
        print("DIAGNOSTIC: First Positive Test")
        print("=" * 70)
        print(f"Query: {t.get('query')}")
        print(f"expected_citation field: {repr(t.get('expected_citation'))}")
        print(f"All fields in test: {list(t.keys())}")
        
        results = engine.search(t["query"], top_k=2)
        print(f"\nRetrieved {len(results)} chunks.")
        
        if results:
            for i, r in enumerate(results):
                print(f"\n--- Chunk {i+1} ---")
                print(f"  celex: {repr(r.get('celex'))}")
                print(f"  article: {repr(r.get('article'))}")
                print(f"  citation: {repr(r.get('citation', '')[:100])}")
                print(f"  score: {r.get('_rerank_score', 0):.4f}")
                print(f"  text (first 300): {r['text'][:300]}")
            
            combined = "\n\n".join(r["text"] for r in results)
            combined_cit = " | ".join(r.get("citation", "") for r in results)
            
            print(f"\n--- LLM INPUT ---")
            print(f"Source: {combined_cit} ()")
            print(f"Context: {combined[:400]}...")
            
            ans = engine.generate_answer(t["query"], combined, combined_cit, "")
            print(f"\n--- LLM OUTPUT ---")
            print(ans)

if __name__ == "__main__":
    diagnose()