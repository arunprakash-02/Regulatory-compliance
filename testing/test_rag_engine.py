import pathlib
from rag_engine import RAGEngine

def test_rag_pipeline():
    db_path = pathlib.Path(__file__).parent.absolute() / "data" / "knowledge_base.jsonl"
    if not db_path.exists():
        print(f"⚠️ Knowledge base not found at {db_path}. Run ingestion first.")
        return

    engine = RAGEngine(db_path)

    # Test 1: In-domain retrieval & generation
    query_in = "What are the customer due diligence requirements?"
    print(f"🔍 Test 1 (In-Domain): '{query_in}'")
    
    results = engine.search(query_in, top_k=2)
    
    if results:
        print(f"✅ Retrieved {len(results)} documents after re-ranking.")
        for i, r in enumerate(results):
            # Handle the new score names
            score = r.get('_rerank_score', r.get('_tfidf_score', 0.0))
            print(f"   Rank {i+1}: {score:.4f} | {r.get('citation')} | {r.get('article')}")
            
        # Combine the text of the top 2 chunks
        combined_text = "\n\n".join([r['text'] for r in results])
        combined_citation = " | ".join([r.get('citation', '') for r in results])
        
        print(f"\n--- TEXT SENT TO LLM ---\n{combined_text[:600]}...\n------------------------")
        
        answer = engine.generate_answer(query_in, combined_text, combined_citation, "")
        print(f"\n🤖 Llama Answer: {answer}\n")
        assert "I don't know" not in answer, "Incorrect abstention on valid query."
    else:
        print("⚠️ Warning: No in-domain results found above threshold.")

    print("-" * 50)

    # Test 2: Out-of-domain abstention
    query_out = "What is the capital of France?"
    print(f"🔍 Test 2 (Out-of-Domain): '{query_out}'")
    bad_results = engine.search(query_out, top_k=1)
    
    if not bad_results:
        print("✅ Correctly abstained at search phase.")
    else:
        print(f"⚠️ Retrieved match. Testing generation abstention...")
        ans = engine.generate_answer(query_out, bad_results[0]['text'], bad_results[0].get('citation', ''), '')
        print(f"🤖 Model Response: {ans}")
        assert "I don't know based on the provided sources." in ans, "Failed to abstain on irrelevant context."
        print("✅ Correctly abstained at generation phase.")

    print("\n✅ All verification tests completed.")

if __name__ == "__main__":
    test_rag_pipeline()