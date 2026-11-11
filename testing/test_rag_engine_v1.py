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
    
    # We need top_k=2 to give the LLM enough context
    results = engine.search(query_in, top_k=2)
    
    if results:
        # Handle the new score keys
        top_score = results[0].get('_rerank_score', results[0].get('_tfidf_score', 0.0))
        print(f"✅ Retrieved {len(results)} documents after re-ranking.")
        print(f"   Top Score: {top_score:.4f} | Citation: {results[0].get('citation')}")
        
        # Combine context
        combined_text = "\n\n".join([r['text'] for r in results])
        combined_citation = " | ".join([r.get('citation', '') for r in results])
        
        # Pass all 4 required arguments
        answer = engine.generate_answer(query_in, combined_text, combined_citation, "")
        print(f"🤖 Answer generated successfully (Length: {len(answer)} chars)")
        assert "I don't know" not in answer, "Incorrect abstention on valid query."
    else:
        print("⚠️ Warning: No in-domain results found above threshold.")

    print("-" * 50)

    # Test 2: Out-of-domain abstention
    query_out = "What is the capital of France?"
    print(f"🔍 Test 2 (Out-of-Domain): '{query_out}'")
    bad_results = engine.search(query_out, top_k=1)
    
    if not bad_results:
        print("✅ Correctly abstained at search phase (Score < 0.30 or keyword mismatch).")
    else:
        best_bad = bad_results[0]
        bad_score = best_bad.get('_rerank_score', best_bad.get('_tfidf_score', 0.0))
        print(f"⚠️ Retrieved match with score {bad_score:.4f}. Testing generation abstention...")
        
        # Pass all 4 required arguments
        ans = engine.generate_answer(query_out, best_bad['text'], best_bad.get('citation', ''), '')
        print(f"🤖 Model Response: {ans}")
        assert "I don't know based on the provided sources." in ans, "Failed to abstain on irrelevant context."
        print("✅ Correctly abstained at generation phase.")

    print("\n✅ All verification tests completed.")

if __name__ == "__main__":
    test_rag_pipeline()