import json, pathlib, random

BASE_DIR = pathlib.Path(__file__).parent.absolute() / "data"
KB_PATH = BASE_DIR / "knowledge_base.jsonl"
OUT_PATH = BASE_DIR / "master_test_suite.jsonl"

def build_test_suite():
    if not KB_PATH.exists():
        print("Knowledge base not found. Run ingestion first.")
        return

    docs = []
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    test_cases = []
    
    # 1. Sample Positive Controls from real chunks
    sample_docs = random.sample(docs, min(5, len(docs)))
    for doc in sample_docs:
        citation = doc.get("citation", "EU Regulation")
        article = doc.get("article", "Article")
        text_snippet = doc.get("text", "")[:100]
        
        test_case = {
            "test_id": f"pos_{random.randint(100, 999)}",
            "type": "positive",
            "query": f"What are the regulations regarding {article} under {citation}?",
            "expected_citation": citation,
            "expected_keywords": text_snippet.split()[:3]
        }
        test_cases.append(test_case)

    # 2. Add Standard Negative Controls (False Positive tests)
    negative_controls = [
        {
            "test_id": "neg_001",
            "type": "negative",
            "query": "What are the mandatory crypto-mining carbon offset taxes under GDPR?",
            "category": "non_existent_obligation",
            "expected_string": "I don't know based on the provided sources."
        },
        {
            "test_id": "neg_002",
            "type": "negative",
            "query": "What is the capital city of France?",
            "category": "out_of_domain",
            "expected_string": "I don't know based on the provided sources."
        },
        {
            "test_id": "neg_003",
            "type": "negative",
            "query": "Under Article 5 of GDPR, what are the capital adequacy ratio minimums for retail banks?",
            "category": "distractor_cross_regulation",
            "expected_string": "I don't know based on the provided sources."
        }
    ]
    test_cases.extend(negative_controls)

    # Save to JSONL
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for case in test_cases:
            f.write(json.dumps(case) + "\n")
            
    print(f"✅ Generated {len(test_cases)} test cases at {OUT_PATH}")

if __name__ == "__main__":
    build_test_suite()