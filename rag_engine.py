import json
import pathlib
import numpy as np
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class RAGEngine:
    def __init__(self, jsonl_path, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "llama3.2:3b"
        self.docs = []
        self._load_data(jsonl_path)
        self._build_index()

    def _load_data(self, path):
        print(f"Loading dataset from {pathlib.Path(path).name}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.docs.append(json.loads(line))
        print(f"✅ Loaded {len(self.docs)} legal/guidance chunks.")

    def _build_index(self):
        if not self.docs:
            raise ValueError("No documents found.")
        print("Building TF-IDF + SVD index...")
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=15000)
        tfidf_matrix = self.vectorizer.fit_transform([d['text'] for d in self.docs])
        n_comp = min(150, max(1, len(self.docs) - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors = self.doc_vectors / norms
        print("✅ Index built. Ready for queries.\n")

    # ---------------- SEARCH (core) ----------------
    def _raw_search(self, query, top_k):
        """Pure retrieval — no retries, no expansion."""
        if not query or not query.strip():
            return []

        q_lower = query.lower()
        q_tfidf = self.vectorizer.transform([query])
        q_vec = self.svd.transform(q_tfidf)[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) or 1.0)
        scores = self.doc_vectors @ q_vec

        query_words = set(re.findall(r'\b[a-z]{4,}\b', q_lower))
        min_matches = 2 if len(query_words) >= 3 else 1

        candidates = []
        for idx in scores.argsort()[::-1]:
            if scores[idx] >= 0.15:
                doc = self.docs[idx].copy()
                matches = sum(1 for word in query_words if word in doc['text'].lower())
                if matches < min_matches:
                    continue
                doc['_score'] = float(scores[idx])
                candidates.append(doc)
                if len(candidates) == 20:
                    break

        if not candidates:
            return []

        # --- SOURCE BOOSTING ---
        regulation_boost = {}
        for celex_key, keywords in {
            "32024R1624": ["aml regulation", "2024/1624", "amlar"],
            "32024L1640": ["aml directive", "2024/1640", "amld"],
            "32024R1620": ["amla regulation", "2024/1620"],
            "32024R1689": ["ai act", "2024/1689", "artificial intelligence"],
            "32024R1623": ["crr3", "2024/1623", "capital requirements"],
            "32016R0679": ["gdpr", "2016/679", "data protection"],
        }.items():
            if any(kw in q_lower for kw in keywords):
                regulation_boost[celex_key] = 0.10

        # --- DEFINITION BOOSTING ---
        is_definition = any(w in q_lower for w in [
            "define", "definition", "what is", "what are", "meaning",
            "criteria", "classification", "considered"
        ])

        definition_markers = [
            "shall be considered to be a high-risk",
            "shall be considered high-risk",
            "is considered to be high-risk",
            "are considered to be high-risk",
            "shall be classified as high-risk",
            "meets the criteria",
            "classification rules",
            "referred to in annex iii",
            "listed in annex iii",
            "set out in annex iii",
        ]

        for doc in candidates:
            if regulation_boost:
                if doc.get("celex") in regulation_boost:
                    if doc.get("source_type") == "authoritative_legal_text":
                        doc["_score"] += regulation_boost[doc["celex"]]
                elif doc.get("source_type") == "institutional_guidance":
                    doc["_score"] -= 0.05

            if is_definition:
                text_lower = doc['text'].lower()
                # Heavy boost for actual definitional language
                for marker in definition_markers:
                    if marker in text_lower:
                        doc["_score"] += 0.30
                        break  # one boost per doc, don't double-count
                # Article 6 is the canonical definition of high-risk in AI Act
                if doc.get("article") in ["Article 6", "Article 7"]:
                    doc["_score"] += 0.15

        candidates.sort(key=lambda x: x["_score"], reverse=True)
        return candidates[:top_k]

    def search(self, query, top_k=3):
        """Public search — with self-query retry if the query is definitional."""
        results = self._raw_search(query, top_k)

        # If the query is definitional and we got results, no retry needed
        # (retry happens in generate_answer if LLM refuses)
        return results

    # ---------------- ANSWER GENERATION ----------------
    def _call_llm(self, query, context, citation, article):
        source_header = f"Source: {citation}"
        if article and article.strip():
            source_header += f" ({article})"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a compliance expert. Use the provided context to answer the question.\n\n"
                        "INSTRUCTIONS:\n"
                        "1. If the context contains partial information or related rules, synthesize the best answer you can from the text.\n"
                        "2. Only respond with 'I don't know based on the provided sources.' if the context is completely unrelated "
                        "(e.g., asking about sports or geography when the context is about financial law).\n"
                        "3. Do not invent numbers, dates, or citations not present in the context.\n"
                        "4. Keep the answer to 2-4 sentences."
                    )
                },
                {
                    "role": "user",
                    "content": f"{source_header}\n\nContext:\n{context}\n\nQuestion: {query}"
                }
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 250}
        }
        try:
            resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "").strip()
            return f"Error: Ollama returned status code {resp.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

    def _expand_query(self, query):
        """Strip definitional boilerplate to get a cleaner retrieval query."""
        q = query.lower()
        # Remove question scaffolding
        for phrase in [
            "how does", "how do", "what does", "what do", "what is", "what are",
            "define", "definition of", "meaning of", "explain",
            "the eu", "the", "under the",
        ]:
            q = q.replace(phrase, " ")
        return re.sub(r'\s+', ' ', q).strip()

    def generate_answer(self, query, context, citation, article):
        """Primary generation — with one retry on refusal for definitional queries."""
        answer = self._call_llm(query, context, citation, article)

        # Check if the LLM refused AND this is a definitional query
        is_refusal = "i don't know" in answer.lower() or "i do not know" in answer.lower()
        is_definitional = any(w in query.lower() for w in [
            "define", "definition", "what is", "what are", "meaning", "criteria"
        ])

        if is_refusal and is_definitional:
            print(f"  🔄 Retry with expanded query for: '{query[:50]}...'")
            expanded = self._expand_query(query)
            retry_docs = self._raw_search(expanded, top_k=5)  # MORE chunks on retry
            if retry_docs:
                retry_context = "\n\n".join(r['text'] for r in retry_docs)
                retry_cit = " | ".join(r.get('citation', '') for r in retry_docs)
                retry_ans = self._call_llm(query, retry_context, retry_cit, "")
                if "i don't know" not in retry_ans.lower():
                    return retry_ans

        return answer

    # ---------------- EXTRACTION HELPERS ----------------
    def extract_obligations(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        markers = r'\b(shall|must|is required to|are required to|ought to|shall ensure|shall apply)\b'
        return [s.strip() for s in sentences if re.search(markers, s, re.I)]

    def extract_deadlines(self, text):
        patterns = [
            r'\bwithin\s+\d+\s+(?:days?|months?|years?|hours?|weeks?)\b',
            r'\bby\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
            r'\bby\s+\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\bnot later than\s+[^.]{5,50}',
            r'\bat least\s+\d+\s+(?:days?|months?|years?)\b',
        ]
        found = []
        for p in patterns:
            found.extend(re.findall(p, text, re.I))
        return list(dict.fromkeys(found))

    def extract_risk_tier(self, text):
        text_l = text.lower()
        if re.search(r'\bhigh[-\s]?risk\b', text_l): return "high"
        if re.search(r'\bmedium[-\s]?risk\b', text_l): return "medium"
        if re.search(r'\blow[-\s]?risk\b', text_l): return "low"
        return "none"