import json
import pathlib
import time
import re
from rag_engine import RAGEngine


class ComplianceTestSuite:
    def __init__(self, dataset_path, test_suite_path, rag_engine: RAGEngine):
        self.dataset_path = pathlib.Path(dataset_path)
        self.test_suite_path = pathlib.Path(test_suite_path)
        self.engine = rag_engine
        self.corpus = self._load_corpus()
        self.tests = self._load_tests()

    def _load_corpus(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_tests(self):
        with open(self.test_suite_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def run_ingestion_metrics(self):
        print("\n" + "=" * 70)
        print("📥 MODULE A — INGESTION METRICS")
        print("=" * 70)
        results = {}
        target_celex = {"32024R1624", "32024L1640", "32024R1620", "32024R1689", "32024R1623", "32016R0679"}
        found_celex = {d.get("celex", "") for d in self.corpus if d.get("celex")}
        coverage = len(found_celex & target_celex) / len(target_celex) * 100
        results["Regulatory Coverage"] = f"{coverage:.2f}% ({len(found_celex & target_celex)}/{len(target_celex)})"

        hashes = [d.get("provenance", {}).get("chunk_hash", "") for d in self.corpus]
        dups = len(hashes) - len(set(hashes))
        dup_rate = (dups / len(hashes)) * 100 if hashes else 0
        results["Duplicate Detection Rate"] = f"{100 - dup_rate:.2f}% unique (dups: {dups})"

        broken = 0
        for d in self.corpus:
            txt = d.get("text", "")
            if not txt or len(txt) < 50:
                broken += 1
                continue
            if re.match(r'^[a-z]', txt) and not txt.startswith(('"', "'")):
                broken += 1
        extraction_acc = ((len(self.corpus) - broken) / len(self.corpus)) * 100 if self.corpus else 0
        results["Extraction Accuracy"] = f"{extraction_acc:.2f}% (broken: {broken})"

        valid_types = {"authoritative_legal_text", "institutional_guidance", "supplementary_sanctions_metadata"}
        valid_status = {"adopted", "draft_or_guidance", "metadata"}
        misclassified = sum(1 for d in self.corpus if d.get("source_type") not in valid_types or d.get("legal_status") not in valid_status)
        class_acc = ((len(self.corpus) - misclassified) / len(self.corpus)) * 100 if self.corpus else 0
        results["Regulatory Classification Accuracy"] = f"{class_acc:.2f}% (misclassified: {misclassified})"

        by_file = {}
        for d in self.corpus:
            fname = d.get("provenance", {}).get("source_file", "")
            fhash = d.get("provenance", {}).get("file_hash", "")
            if fname:
                if fname in by_file and by_file[fname] != fhash:
                    by_file[fname] = "INCONSISTENT"
                else:
                    by_file[fname] = fhash
        inconsistent = sum(1 for v in by_file.values() if v == "INCONSISTENT")
        change_acc = ((len(by_file) - inconsistent) / len(by_file)) * 100 if by_file else 0
        results["Change Detection Accuracy"] = f"{change_acc:.2f}% (inconsistent: {inconsistent})"

        self._print_metrics(results)
        return results

    def run_retrieval_metrics(self):
        print("\n" + "=" * 70)
        print("🔍 MODULE B — RETRIEVAL METRICS")
        print("=" * 70)
        results = {}
        latencies = []
        total_pos = total_neg = 0
        false_neg = false_pos = 0
        citation_passes = 0

        for t in self.tests:
            ttype = t.get("type")
            if ttype not in ("positive", "negative"): continue
            query = t["query"]
            t0 = time.time()
            results_list = self.engine.search(query, top_k=2)
            latencies.append(time.time() - t0)

            if ttype == "positive":
                total_pos += 1
                if not results_list:
                    false_neg += 1
                    print(f"  ❌ FN (Retrieval): No chunks found for '{query}'")
                    continue
                if self._citations_match_any(results_list, t.get("expected_citation", "")):
                    citation_passes += 1
                else:
                    print(f"  ⚠️ Citation Miss: Expected '{t.get('expected_citation')}', got '{results_list[0].get('citation')}'")
                    
                combined = "\n\n".join(r["text"] for r in results_list)
                combined_cit = " | ".join(r.get("citation", "") for r in results_list)
                ans = self.engine.generate_answer(query, combined, combined_cit, "")
                if "I don't know" in ans:
                    false_neg += 1
                    print(f"  ❌ FN (LLM Refused): '{query}'")
            else:
                total_neg += 1
                if results_list:
                    combined = "\n\n".join(r["text"] for r in results_list)
                    combined_cit = " | ".join(r.get("citation", "") for r in results_list)
                    ans = self.engine.generate_answer(query, combined, combined_cit, "")
                    if "I don't know" not in ans:
                        false_pos += 1
                        print(f"  ❌ FP: LLM answered out-of-domain query '{query}'")

        results["Detection Latency"] = f"{(sum(latencies)/len(latencies)):.3f}s (avg over {len(latencies)})" if latencies else "N/A"
        results["False Negative Rate"] = f"{(false_neg/total_pos)*100:.2f}%" if total_pos else "N/A"
        results["False Positive Rate"] = f"{(false_pos/total_neg)*100:.2f}%" if total_neg else "N/A"
        results["Citation Accuracy"] = f"{(citation_passes/total_pos)*100:.2f}%" if total_pos else "N/A"
        self._print_metrics(results)
        return results

    def run_generation_metrics(self):
        print("\n" + "=" * 70)
        print("🤖 MODULE C — GENERATION METRICS")
        print("=" * 70)
        results = {}
        hallucinations = 0
        tested = 0

        for t in self.tests:
            if t.get("type") != "positive": continue
            tested += 1
            res = self.engine.search(t["query"], top_k=2)
            if not res: continue
            combined = "\n\n".join(r["text"] for r in res)
            ans = self.engine.generate_answer(t["query"], combined, "", "")
            
            if "I don't know" in ans: continue

            is_hallucination = False
            ans_numbers = re.findall(r'\b\d{3,}\b', ans)
            for num in ans_numbers:
                if num not in combined:
                    is_hallucination = True
                    break
            ans_articles = re.findall(r'Article\s+\d+', ans, re.I)
            for art in ans_articles:
                if art.lower() not in combined.lower():
                    is_hallucination = True
                    break

            if is_hallucination:
                hallucinations += 1
                print(f"  ⚠️ Hallucination detected in: '{t['query']}'")

        results["Hallucination Rate"] = f"{(hallucinations/tested)*100:.2f}%" if tested else "N/A"
        self._print_metrics(results)
        return results

    def run_extraction_metrics(self):
        print("\n" + "=" * 70)
        print("📑 MODULE D — EXTRACTION METRICS")
        print("=" * 70)
        results = {}
        recall_hits = recall_total = 0
        deadline_hits = deadline_total = 0
        risk_hits = risk_total = 0
        applicability_hits = applicability_total = 0

        for t in self.tests:
            if t.get("type") == "obligation":
                recall_total += 1
                found = self.engine.extract_obligations(t["text"])
                expected = t.get("expected_obligations", [])
                if any(exp.lower() in " ".join(found).lower() for exp in expected):
                    recall_hits += 1
            elif t.get("type") == "deadline":
                deadline_total += 1
                found = self.engine.extract_deadlines(t["text"])
                expected = t.get("expected_deadlines", [])
                if any(exp.lower() in [f.lower() for f in found] for exp in expected):
                    deadline_hits += 1
            elif t.get("type") == "risk":
                risk_total += 1
                found = self.engine.extract_risk_tier(t["text"])
                if found == t.get("expected_risk_tier", "").lower():
                    risk_hits += 1
            elif t.get("type") == "applicability":
                applicability_total += 1
                expected = t.get("expected_applicable", None)
                if expected is None: continue
                res = self.engine.search(t["query"], top_k=1)
                if res:
                    text = res[0]["text"].lower()
                    has_future = "applies from" in text or "shall apply" in text
                    predicted = not has_future or t.get("query_date", "") > (t.get("effective_date", ""))
                    if predicted == expected: applicability_hits += 1

        results["Obligation Extraction Recall"] = f"{(recall_hits/recall_total)*100:.2f}%" if recall_total else "N/A (no test data)"
        results["Deadline Extraction Accuracy"] = f"{(deadline_hits/deadline_total)*100:.2f}%" if deadline_total else "N/A (no test data)"
        results["Compliance Risk Prioritisation"] = f"{(risk_hits/risk_total)*100:.2f}%" if risk_total else "N/A (no test data)"
        results["Applicability Accuracy"] = f"{(applicability_hits/applicability_total)*100:.2f}%" if applicability_total else "N/A (requires effective_date field)"
        self._print_metrics(results)
        return results

    def run_pipeline_metrics(self):
        print("\n" + "=" * 70)
        print("⏱️  MODULE E — PIPELINE LATENCY METRICS")
        print("=" * 70)
        results = {}
        t0 = time.time()
        _ = self.engine.search("test", top_k=1)
        results["Update Processing Time"] = f"{time.time() - t0:.3f}s (single query roundtrip)"
        results["Alert Delivery Latency"] = "N/A (no alerting subsystem)"
        self._print_metrics(results)
        return results

    def _citations_match_any(self, results_list, expected):
        if not expected: return False
        exp_lower = expected.lower()
        expected_celex = set(re.findall(r'\b3\d{4}[A-Z]\d{4}\b', expected))
        year_num = re.findall(r'\b20\d{2}/(\d{3,4})\b', expected)
        for num in year_num:
            year = re.search(r'\b(20\d{2})/', expected).group(1)
            expected_celex.add(f"3{year}R{num.zfill(4)}")
            expected_celex.add(f"3{year}L{num.zfill(4)}")
        expected_articles = set(re.findall(r'article\s+(\d+[a-z]?)', expected, re.I))
        for r in results_list:
            r_celex = r.get("celex", "").upper()
            if r_celex and r_celex in expected_celex: return True
            if r_celex and r_celex.lower() in exp_lower: return True
            r_article = r.get("article", "").lower()
            if r_article:
                for exp_art in expected_articles:
                    if exp_art in r_article: return True
            r_citation = r.get("citation", "").lower()
            if r_citation:
                exp_words = set(w for w in re.findall(r'\b[a-z]{5,}\b', exp_lower))
                cit_words = set(w for w in re.findall(r'\b[a-z]{5,}\b', r_citation))
                if len(exp_words & cit_words) >= 3: return True
        return False

    def _print_metrics(self, results):
        for k, v in results.items():
            print(f"  • {k:<38}: {v}")

    def run_all_tests(self):
        print("\n" + "█" * 70)
        print("🧪 FULL COMPLIANCE METRICS SUITE — 15 METRICS")
        print("█" * 70)
        self.run_ingestion_metrics()
        self.run_retrieval_metrics()
        self.run_generation_metrics()
        self.run_extraction_metrics()
        self.run_pipeline_metrics()
        print("\n" + "█" * 70)
        print("✅ SUITE COMPLETE")
        print("█" * 70 + "\n")

if __name__ == "__main__":
    base = pathlib.Path(__file__).parent.absolute()
    db_path = base / "data" / "knowledge_base.jsonl"
    test_path = base / "data" / "master_test_suite.jsonl"

    if db_path.exists() and test_path.exists():
        engine = RAGEngine(db_path)
        suite = ComplianceTestSuite(db_path, test_path, engine)
        suite.run_all_tests()
    else:
        print("⚠️ Missing knowledge_base.jsonl or master_test_suite.jsonl")