"""
Compliance test suite v2 -- drop-in replacement for test_compliance_suite.py.

Run from D:\\MyLLM\\rag (same folder as rag_engine.py):

    python test_compliance_suite_v2.py                       # full run
    python test_compliance_suite_v2.py --no-llm              # ingestion + retrieval only (no Ollama)
    python test_compliance_suite_v2.py --prev data\\knowledge_base.prev.jsonl
    python test_compliance_suite_v2.py --inspect "How does the EU AI Act define high-risk systems?"

What changed versus v1 (see the README comments next to each module):
  * every rate prints hits/total plus a 95% Wilson interval, and warns when n < 30
  * retrieval is scored by CELEX + article (recall@2/5/10, MRR), not by loose word overlap
  * retrieval misses, LLM over-refusals and Ollama errors are reported separately
  * hallucination check covers numbers of any length with units, dates, article/annex refs
    (whole-token match, so "Article 6" no longer passes because the context has "Article 60")
  * missing chunk hashes are no longer counted as duplicates
  * change detection compares against a previous corpus (--prev) instead of checking that
    one file has one hash
  * "Update Processing Time" is now the real index rebuild time
  * --inspect prints the retrieved chunks and the answer for one query (use it to build gold labels)

Test records (data/master_test_suite.jsonl), one JSON object per line.  Old records still work.

  {"type":"positive","query":"...","expected_celex":"32024R1689","expected_article":"Article 6"}
  {"type":"positive","query":"...","expected_citation":"Regulation (EU) 2024/1689, Article 6"}   # legacy
  {"type":"negative","query":"What is the VAT rate in Germany?"}
  {"type":"obligation","text":"...","expected_obligations":["..."],"exhaustive":false}
  {"type":"deadline","text":"...","expected_deadlines":["within 30 days"],"exhaustive":false}
  {"type":"risk","text":"...","expected_risk_tier":"high"}
  {"type":"applicability","query":"...","query_date":"2027-01-01","effective_date":"2027-07-10",
   "expected_applicable":false}
  {"type":"classification","source_file":"...","expected_source_type":"...","expected_legal_status":"..."}
"""
import argparse
import contextlib
import io
import json
import math
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date

from rag_engine import RAGEngine

# --------------------------------------------------------------------------- config
TARGETS = {
    "32024R1624": "AML Regulation (AMLR)",
    "32024L1640": "AML Directive (AMLD6)",
    "32024R1620": "AMLA Regulation",
    "32024R1689": "EU AI Act",
    "32024R1623": "CRR3",
    "32016R0679": "GDPR",
}
MIN_CHUNKS_PER_TARGET = 20          # fewer chunks than this = "thin", counts as not covered
KS = (2, 5, 10)                     # recall@k cut-offs; the app sends the top 2 chunks to the LLM
LLM_TOP_K = 2
VALID_TYPES = {"authoritative_legal_text", "institutional_guidance", "supplementary_sanctions_metadata"}
VALID_STATUS = {"adopted", "draft_or_guidance", "metadata"}
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
UNITS = r"(?:days?|weeks?|months?|years?|hours?|percent|euros?|eur|million|billion)"


# --------------------------------------------------------------------------- helpers
def wilson(hits, total, z=1.96):
    if total == 0:
        return 0.0, 0.0
    p = hits / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(hits, total):
    if total == 0:
        return "N/A (0 tests)"
    lo, hi = wilson(hits, total)
    note = "   <- anecdotal (n<30), add tests" if total < 30 else ""
    return f"{hits / total * 100:.1f}% ({hits}/{total}, 95% CI {lo * 100:.0f}-{hi * 100:.0f}%){note}"


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def norm_text(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def is_error(ans):
    return bool(ans) and (ans.startswith("Error: Ollama") or ans.startswith("Error connecting to Ollama"))


def is_refusal(ans):
    a = (ans or "").replace("\u2019", "'").strip().lower()
    return a.startswith("i don't know") or "i don't know based on the provided sources" in a


def article_key(s, strict=False):
    """'Article 6(2)' -> 'article 6', 'Annex III' -> 'annex iii'. strict=True needs the word Article/Annex/Recital."""
    s = (s or "").lower()
    m = re.search(r"annex\s+([ivxlc]+|\d+)\b", s)
    if m:
        return f"annex {m.group(1)}"
    m = re.search(r"recital\s+(\d+)", s)
    if m:
        return f"recital {m.group(1)}"
    pat = r"article\s+(\d+[a-z]?)\b" if strict else r"(?:article\s+)?(\d+[a-z]?)\b"
    m = re.search(pat, s)
    return f"article {m.group(1)}" if m else ""


def derive_expected(t):
    """Return (celex_set, article_key_set) for a positive test, supporting the legacy expected_citation."""
    celex, arts = set(), set()
    ec = t.get("expected_celex")
    if ec:
        celex |= {c.upper() for c in ([ec] if isinstance(ec, str) else ec)}
    ea = t.get("expected_article")
    if ea:
        arts |= {article_key(a) for a in ([ea] if isinstance(ea, str) else ea)}
    legacy = t.get("expected_citation", "") or ""
    if legacy:
        celex |= set(re.findall(r"\b3\d{4}[A-Z]\d{4}\b", legacy))
        for year, num in re.findall(r"\b(20\d{2})/(\d{3,4})\b", legacy):
            celex.add(f"3{year}R{num.zfill(4)}")
            celex.add(f"3{year}L{num.zfill(4)}")
        arts |= {article_key(f"article {a}") for a in re.findall(r"article\s+(\d+[a-z]?)", legacy, re.I)}
    arts.discard("")
    return celex, arts


def doc_hit(doc, celex, arts):
    if (doc.get("celex") or "").upper() not in celex:
        return False
    if not arts:
        return True
    key = article_key(doc.get("article", "")) or article_key(doc.get("citation", ""), strict=True)
    return key in arts


def _squash_digits(s):
    return re.sub(r"(?<=\d)[,\u00a0\u202f](?=\d{3}(?!\d))", "", s or "")


def unsupported_claims(answer, context):
    """Lexical groundedness: every date, article/annex ref and number-with-unit (or 2+ digit number)
    in the answer must appear, as a whole token, in the context. Does NOT catch wrong prose claims."""
    ans, ctx = _squash_digits(answer), norm_text(_squash_digits(context))
    issues = []
    date_pat = re.compile(rf"\b\d{{1,2}}\s+(?:{MONTHS})\s+\d{{4}}\b", re.I)
    art_pat = re.compile(r"\barticles?\s+(\d+[a-z]?)\b", re.I)
    annex_pat = re.compile(r"\bannex\s+([ivxlc]+)\b", re.I)
    for m in date_pat.finditer(ans):
        if norm_text(m.group(0)) not in ctx:
            issues.append(f"date '{m.group(0)}'")
    for m in art_pat.finditer(ans):
        if not re.search(rf"\barticles?\s+{re.escape(m.group(1).lower())}\b", ctx):
            issues.append(f"article '{m.group(1)}'")
    for m in annex_pat.finditer(ans):
        if not re.search(rf"\bannex\s+{re.escape(m.group(1).lower())}\b", ctx):
            issues.append(f"annex '{m.group(1)}'")
    scrub = annex_pat.sub(" ", art_pat.sub(" ", date_pat.sub(" ", ans)))
    nums = set(re.findall(r"(?<![\w.])(\d{2,}(?:\.\d+)?)(?![\w])", scrub))
    nums |= set(re.findall(rf"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|{UNITS}\b)", scrub, re.I))
    for n in sorted(nums):
        if not re.search(rf"(?<![\d.]){re.escape(n)}(?!\d)", ctx):
            issues.append(f"number '{n}'")
    return issues


def parse_date(s):
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] {pathlib.Path(path).name} line {i}: invalid JSON ({e}); skipped")
    return out


# --------------------------------------------------------------------------- suite
class ComplianceTestSuite:
    def __init__(self, dataset_path, test_suite_path, rag_engine, prev_dataset_path=None, use_llm=True):
        self.engine = rag_engine
        self.use_llm = use_llm
        self.corpus = load_jsonl(dataset_path)
        tp = pathlib.Path(test_suite_path) if test_suite_path else None
        if tp and tp.exists():
            self.tests = load_jsonl(tp)
        else:
            self.tests = []
            print(f"  [WARN] test file not found ({tp}); only ingestion metrics will be meaningful")
        self.prev_corpus = load_jsonl(prev_dataset_path) if prev_dataset_path else None
        self._ret_cache = {}
        self._ans_cache = {}

    # ---- plumbing ---------------------------------------------------------
    def _banner(self, title):
        print("\n" + "=" * 74)
        print(title)
        print("=" * 74)

    def _row(self, label, value):
        print(f"  * {label:<50}: {value}")

    def _by_type(self, ttype):
        return [t for t in self.tests if t.get("type") == ttype]

    def _retrieve(self, query):
        """One search per query, top-10; slices give the top-2/5 results (ranking is identical)."""
        if query not in self._ret_cache:
            t0 = time.perf_counter()
            res = self.engine.search(query, top_k=max(KS))
            self._ret_cache[query] = (res, time.perf_counter() - t0)
        return self._ret_cache[query]

    def _answer(self, query, k=LLM_TOP_K):
        """One LLM call per (query, k), same prompt inputs as production: joined text + joined citations."""
        if not self.use_llm:
            return None
        key = (query, k)
        if key not in self._ans_cache:
            res, _ = self._retrieve(query)
            top = res[:k]
            if not top:
                self._ans_cache[key] = None
            else:
                context = "\n\n".join(r["text"] for r in top)
                cits = " | ".join(r.get("citation", "") for r in top)
                t0 = time.perf_counter()
                text = self.engine.generate_answer(query, context, cits, "")
                self._ans_cache[key] = {"text": text, "context": context, "latency": time.perf_counter() - t0}
        return self._ans_cache[key]

    @staticmethod
    def _ex(d):
        src = d.get("provenance", {}).get("source_file", "?")
        return f"{src} | {norm_text(d.get('text', ''))[:80]!r}"

    # ---- module A ---------------------------------------------------------
    def run_ingestion_metrics(self):
        self._banner("MODULE A - INGESTION")
        corpus, n = self.corpus, len(self.corpus)
        if not n:
            print("  corpus is empty")
            return

        # coverage: per-regulation chunk counts, not just presence
        counts = Counter((d.get("celex") or "") for d in corpus)
        covered = [c for c in TARGETS if counts.get(c, 0) >= MIN_CHUNKS_PER_TARGET]
        self._row(f"Regulatory coverage (>= {MIN_CHUNKS_PER_TARGET} chunks each)", fmt_rate(len(covered), len(TARGETS)))
        for c, name in TARGETS.items():
            k = counts.get(c, 0)
            flag = "" if k >= MIN_CHUNKS_PER_TARGET else ("   <- THIN" if k else "   <- MISSING")
            print(f"        {c}  {name:<24}{k:>6} chunks{flag}")
        no_celex = counts.get("", 0)
        files = Counter(d.get("provenance", {}).get("source_file", "?") for d in corpus)
        print(f"        chunks without CELEX (EBA/RBI/other): {no_celex}; source files: {len(files)}")
        print("        largest source files: " + ", ".join(f"{f} ({c})" for f, c in files.most_common(5)))

        # duplicates: missing hashes are NOT duplicates
        hashes = [d.get("provenance", {}).get("chunk_hash") for d in corpus]
        missing = sum(1 for h in hashes if not h)
        present = [h for h in hashes if h]
        dup_hash = len(present) - len(set(present))
        norm_counts = Counter(norm_text(d.get("text")) for d in corpus)
        dup_text = sum(c - 1 for c in norm_counts.values() if c > 1)
        self._row("Chunks missing provenance.chunk_hash", f"{missing}   (v1 counted these as duplicates)")
        self._row("Duplicate chunk_hash (missing excluded)", dup_hash)
        self._row("Unique by normalised text", fmt_rate(n - dup_text, n) + f"   dup chunks: {dup_text}")
        shown = 0
        for d in corpus:
            if norm_counts[norm_text(d.get("text"))] > 1 and shown < 3:
                print(f"        dup example: {self._ex(d)}")
                shown += 1

        # chunk integrity (heuristic, not accuracy against source)
        short, lower = [], []
        for d in corpus:
            txt = d.get("text", "") or ""
            if len(txt) < 50:
                short.append(d)
            elif re.match(r"^[a-z]", txt) and not txt.startswith(('"', "'")):
                lower.append(d)
        self._row("Chunk integrity (heuristic)", fmt_rate(n - len(short) - len(lower), n)
                  + f"   too short: {len(short)}, starts mid-sentence: {len(lower)}")
        for label, bucket in (("short", short), ("mid-sentence", lower)):
            for d in bucket[:3]:
                print(f"        {label}: {self._ex(d)}")

        # classification: schema validity vs real gold labels
        invalid = sum(1 for d in corpus if d.get("source_type") not in VALID_TYPES or d.get("legal_status") not in VALID_STATUS)
        self._row("Classification schema validity (NOT accuracy)", fmt_rate(n - invalid, n))
        gold = self._by_type("classification")
        if not gold:
            self._row("Classification accuracy vs gold labels", "N/A - add {'type':'classification',...} records")
        else:
            ok = 0
            for t in gold:
                docs = [d for d in corpus
                        if (t.get("source_file") and d.get("provenance", {}).get("source_file") == t["source_file"])
                        or (t.get("celex") and d.get("celex") == t["celex"])]
                bad = [d for d in docs
                       if d.get("source_type") != t.get("expected_source_type")
                       or (t.get("expected_legal_status") and d.get("legal_status") != t["expected_legal_status"])]
                if docs and not bad:
                    ok += 1
                else:
                    print(f"        [FAIL] classification {t.get('source_file') or t.get('celex')}: {len(bad)}/{len(docs)} chunks differ")
            self._row("Classification accuracy vs gold labels", fmt_rate(ok, len(gold)))

        self._change_detection()

    @staticmethod
    def _file_index(corpus):
        idx = defaultdict(lambda: {"file_hashes": set(), "chunk_hashes": set()})
        for d in corpus:
            p = d.get("provenance", {})
            f = p.get("source_file")
            if not f:
                continue
            if p.get("file_hash"):
                idx[f]["file_hashes"].add(p["file_hash"])
            if p.get("chunk_hash"):
                idx[f]["chunk_hashes"].add(p["chunk_hash"])
        return idx

    def _change_detection(self):
        cur = self._file_index(self.corpus)
        if self.prev_corpus is None:
            consistent = sum(1 for v in cur.values() if len(v["file_hashes"]) <= 1)
            self._row("File-hash consistency (NOT change detection)", fmt_rate(consistent, len(cur)))
            self._row("Change detection vs previous run", "N/A - copy knowledge_base.jsonl before re-ingesting, then pass --prev")
            return
        prev = self._file_index(self.prev_corpus)
        common = set(cur) & set(prev)
        unstable, review, ok = [], [], 0
        for f in sorted(common):
            same_file = cur[f]["file_hashes"] == prev[f]["file_hashes"]
            same_chunks = cur[f]["chunk_hashes"] == prev[f]["chunk_hashes"]
            if same_file and not same_chunks:
                unstable.append(f)
            elif not same_file and same_chunks:
                review.append(f)
            else:
                ok += 1
        self._row("Change detection vs previous run", fmt_rate(ok, len(common)))
        self._row("Files added / removed since previous run", f"{len(set(cur) - set(prev))} / {len(set(prev) - set(cur))}")
        for f in unstable:
            print(f"        [FAIL] file unchanged but chunk hashes differ (non-deterministic chunking): {f}")
        for f in review:
            print(f"        [REVIEW] file hash changed but no chunk changed: {f}")

    # ---- module B ---------------------------------------------------------
    def run_retrieval_metrics(self):
        self._banner("MODULE B - RETRIEVAL" + ("" if self.use_llm else "  (--no-llm: LLM metrics skipped)"))
        scorable, unscorable = [], 0
        for t in self._by_type("positive"):
            celex, arts = derive_expected(t)
            if celex:
                scorable.append((t, celex, arts))
            else:
                unscorable += 1
        strict = sum(1 for _, _, a in scorable if a)
        self._row("Positive tests scored", f"{len(scorable)} ({strict} by CELEX+article, {len(scorable) - strict} by CELEX only); unscorable: {unscorable}")

        hits_at, rr_sum = Counter(), 0.0
        search_lat, e2e_lat = [], []
        over_refusal = errors = e2e_ok = fn_any = answered_n = 0
        for t, celex, arts in scorable:
            q = t["query"]
            res, lat = self._retrieve(q)
            search_lat.append(lat)
            rank = next((i + 1 for i, d in enumerate(res) if doc_hit(d, celex, arts)), None)
            for k in KS:
                if rank is not None and rank <= k:
                    hits_at[k] += 1
            rr_sum += (1 / rank) if rank else 0.0
            if rank is None or rank > LLM_TOP_K:
                fn_any += 1
                got = "; ".join(f"{d.get('celex', '-')}/{d.get('article', '-')}" for d in res[:3]) or "no chunks"
                where = f"rank {rank}" if rank else "not in top 10"
                print(f"  [MISS] {where}: '{q}'\n         expected {sorted(celex)} {sorted(arts)}; top-3 got: {got}")
                continue
            ans = self._answer(q)
            if ans is None:
                continue
            if is_error(ans["text"]):
                errors += 1
                print(f"  [ERROR] Ollama: {ans['text'][:100]}")
                continue
            answered_n += 1
            e2e_lat.append(lat + ans["latency"])
            if is_refusal(ans["text"]):
                over_refusal += 1
                fn_any += 1
                print(f"  [OVER-REFUSAL] right chunk retrieved (rank {rank}) but LLM refused: '{q}'")
            else:
                e2e_ok += 1

        for k in KS:
            self._row(f"Retrieval hit@{k}", fmt_rate(hits_at[k], len(scorable)))
        self._row("MRR@10", f"{rr_sum / len(scorable):.3f}" if scorable else "N/A")
        if self.use_llm:
            self._row("Over-refusal (right chunk in top 2, LLM refused)", fmt_rate(over_refusal, answered_n))
            denom = len(scorable) - errors
            self._row("End-to-end success (hit@2 and answered)", fmt_rate(e2e_ok, denom))
            self._row("False-negative rate, any cause (v1 metric)", fmt_rate(fn_any, denom))
            self._row("Ollama errors (excluded from rates)", errors)

        # negatives
        negs = self._by_type("negative")
        gate = refused = fp = neg_err = reach_llm = 0
        for t in negs:
            q = t["query"]
            res, lat = self._retrieve(q)
            search_lat.append(lat)
            if not res:
                gate += 1
                continue
            reach_llm += 1
            ans = self._answer(q)
            if ans is None:
                continue
            if is_error(ans["text"]):
                neg_err += 1
                continue
            if is_refusal(ans["text"]):
                refused += 1
            else:
                fp += 1
                print(f"  [FALSE POSITIVE] answered out-of-scope query '{q}': {ans['text'][:100]!r}")
        if self.use_llm:
            tested = gate + refused + fp
            self._row("Negatives blocked by retrieval (no chunks)", fmt_rate(gate, tested))
            self._row("Negatives refused by the LLM", fmt_rate(refused, tested))
            self._row("False-positive rate (answered out-of-scope)", fmt_rate(fp, tested))
            if neg_err:
                self._row("Negatives with Ollama errors (excluded)", neg_err)
        else:
            self._row("Negatives blocked by retrieval", f"{gate}/{len(negs)}; {reach_llm} would reach the LLM (not tested)")

        if search_lat:
            self._row("Search latency p50 / p95 (ms)", f"{pct(search_lat, .5) * 1000:.1f} / {pct(search_lat, .95) * 1000:.1f}  (n={len(search_lat)})")
        if e2e_lat:
            self._row("End-to-end latency mean / p95 (s)", f"{sum(e2e_lat) / len(e2e_lat):.1f} / {pct(e2e_lat, .95):.1f}  (n={len(e2e_lat)})")

    # ---- module C ---------------------------------------------------------
    def run_generation_metrics(self):
        self._banner("MODULE C - GENERATION")
        if not self.use_llm:
            print("  skipped (--no-llm)")
            return
        tested = flagged = refused = no_ctx = errors = 0
        for t in self._by_type("positive"):
            q = t["query"]
            ans = self._answer(q)
            if ans is None:
                no_ctx += 1
                continue
            if is_error(ans["text"]):
                errors += 1
                continue
            if is_refusal(ans["text"]):
                refused += 1
                continue
            tested += 1
            issues = unsupported_claims(ans["text"], ans["context"])
            if issues:
                flagged += 1
                print(f"  [UNSUPPORTED] '{q}': {', '.join(issues[:5])}")
        self._row("Unsupported-claim rate (dates, articles, numbers)", fmt_rate(flagged, tested))
        self._row("Excluded from denominator", f"{refused} refusals, {no_ctx} no-chunk, {errors} errors")
        print("        note: lexical check. It catches invented figures/dates/article refs, not wrong prose claims.")

    # ---- module D ---------------------------------------------------------
    def run_extraction_metrics(self):
        self._banner("MODULE D - EXTRACTION (rule-based extractors in rag_engine.py)")
        # obligations
        tests = self._by_type("obligation")
        exp_total = exp_hit = full = p_tot = p_hit = 0
        for t in tests:
            found = self.engine.extract_obligations(t["text"])
            blob = " ".join(found).lower()
            exp = t.get("expected_obligations", [])
            hit = [e for e in exp if e.lower() in blob]
            exp_total += len(exp)
            exp_hit += len(hit)
            full += int(len(hit) == len(exp))
            if len(hit) < len(exp):
                print(f"  [FAIL] obligation missed: {[e for e in exp if e not in hit]}")
            if t.get("exhaustive"):
                p_tot += len(found)
                p_hit += sum(1 for s in found if any(e.lower() in s.lower() for e in exp))
        self._row("Obligation recall (per expected item)", fmt_rate(exp_hit, exp_total) if tests else "N/A (no test data)")
        if tests:
            self._row("Obligation tests fully matched", fmt_rate(full, len(tests)))
        if p_tot:
            self._row("Obligation precision (exhaustive tests only)", fmt_rate(p_hit, p_tot))

        # deadlines
        tests = self._by_type("deadline")
        exp_total = exp_hit = p_tot = p_hit = 0
        for t in tests:
            found = {norm_text(f) for f in self.engine.extract_deadlines(t["text"])}
            exp = [norm_text(e) for e in t.get("expected_deadlines", [])]
            hit = [e for e in exp if e in found]
            exp_total += len(exp)
            exp_hit += len(hit)
            if len(hit) < len(exp):
                print(f"  [FAIL] deadline missed: {[e for e in exp if e not in hit]}; found {sorted(found)}")
            if t.get("exhaustive"):
                p_tot += len(found)
                p_hit += len(found & set(exp))
        self._row("Deadline recall (per expected item)", fmt_rate(exp_hit, exp_total) if tests else "N/A (no test data)")
        if p_tot:
            self._row("Deadline precision (exhaustive tests only)", fmt_rate(p_hit, p_tot))

        # risk tier
        tests = self._by_type("risk")
        hits = 0
        for t in tests:
            got, want = self.engine.extract_risk_tier(t["text"]), t.get("expected_risk_tier", "").lower()
            if got == want:
                hits += 1
            else:
                print(f"  [FAIL] risk tier: expected {want!r}, got {got!r}: {norm_text(t['text'])[:70]!r}")
        self._row("Risk-tier match", fmt_rate(hits, len(tests)) if tests else "N/A (no test data)")
        if tests:
            print("        note: extractor is a keyword rule ('high-risk' anywhere => 'high'); this tests keyword presence.")

        # applicability (heuristic carried over from v1)
        tests = self._by_type("applicability")
        hits = total = skipped = 0
        for t in tests:
            expected = t.get("expected_applicable")
            qd, ed = parse_date(t.get("query_date")), parse_date(t.get("effective_date"))
            if expected is None or qd is None or ed is None:
                skipped += 1
                continue
            total += 1
            res, _ = self._retrieve(t["query"])
            if not res:
                continue
            text = res[0]["text"].lower()
            has_future = "applies from" in text or "shall apply" in text
            predicted = (not has_future) or (qd >= ed)
            hits += int(predicted == expected)
        if not tests:
            self._row("Applicability accuracy", "N/A - no {'type':'applicability'} records in the test file")
        else:
            self._row("Applicability accuracy (toy heuristic)", fmt_rate(hits, total) + f"   skipped (bad/missing fields): {skipped}")

    # ---- module E ---------------------------------------------------------
    def run_pipeline_metrics(self):
        self._banner("MODULE E - PIPELINE")
        if hasattr(self.engine, "_build_index"):
            buf = io.StringIO()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(buf):
                self.engine._build_index()
            dt = time.perf_counter() - t0
            self._row("Index rebuild time (new data -> searchable)", f"{dt:.2f}s for {len(self.engine.docs)} chunks")
        self._row("Alert delivery latency", "N/A (no alerting subsystem)")

    # ---- inspect ----------------------------------------------------------
    def inspect(self, query, k=10):
        self._banner(f"INSPECT: {query}")
        res, lat = self._retrieve(query)
        print(f"  search took {lat * 1000:.1f} ms; {len(res)} chunk(s) returned (the LLM sees the first {LLM_TOP_K})\n")
        for i, d in enumerate(res[:k], 1):
            mark = "  <- sent to LLM" if i <= LLM_TOP_K else ""
            print(f"{i:>2}. score={d.get('_score', 0):.3f}  celex={d.get('celex', '-')}  article={d.get('article', '-')!r}"
                  f"  type={d.get('source_type', '-')}{mark}")
            print(f"    citation: {d.get('citation', '')}")
            print(f"    {norm_text(d.get('text', ''))[:300]}\n")
        if not res:
            print("  no chunks passed the score/keyword gates (the LLM is not called in production for this?)")
        ans = self._answer(query)
        if ans:
            verdict = "ERROR" if is_error(ans["text"]) else "REFUSAL" if is_refusal(ans["text"]) else "ANSWER"
            print(f"  LLM [{verdict}, {ans['latency']:.1f}s]: {ans['text']}")
            if verdict == "ANSWER":
                print(f"  unsupported claims: {unsupported_claims(ans['text'], ans['context']) or 'none found'}")

    # ---- run --------------------------------------------------------------
    def run_all_tests(self):
        print("\n" + "#" * 74)
        print("COMPLIANCE METRICS SUITE v2")
        print("#" * 74)
        self.run_ingestion_metrics()
        self.run_retrieval_metrics()
        self.run_generation_metrics()
        self.run_extraction_metrics()
        self.run_pipeline_metrics()
        print("\n" + "-" * 74)
        print("Not measured here: the OFSI sanctions file (not loaded by RAGEngine), semantic answer")
        print("correctness (needs reference answers or a human/LLM judge), alerting, real applicability.")
        print("-" * 74 + "\n")


def main():
    base = pathlib.Path(__file__).parent.absolute()
    ap = argparse.ArgumentParser(description="Compliance RAG test suite v2")
    ap.add_argument("--corpus", default=str(base / "data" / "knowledge_base.jsonl"))
    ap.add_argument("--tests", default=str(base / "data" / "master_test_suite.jsonl"))
    ap.add_argument("--prev", help="previous knowledge_base.jsonl, for change detection")
    ap.add_argument("--no-llm", action="store_true", help="skip every Ollama call")
    ap.add_argument("--inspect", metavar="QUERY", help="show retrieved chunks and the answer for one query")
    args = ap.parse_args()

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not pathlib.Path(args.corpus).exists():
        print(f"Missing corpus: {args.corpus}")
        sys.exit(1)
    engine = RAGEngine(args.corpus)
    suite = ComplianceTestSuite(args.corpus, args.tests, engine, prev_dataset_path=args.prev, use_llm=not args.no_llm)
    if args.inspect:
        suite.inspect(args.inspect)
    else:
        suite.run_all_tests()


if __name__ == "__main__":
    main()