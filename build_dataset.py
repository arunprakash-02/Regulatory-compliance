import json, pathlib, re, time, hashlib, os, csv
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
import pymupdf
from urllib.parse import urljoin, urlparse

# --- CONFIG (Portable Path) ---
BASE_DIR = pathlib.Path(__file__).parent.absolute() / "data"
RAW_DIR = BASE_DIR / "raw"
OUT_FILE = BASE_DIR / "knowledge_base.jsonl"
OFSI_OUT_FILE = BASE_DIR / "ofsi_sanctions.jsonl"
URL_LEDGER_FILE = BASE_DIR / "downloaded_urls.json"
FAILED_URLS_FILE = BASE_DIR / "failed_urls.jsonl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_FILES_PER_PORTAL = 5
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

CRAWL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/pdf,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9"
}

CELEX_ACTS = {
    "32024R1624": "EU_AML - AML Regulation",
    "32024L1640": "EU_AML - AML Directive",
    "32024R1620": "EU_AML - AMLA Regulation",
    "32024R1689": "EU_AI - AI Act",
    "32024R1623": "CRR3 - Basel III Regulation",
    "32016R0679": "GDPR - General Data Protection Regulation"
}

REGULATION_TITLES = {
    "32024R1624": "Regulation (EU) 2024/1624 on the prevention of the use of the financial system for ML/TF",
    "32024L1640": "Directive (EU) 2024/1640 on the mechanisms to be put in place by Member States for the prevention of ML/TF",
    "32024R1620": "Regulation (EU) 2024/1620 establishing the Authority for Anti-Money Laundering and Countering the Financing of Terrorism",
    "32024R1689": "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (AI Act)",
    "32024R1623": "Regulation (EU) 2024/1623 amending Regulation (EU) No 575/2013 (CRR3)",
    "32016R0679": "Regulation (EU) 2016/679 (General Data Protection Regulation - GDPR)"
}

INSTITUTIONAL_PORTALS = [
    ("AMLA", "https://www.amla.europa.eu/policy/regulatory-instruments_en"),
    ("EDPB", "https://www.edpb.europa.eu/")
]

TARGET_KEYWORDS = ["final report", "rts", "its", "guideline", "consultation paper", "opinion", "recommendation", "guidance"]

OFSI_FALLBACK_URLS = [
    "https://sanctionslist.fcdo.gov.uk/docs/UK-Sanctions-List.xml",
    "https://sanctionslist.fcdo.gov.uk/docs/UK-Sanctions-List.csv",
    "https://sanctionslist.fcdo.gov.uk/docs/UK-Sanctions-List.html",
]

# --- HELPER: SIDECAR METADATA ---
def save_sidecar(file_path, url):
    meta = {"source_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat()}
    sidecar = pathlib.Path(str(file_path) + ".meta.json")
    with open(sidecar, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

def load_sidecar(file_path):
    sidecar = pathlib.Path(str(file_path) + ".meta.json")
    if sidecar.exists():
        try:
            with open(sidecar, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: pass
    return {}

def log_failure(url, reason):
    record = {"url": url, "reason": str(reason), "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(FAILED_URLS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")

def load_url_ledger():
    if URL_LEDGER_FILE.exists():
        try:
            with open(URL_LEDGER_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except: pass
    return set()

def save_url_ledger(ledger):
    with open(URL_LEDGER_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(ledger), f, indent=2)

# --- FETCH WITH RETRY ---
def fetch_with_retry(url, headers=None, timeout=30, max_retries=3):
    h = headers or HEADERS
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=h, timeout=timeout)
            ct = resp.headers.get('Content-Type', '').lower()
            if any(bad in ct for bad in ['image/', 'video/', 'audio/', 'application/zip']):
                log_failure(url, f"Unsupported Content-Type: {ct}")
                return None
                
            if resp.status_code == 200:
                if len(resp.content) > MAX_FILE_SIZE_BYTES:
                    log_failure(url, f"Downloaded file too large: {len(resp.content)} bytes")
                    return None
                return resp
            elif resp.status_code in [429, 500, 502, 503, 504]:
                wait = (2 ** attempt) + 1
                time.sleep(wait)
            else:
                log_failure(url, f"HTTP Status {resp.status_code}")
                return None
        except Exception as e:
            wait = (2 ** attempt) + 1
            time.sleep(wait)
            
    log_failure(url, "Max retries reached")
    return None

def clean_text(text, preserve_newlines=False):
    text = str(text)
    if preserve_newlines:
        text = re.sub(r'[^\S\n]+', ' ', text) 
        text = re.sub(r'\n+', '\n', text)     
    else:
        text = re.sub(r'\s+', ' ', text)      
    return text.strip()

# --- JSON EXTRACTOR ---
def extract_all_json_strings(data, min_length=15):
    texts = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() in ['id', 'code', 'url', 'link', 'color', 'type', 'hash', 'icon']:
                continue
            if isinstance(v, str) and len(v) > min_length:
                texts.append(v)
            elif isinstance(v, (dict, list)):
                texts.extend(extract_all_json_strings(v, min_length))
    elif isinstance(data, list):
        for item in data:
            texts.extend(extract_all_json_strings(item, min_length))
    return texts

# --- CHUNKING (Fixed) ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text: return []
    
    article_blocks = re.split(r'(?=\n\s*Article\s+\d+[a-zA-Z]?\b)', text)
    results = []
    
    def _trim_to_uppercase(chunk):
        chunk = chunk.strip()
        for _ in range(5):
            if not chunk: return ""
            if chunk[0].isupper() or chunk[0].isdigit() or chunk[0] in '"([':
                return chunk
            first_space = chunk.find(' ')
            if first_space == -1 or first_space > 50:
                return ""
            chunk = chunk[first_space + 1:].strip()
        return chunk
    
    for block in article_blocks:
        block = block.strip()
        if not block: continue
        
        art_match = re.match(r'^(Article\s+\d+[a-zA-Z]?)\b', block, re.IGNORECASE)
        article_name = art_match.group(1) if art_match else ""
        
        sentences = re.split(r'(?<=[.!?])\s+', block)
        
        current_chunk = ""
        for sentence in sentences:
            if len(sentence) > chunk_size:
                cleaned = _trim_to_uppercase(current_chunk)
                if len(cleaned) >= 100:
                    results.append((cleaned, article_name))
                
                start = 0
                while start < len(sentence):
                    end = min(start + chunk_size, len(sentence))
                    if end < len(sentence):
                        last_space = sentence.rfind(' ', start, end)
                        if last_space > start + (chunk_size // 2):
                            end = last_space
                    
                    chunk = sentence[start:end].strip()
                    if start > 0:
                        chunk = _trim_to_uppercase(chunk)
                    
                    if len(chunk) >= 100 and (chunk[0].isupper() or chunk[0].isdigit() or chunk[0] in '"(['):
                        results.append((chunk, article_name))
                    
                    if end >= len(sentence): break
                    
                    start = end - overlap
                    if start < 0: start = 0
                    next_space = sentence.find(' ', start)
                    if next_space != -1 and next_space < end:
                        start = next_space + 1
                
                current_chunk = ""
                continue
            
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                cleaned = _trim_to_uppercase(current_chunk)
                if len(cleaned) >= 100:
                    results.append((cleaned, article_name))
                current_chunk = sentence
        
        cleaned = _trim_to_uppercase(current_chunk)
        if len(cleaned) >= 100:
            results.append((cleaned, article_name))
            
    return results

def ensure_dirs():
    for folder in ["EURLEX", "SANCTIONS_MAP", "OFAC", "OFSI", "INSTITUTIONAL"]:
        (RAW_DIR / folder).mkdir(parents=True, exist_ok=True)

# --- PHASES 1-4: FETCHING ---
def fetch_eurlex():
    print("\n--- Phase 1: EUR-Lex (Cellar API) ---")
    for celex, desc in CELEX_ACTS.items():
        url = f"http://publications.europa.eu/resource/celex/{celex}"
        headers = dict(HEADERS)
        headers["Accept"] = "application/xhtml+xml, application/xml;q=0.9, application/pdf, text/html;q=0.8"
        resp = fetch_with_retry(url, headers=headers)
        if not resp: continue
            
        ct = resp.headers.get('Content-Type', '').lower()
        ext = ".xhtml"
        if "pdf" in ct: ext = ".pdf"
        elif "xml" in ct: ext = ".xml"
        elif "html" in ct: ext = ".html"
        
        save_path = RAW_DIR / "EURLEX" / f"EURLEX_{celex}{ext}"
        if save_path.exists(): continue
        
        with open(save_path, 'wb' if ext == '.pdf' else 'w', encoding=None if ext == '.pdf' else 'utf-8') as f:
            f.write(resp.content if ext == '.pdf' else resp.text)
        save_sidecar(save_path, url)
        print(f"    -> Saved {celex}{ext}")

def fetch_sanctions_map():
    print("\n--- Phase 2: EU Sanctions Map API ---")
    url = "https://www.sanctionsmap.eu/api/v1/data"
    resp = fetch_with_retry(url)
    if not resp: return
    
    save_path = RAW_DIR / "SANCTIONS_MAP" / "discovery.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(resp.json(), f, indent=2)
    save_sidecar(save_path, url)
        
    verified_regimes = [50]
    for rid in verified_regimes:
        r_url = f"https://www.sanctionsmap.eu/api/v1/regime/{rid}"
        r_resp = fetch_with_retry(r_url)
        if r_resp:
            r_save = RAW_DIR / "SANCTIONS_MAP" / f"regime_{rid}.json"
            with open(r_save, 'w', encoding='utf-8') as f:
                json.dump(r_resp.json(), f, indent=2)
            save_sidecar(r_save, r_url)
            print(f"    -> Saved regime {rid}")

def fetch_ofac():
    print("\n--- Phase 3: OFAC SLS API ---")
    url = "https://sanctionslistservice.ofac.treas.gov/sanctions-programs"
    resp = fetch_with_retry(url)
    if resp:
        save_path = RAW_DIR / "OFAC" / "OFAC_programs.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(resp.json(), f, indent=2)
        save_sidecar(save_path, url)
        print("  [SAVED] OFAC metadata")

def fetch_ofsi_data():
    print("\n--- Phase 3b: OFSI UK Sanctions List ---")
    save_dir = RAW_DIR / "OFSI"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if list(save_dir.glob("OFSI_UK-Sanctions-List.*")):
        print("  [SKIP] OFSI UK Sanctions List already exists.")
        return

    for url in OFSI_FALLBACK_URLS:
        print(f"  [TRY] {url}")
        resp = fetch_with_retry(url)
        if resp:
            ext = ".xml"
            if ".csv" in url or "csv" in resp.headers.get("Content-Type", "").lower():
                ext = ".csv"
            elif ".html" in url or "html" in resp.headers.get("Content-Type", "").lower():
                ext = ".html"
            
            save_path = save_dir / f"OFSI_UK-Sanctions-List{ext}"
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(resp.text)
            save_sidecar(save_path, url)
            print(f"  [SAVED] OFSI_UK-Sanctions-List{ext} ({len(resp.text)} chars)")
            break

def crawl_institutional_portals():
    print("\n--- Phase 4: Institutional Portals ---")
    downloaded_urls = load_url_ledger()
    
    for source, base_url in INSTITUTIONAL_PORTALS:
        print(f"  [CRAWL] {source}")
        try:
            resp = fetch_with_retry(base_url, headers=CRAWL_HEADERS)
            if not resp: continue
                
            base_domain = urlparse(base_url).netloc
            soup = BeautifulSoup(resp.text, 'html.parser')
            files_downloaded = 0
            
            for a in soup.find_all('a', href=True):
                if files_downloaded >= MAX_FILES_PER_PORTAL: break
                
                href = a['href']
                full_url = urljoin(base_url, href)
                
                if urlparse(full_url).netloc != base_domain: continue
                if full_url in downloaded_urls: continue
                    
                text = a.get_text().lower()
                if not any(kw in text or kw in href.lower() for kw in TARGET_KEYWORDS): continue
                    
                is_pdf = full_url.lower().endswith('.pdf')
                url_hash = hashlib.md5(full_url.encode()).hexdigest()[:8]
                
                check_ext = ".pdf" if is_pdf else ".html"
                check_path = RAW_DIR / "INSTITUTIONAL" / f"{source}_{url_hash}{check_ext}"
                
                if check_path.exists() or (RAW_DIR / "INSTITUTIONAL" / f"{source}_{url_hash}.xml").exists():
                    continue
                
                file_resp = fetch_with_retry(full_url, headers=CRAWL_HEADERS)
                if not file_resp: continue
                    
                ct = file_resp.headers.get('Content-Type', '').lower()
                if "pdf" in ct: ext = ".pdf"
                elif "html" in ct: ext = ".html"
                elif "xml" in ct: ext = ".xml"
                else: continue
                    
                filename = f"{source}_{url_hash}{ext}"
                save_path = RAW_DIR / "INSTITUTIONAL" / filename
                
                with open(save_path, 'wb' if ext == '.pdf' else 'w', encoding=None if ext == '.pdf' else 'utf-8') as f:
                    f.write(file_resp.content if ext == '.pdf' else file_resp.text)
                    
                save_sidecar(save_path, full_url)
                downloaded_urls.add(full_url)
                save_url_ledger(downloaded_urls)
                
                files_downloaded += 1
                print(f"    -> Downloaded {filename}")
                time.sleep(1)
        except Exception as e:
            print(f"    [WARN] Portal {source} failed: {e}. Continuing...")

# --- PHASE 5: EXTRACTION & CHUNKING ---
def process_raw_files():
    print("\n--- Phase 5: Extraction & Chunking ---")
    documents = []
    ofsi_documents = []
    doc_id = 0
    ofsi_doc_id = 0
    seen_hashes = set()

    for folder in RAW_DIR.iterdir():
        if not folder.is_dir(): continue
        
        source_type = "authoritative_legal_text" if folder.name == "EURLEX" else "institutional_guidance"
        source_subtype = ""
        legal_status = "adopted"
        
        if folder.name == "SANCTIONS_MAP":
            source_type = "supplementary_sanctions_metadata"
            source_subtype = "sanctions_regime_metadata"
            legal_status = "metadata"
        elif folder.name == "OFAC":
            source_type = "supplementary_sanctions_metadata"
            source_subtype = "sanctions_program_codes"
            legal_status = "metadata"
        elif folder.name == "OFSI":
            source_type = "supplementary_sanctions_metadata"
            source_subtype = "uk_sanctions_list"
            legal_status = "metadata"
        elif folder.name == "INSTITUTIONAL":
            legal_status = "draft_or_guidance"

        for file_path in folder.glob("*"):
            if file_path.name.endswith(".meta.json"): continue
            if file_path.suffix.lower() not in ['.pdf', '.html', '.xhtml', '.xml', '.json', '.csv']: continue
                
            print(f"  Processing: {folder.name}/{file_path.name}")
            file_bytes = file_path.read_bytes()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            
            text = ""
            celex = ""
            regulation_title = ""
            
            if folder.name == "EURLEX":
                m = re.search(r'(\d{5}[A-Z]\d{4})', file_path.name)
                if m:
                    celex = m.group(1)
                    regulation_title = REGULATION_TITLES.get(celex, "")

            if file_path.suffix.lower() == '.pdf':
                if file_bytes[:4] != b'%PDF':
                    continue
                try:
                    doc = pymupdf.open(file_path)
                    text = clean_text(" ".join([page.get_text("text") for page in doc]))
                except Exception as e: print(f"    -> PDF Error: {e}")
                
            elif file_path.suffix.lower() in ['.html', '.xhtml', '.xml']:
                try:
                    with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                        raw_text = f.read()
                    if file_path.suffix.lower() == '.xml':
                        try: soup = BeautifulSoup(raw_text, 'lxml-xml')
                        except Exception: soup = BeautifulSoup(raw_text, 'html.parser')
                    else:
                        soup = BeautifulSoup(raw_text, 'html.parser')
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header']): tag.decompose()
                    text = clean_text(soup.get_text(separator='\n'), preserve_newlines=True)
                except Exception as e: print(f"    -> HTML/XML Error: {e}")
                
            elif file_path.suffix.lower() in ['.json', '.csv']:
                try:
                    with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                        if file_path.suffix.lower() == '.json':
                            data = json.load(f)
                            if folder.name in ["SANCTIONS_MAP", "OFAC"]:
                                text = clean_text(" ".join(extract_all_json_strings(data)))
                        else: 
                            text = clean_text(" ".join([" ".join(row) for row in csv.reader(f)]))
                except Exception as e: print(f"    -> JSON/CSV Error: {e}")

            if not text or len(text) < 100: continue

            chunks_with_articles = chunk_text(text)
            
            meta = load_sidecar(file_path)
            source_url = meta.get("source_url", "")
            retrieved_at = meta.get("retrieved_at", "")
            citation = regulation_title if regulation_title else file_path.stem.replace('_', ' ').title()

            for chunk, article in chunks_with_articles:
                chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
                dedup_key = f"{file_hash}_{chunk_hash}"
                
                if dedup_key in seen_hashes: continue
                seen_hashes.add(dedup_key)
                
                record = {
                    "schema_version": "1.2",
                    "citation": citation,
                    "regulation": regulation_title,
                    "text": chunk,
                    "language": "en",
                    "celex": celex,
                    "article": article,
                    "legal_status": legal_status,
                    "provenance": {
                        "source_file": file_path.name,
                        "source_url": source_url,
                        "retrieved_at": retrieved_at,
                        "file_hash": file_hash,
                        "chunk_hash": chunk_hash
                    }
                }
                
                if folder.name == "OFSI":
                    record["id"] = f"ofsi_{ofsi_doc_id}"
                    record["source_type"] = source_type
                    record["source_subtype"] = source_subtype
                    ofsi_documents.append(record)
                    ofsi_doc_id += 1
                else:
                    record["id"] = f"doc_{doc_id}"
                    record["source_type"] = source_type
                    record["source_subtype"] = source_subtype
                    documents.append(record)
                    doc_id += 1

    if documents:
        tmp_out = pathlib.Path(str(OUT_FILE) + ".tmp")
        with open(tmp_out, 'w', encoding='utf-8') as f:
            for doc in documents: f.write(json.dumps(doc) + "\n")
        os.replace(tmp_out, OUT_FILE)
        print(f"\n✅ SUCCESS: Saved {len(documents)} UNIQUE chunks to {OUT_FILE}")
    
    if ofsi_documents:
        tmp_ofsi = pathlib.Path(str(OFSI_OUT_FILE) + ".tmp")
        with open(tmp_ofsi, 'w', encoding='utf-8') as f:
            for doc in ofsi_documents: f.write(json.dumps(doc) + "\n")
        os.replace(tmp_ofsi, OFSI_OUT_FILE)
        print(f"✅ SUCCESS: Saved {len(ofsi_documents)} UNIQUE OFSI chunks to {OFSI_OUT_FILE}")

def generate_cfo_schema():
    schema = {"obligation_id": "string", "regulation": "string", "jurisdiction": "string", "affected_business_line": "string", "customer_segment": "string", "product": "string", "corridor": "string", "compliance_cost": "float", "revenue_at_risk": "float", "cash_outflow": "float", "provision": "float", "capital_impact": "float", "roe_impact": "float", "verification_status": "string"}
    with open(BASE_DIR / "cfo_exposure_schema.json", 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2)

if __name__ == "__main__":
    ensure_dirs()
    fetch_eurlex()
    fetch_sanctions_map()
    fetch_ofac()
    fetch_ofsi_data()
    crawl_institutional_portals()
    process_raw_files()
    generate_cfo_schema()
    print("\n🚀 PIPELINE COMPLETE.")