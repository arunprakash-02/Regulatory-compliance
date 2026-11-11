# Architecture

## Overview

This project implements a **hybrid retrieval-augmented generation (RAG)** pipeline optimized for offline, CPU-only deployment on regulatory text.

## Pipeline Stages

### 1. Ingestion (`build_dataset.py`)
- **Fetch:** `requests` with retry/backoff (EU Cellar API, Sanctions Map, OFAC)
- **Extract (PDF):** `pymupdf` (validates `%PDF` magic bytes first)
- **Extract (XML):** `lxml` + `BeautifulSoup` (preserves paragraph breaks and article boundaries)
- **Extract (HTML):** `BeautifulSoup` (strips nav/footer/script)
- **Chunk:** Custom sentence-aware chunker (800 chars, 150 overlap, Article-boundary splits)
- **Deduplicate:** SHA-256 hash of `file_hash + chunk_hash`
- **Persist:** Atomic write (`.tmp` → `os.replace`)

### 2. Retrieval (`rag_engine.py::search`)
- **Stage 1 (TF-IDF + SVD):** `TfidfVectorizer` (ngram_range=(1,2), max_features=15000) reduced to 150 dimensions via `TruncatedSVD`. Top-20 candidates above score threshold 0.15.
- **Stage 2 (Keyword Gate):** Requires minimum content-word matches (len ≥ 4).
- **Stage 3 (Source Boost):** +0.10 for chunks matching the explicitly mentioned regulation.
- **Stage 4 (Definition Boost):** +0.30 for definitional markers ("shall be considered to be...") on definition queries.
- **Stage 5:** Returns top-K (default 3).

### 3. Generation (`rag_engine.py::generate_answer`)
- Queries local Llama 3.2 3B via Ollama `/api/chat`.
- Strict system prompt with automatic retry-on-refusal logic for definitional queries using expanded context.
- Settings: `temperature=0.0`, `num_predict=250`.

### 4. Evaluation (`test_compliance_suite_v1.py`)
- Five modules (Ingestion, Retrieval, Generation, Extraction, Pipeline) annotated with Wilson 95% confidence intervals and sample size warnings.
