# Quick Start Guide

This guide takes you from zero to a running compliance Q&A system in under 15 minutes.

## Prerequisites
- Python 3.11+
- Ollama installed and running
- ~4 GB free disk space

## Step-by-Step
1. **Install Ollama and Pull Models:**
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
2. **Clone and Setup:**
   ```bash
   git clone https://github.com/yourusername/bfsi-compliance-rag.git
   cd bfsi-compliance-rag
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Ingest and Run:**
   ```bash
   python build_dataset.py
   python app.py
   ```
   Open `http://127.0.0.1:8001` in your browser.
