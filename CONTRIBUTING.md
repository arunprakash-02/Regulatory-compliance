# Contributing

This is a personal Proof of Concept. Contributions are welcome but not expected.

## Development Setup

```bash
git clone https://github.com/yourusername/bfsi-compliance-rag.git
cd bfsi-compliance-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Before Submitting a PR
- Run the evaluation suite — `python test_compliance_suite_v1.py --no-llm` should pass without errors.
- Check code style — keep functions ≤50 lines and add docstrings.
- Update `CHANGELOG.md` — add an "Unreleased" section entry.
- Add a test record if your PR changes retrieval or generation.
