# Troubleshooting

## Common Issues

### 1. Ollama Connection Refused
- **Symptom:** Error connecting to Ollama: `[Errno 111] Connection refused`
- **Fix:** Ensure Ollama is running (`ollama ps` or start the Ollama desktop app).

### 2. Missing Python Dependencies (`lxml`, `pymupdf`)
- **Fix:** Install pre-built wheels:
  ```bash
  pip install --only-binary :all: lxml pymupdf
  ```

### 3. Flask Address Already in Use
- **Fix:** Kill the existing process on port 8001 or change `FLASK_PORT` in `.env`.
