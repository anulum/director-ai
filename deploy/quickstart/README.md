# Quickstart — Director-AI Proxy in 2 Minutes

## Setup

```bash
cd deploy/quickstart
docker compose up
```

This starts a Director-AI proxy on port 8080 with `kb.txt` as the knowledge base.
Every LLM response routed through the proxy is scored against these facts.
Hallucinations are rejected (HTTP 422).

## Test

```bash
# Health check
curl http://localhost:8080/v1/health

# Score a response
curl -X POST http://localhost:8080/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the refund policy?",
    "response": "Refunds are available within 30 days."
  }'
```

## Use as OpenAI Proxy

Point any OpenAI-compatible client at the proxy:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the refund policy?"}],
)
```

Set `OPENAI_API_KEY` in the environment or pass `--upstream-url` to point at
a different LLM backend (vLLM, Ollama, etc.).

## Knowledge Base

Edit `kb.txt` — one fact per line, format `key: value`. The proxy reloads
facts on startup. Restart `docker compose` after changes.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | 0.3 | Minimum coherence score (0.0–1.0) |
| `--on-fail` | reject | `reject` (HTTP 422) or `warn` (pass with headers) |
| `--upstream-url` | OpenAI | LLM backend URL |
| `--facts` | — | Path to knowledge base file |
