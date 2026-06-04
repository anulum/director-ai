# Quickstart - Director-AI Proxy in 2 Minutes

## Setup

Recommended:

```bash
director-ai quickstart --run
```

This creates `director_guard/` with a facts file, local config, a
standalone Python guard, Docker Compose, Chroma persistence, and an
optional FactCG ONNX profile.

Manual repo-local path:

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
curl http://localhost:8080/health

# Score a response
curl -X POST http://localhost:8080/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the refund policy?",
    "response": "Refunds are available within 30 days."
  }'
```

## Use as Chat Proxy

Point any chat-completions-compatible client at the proxy:

```python
from provider_sdk import Client

client = Client(base_url="http://localhost:8080/v1")
response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "What is the refund policy?"}],
)
```

Set the provider API key in the environment or pass `--upstream-url` to point
at a different LLM backend such as vLLM, Ollama, or llama.cpp.

## Generated Compose Path

The CLI-generated quickstart runs two default services:

| Service | Port | Purpose |
|---------|------|---------|
| `director-proxy` | 8080 | Guarded chat proxy with `facts.txt` |
| `director-api` | 8000 | FastAPI service with local Chroma persistence |

The ONNX scorer is opt-in:

```bash
cd director_guard
docker compose --profile onnx up director-proxy-onnx
```

Place exported FactCG ONNX files in `models/factcg-onnx/` before enabling
that profile. Keeping it behind a profile makes the default path usable on
CPU-only machines without installing the heavy NLI stack.

## Production Scaffold

For an authenticated production scaffold generated from the installed CLI:

```bash
director-ai quickstart --profile production
cd director_guard
```

Fill `.env` with `DIRECTOR_API_KEY_TENANT_MAP`, `DIRECTOR_PROXY_API_KEYS`,
`DIRECTOR_LLM_API_URL`, `DIRECTOR_UPSTREAM_URL`, `DIRECTOR_KB_HMAC_KEYS`, and exact
`DIRECTOR_CORS_ORIGINS`, then start:

```bash
docker compose up
```

The production scaffold enables tenant-bound auth, signed KB writes, audit,
compliance and feedback stores, JSON logs, rate limiting, review queueing, and
authenticated Prometheus metrics. To run Prometheus, write the matching API key
to `secrets/director-api-key` and use:

```bash
docker compose --profile monitoring up
```

## Knowledge Base

Edit `kb.txt` — one fact per line, format `key: value`. The proxy reloads
facts on startup. Restart `docker compose` after changes.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | 0.3 | Minimum coherence score (0.0–1.0) |
| `--on-fail` | reject | `reject` (HTTP 422) or `warn` (pass with headers) |
| `--upstream-url` | Provider default | LLM backend URL |
| `--facts` | — | Path to knowledge base file |
