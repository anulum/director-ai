# OpenAI-Compatible Proxy

The guardrail proxy fronts any OpenAI-compatible API and scores
responses for hallucination before they reach the caller. Point a
client at it with `OPENAI_BASE_URL` and no code changes:

```bash
director-ai proxy --port 8080 --facts kb.txt --threshold 0.6
export OPENAI_BASE_URL=http://localhost:8080/v1
```

Build it programmatically with `create_proxy_app` (reference below):

```python
from director_ai.proxy import create_proxy_app

app = create_proxy_app(
    threshold=0.6,
    facts_path="kb.txt",
    upstream_url="https://api.openai.com",
    on_fail="reject",          # or "warn"
    moderations="local",       # or "upstream"
)
```

## Routes

| Route | Behaviour |
|-------|-----------|
| `POST /v1/chat/completions` | Forwarded upstream; the assistant message (or the accumulated stream) is scored. `on_fail="reject"` returns HTTP 422 (`content_filter`) on hallucination; streams are halted mid-flight with a `content_filter` finish reason. |
| `POST /v1/completions` | Legacy text completions, same scoring flow as chat — non-streaming and streaming (`choices[0].text` deltas). |
| `POST /v1/moderations` | `moderations="local"` (default) analyses every input with the shipped dependency-free detectors and answers in the OpenAI moderations shape; `"upstream"` forwards the request verbatim. |
| `POST /v1/embeddings` | Plain passthrough — embeddings carry no natural-language claims to verify. |
| `GET /v1/models` | Plain passthrough. |
| `GET /health` | Proxy status, threshold, and failure mode. |

Scored responses carry `X-Director-Score` and `X-Director-Approved`
headers; rejected ones return the OpenAI error shape with
`"type": "content_filter"`.

## Local moderations

Local mode needs no upstream moderations endpoint — it works in front
of vLLM, llama.cpp, or any self-hosted gateway. Each input is analysed
by `KeywordToxicityDetector` (word-boundary seed list plus attack
patterns) and `RegexPIIDetector` (email, phone, credit card, SSN, PHI,
IBAN, passport, IPv4). The response follows the OpenAI shape with
Director's category names:

```json
{
  "id": "modr-…",
  "model": "director-ai-local-moderation",
  "results": [
    {
      "flagged": true,
      "categories": {"email": true},
      "category_scores": {"email": 1.0}
    }
  ]
}
```

`input` accepts a string or a non-empty list of strings; anything else
returns HTTP 400 in the OpenAI error shape. Category names are
Director's own (`keyword`, `threat`, `self_harm_encouragement`, plus
the PII categories above) — clients that only read `flagged` need no
changes.

## CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--port` | `8080` | Listen port. |
| `--threshold` | `0.6` | Coherence threshold. |
| `--facts` / `--facts-root` | — | Ground-truth facts file (and its allowed root). |
| `--upstream-url` | `https://api.openai.com` | Upstream base URL (HTTPS enforced unless `--allow-http-upstream`). |
| `--on-fail` | `reject` | `reject` (HTTP 422) or `warn` (forward with headers). |
| `--api-keys` | — | Comma-separated keys; clients must send `X-API-Key`. |
| `--moderations` | `local` | `local` or `upstream`. |
| `--audit-db` | — | SQLite compliance audit database path. |
| `--config-env` | off | Build the scorer from `DIRECTOR_*` environment configuration. |

## API reference

::: director_ai.proxy.create_proxy_app
