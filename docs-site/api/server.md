# REST Server

Production-ready FastAPI server exposing Director-AI scoring over HTTP.

## Starting the Server

=== "CLI"

    ```bash
    director-ai serve --port 8080 --workers 4
    ```

=== "Python"

    ```python
    from director_ai.server import create_app

    app = create_app()
    # Run with: uvicorn director_ai.server:app --host 0.0.0.0 --port 8080
    ```

=== "Docker"

    ```bash
    docker run -p 8080:8080 ghcr.io/anulum/director-ai:latest
    ```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/review` | Score a prompt/response pair |
| `POST` | `/v1/batch` | Batch score multiple pairs |
| `GET` | `/v1/health` | Health check (NLI status, cache stats) |
| `GET` | `/v1/metrics` | Metrics as JSON |
| `GET` | `/v1/metrics/prometheus` | Prometheus-compatible metrics |
| `GET` | `/v1/source` | Source code URL (AGPL compliance) |
| `WS` | `/v1/stream` | WebSocket streaming oversight |

## Review Request

```bash
curl -X POST http://localhost:8080/v1/review \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{
    "prompt": "What is the refund policy?",
    "response": "Refunds within 30 days.",
    "session_id": "optional-session-id"
  }'
```

### Response

```json
{
  "approved": true,
  "score": 0.85,
  "h_logical": 0.10,
  "h_factual": 0.15,
  "warning": false,
  "evidence": {
    "chunks": [
      {"text": "Refunds within 30 days of purchase.", "distance": 0.12}
    ]
  }
}
```

## Authentication

Set `api_keys` in config or via `DIRECTOR_API_KEYS` env var (comma-separated):

```bash
DIRECTOR_API_KEYS=key1,key2 director-ai serve
```

Clients send `X-API-Key: key1` header. Unauthenticated requests receive 401.

## Rate Limiting

```bash
DIRECTOR_RATE_LIMIT_RPM=60 director-ai serve
```

Returns 429 when exceeded. Install `pip install director-ai[ratelimit]` for Redis-backed distributed rate limiting.

## CORS

```bash
DIRECTOR_CORS_ORIGINS=https://example.com,https://app.example.com director-ai serve
```

Default is empty (no CORS). Do not use `*` in production.

## Continuous Batching (ReviewQueue)

For high-concurrency deployments, enable server-level request accumulation:

```bash
DIRECTOR_REVIEW_QUEUE_ENABLED=1 \
DIRECTOR_REVIEW_QUEUE_MAX_BATCH=32 \
DIRECTOR_REVIEW_QUEUE_FLUSH_TIMEOUT_MS=10 \
director-ai serve
```

The queue collects concurrent `/v1/review` requests and flushes them as a single `review_batch()` call, reducing GPU kernel launches from 2×N to 2 per flush window.

## Full API

::: director_ai.server.create_app
