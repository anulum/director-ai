# FastAPI Server

Production server with auth, rate limiting, batch, streaming, and metrics.

## `create_app(config) -> FastAPI`

```python
from director_ai.server import create_app
app = create_app(DirectorConfig.from_profile("thorough"))  # pip install director-ai[server]
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/review` | Score an existing LLM response |
| POST | `/v1/process` | Generate + score (best of `max_candidates`) |
| POST | `/v1/batch` | Process up to 1000 prompts concurrently |
| GET | `/v1/health` | Version, profile, NLI status, uptime |
| GET | `/v1/metrics` | JSON metrics snapshot |
| GET | `/v1/metrics/prometheus` | Prometheus exposition format |
| GET | `/v1/config` | Current config (secrets redacted) |
| WS | `/v1/stream` | Token-by-token streaming oversight |

### `POST /v1/review`

```json
// Request
{"prompt": "What is 2+2?", "response": "2+2 equals 4."}

// Response
{"approved": true, "coherence": 0.85, "h_logical": 0.12, "h_factual": 0.20,
 "warning": false, "evidence": null}
```

### `WS /v1/stream`

Send `{"prompt": "...", "streaming_oversight": true}`. Receives:

- `{"type": "token", "token": "Einstein", "coherence": 0.92, "index": 0}`
- `{"type": "halt", "token": "...", "reason": "hard_limit (...)"}`
- `{"type": "result", "output": "...", "halted": false}`

## Auth

Set `api_keys` in config. All endpoints except `/v1/health` and
`/v1/metrics/prometheus` require `X-API-Key` header.

```bash
curl -H "X-API-Key: sk-abc123" -X POST http://localhost:8080/v1/review \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "response": "test reply"}'
```

## Rate Limiting

Requires `pip install director-ai[ratelimit]`. Set `rate_limit_rpm`. Returns 429 when exceeded.

## Request ID

Pass `X-Request-ID` for log correlation. Auto-generated UUID if omitted. Echoed in response.

## Usage

```bash
director-ai serve --port 8080 --profile thorough
```

```python
import uvicorn
from director_ai.server import create_app
from director_ai.core.config import DirectorConfig

cfg = DirectorConfig.from_profile("medical")
cfg.api_keys = ["sk-prod-key"]
cfg.rate_limit_rpm = 120
app = create_app(cfg)
uvicorn.run(app, host=cfg.server_host, port=cfg.server_port)
```
