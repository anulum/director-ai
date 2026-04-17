# Director-AI Gateway (Go)

Phase 2 of the multi-language refactor. The Go gateway is the
front-door HTTP endpoint clients talk to. In this phase it is a
pure **passthrough proxy** to an upstream OpenAI-compatible API —
the point is to own the Go hop (TLS termination, auth, rate limit,
audit) without yet wiring the Python scoring service. Phase 3 adds
the scoring integration.

## Why a Go front door

- Goroutine-based concurrency gives thousands of simultaneous
  SSE streams per process with flat memory. Python + Uvicorn
  reaches the same throughput only by scaling replicas.
- `net/http` has a mature streaming `Flusher` path that matches the
  SSE behaviour of OpenAI-compatible APIs without reinventing
  anything.
- The gateway is stateless — config on boot, buckets in memory,
  audit to disk. A replica can restart without losing correctness.

## Layout

```
gateway/go/
├── cmd/director-gateway/main.go   # binary entrypoint
├── go.mod / go.sum
├── internal/
│   ├── config/      # env-driven config loader
│   ├── auth/        # API-key middleware + fingerprint
│   ├── ratelimit/   # per-key token bucket
│   ├── proxy/       # upstream passthrough
│   ├── audit/       # JSONL audit sink
│   └── server/      # middleware chain + handlers
├── proto/director/v1/ # generated stubs (Phase 1)
└── bench/passthrough.js # k6 load test
```

## Configuration

All via environment variables so the binary can be dropped into any
process supervisor without a config file.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DIRECTOR_LISTEN_ADDR` | `:8080` | HTTP listen address |
| `DIRECTOR_UPSTREAM_URL` | `https://api.openai.com` | target LLM API |
| `DIRECTOR_UPSTREAM_TIMEOUT_SECONDS` | `30` | per-request timeout |
| `DIRECTOR_API_KEYS` | *(empty)* | comma-separated valid keys |
| `DIRECTOR_RATE_LIMIT_RPM` | `600` | per-key requests per minute |
| `DIRECTOR_RATE_LIMIT_BURST` | `60` | max tokens in bucket |
| `DIRECTOR_AUDIT_SALT` | — | salt for fingerprint HMAC |
| `DIRECTOR_AUDIT_SALT_FILE` | — | read salt from file |
| `DIRECTOR_AUDIT_LOG` | *(stdout)* | path to JSONL audit sink |
| `DIRECTOR_ALLOW_HTTP_UPSTREAM` | `0` | accept plain-HTTP upstream |

If neither `DIRECTOR_AUDIT_SALT` nor `DIRECTOR_AUDIT_SALT_FILE` is
set, the legacy constant `director-ai-audit-v1` is used so
fingerprints stay comparable with older deployments. Production
deployments should override.

Empty `DIRECTOR_API_KEYS` disables authentication — useful in
integration tests, never in production; the binary prints a warning
at startup.

## Endpoints

| Path | Behaviour |
| --- | --- |
| `GET /health`, `/healthz`, `/ready` | returns `{"status":"ok"}` |
| `/v1/*` | passthrough to `DIRECTOR_UPSTREAM_URL` |

## Request pipeline

```
client ──► requestID ──► auth ──► audit ──► rate ──► proxy ──► upstream
                                     │
                                     └─► JSONL audit sink
```

- `requestID` echoes `X-Request-ID` or mints one.
- `auth` validates `Authorization: Bearer` / `X-API-Key` and
  stamps the audit fingerprint onto the request context.
- `audit` captures status + bytes + latency + fingerprint.
- `rate` token-bucket per fingerprint (or remote host in no-auth
  mode). 429 + `Retry-After` on overflow.
- `proxy` forwards path, query, headers (hop-by-hop stripped) and
  body. Streams SSE chunks with `Flusher`.

## Running

```bash
# From the repo root
cd gateway/go
go build -o director-gateway ./cmd/director-gateway

DIRECTOR_API_KEYS="sk-test-1,sk-test-2" \
DIRECTOR_AUDIT_SALT="deployment-local" \
./director-gateway
```

## Testing

```bash
cd gateway/go
go test ./...
```

Coverage (Phase 2):

- `config` — 8 cases (defaults, env overrides, salt file, invalid
  timeout, invalid RPM, HTTPS enforcement, key splitting)
- `auth` — 13 cases (fingerprint stability, salt sensitivity, 16-hex
  length, exempt paths, no-auth mode, missing/wrong/valid key,
  context propagation, key rotation, Bearer precedence, extract
  edge cases)
- `ratelimit` — 10 cases (unlimited mode, burst, refill, cap, per-key
  isolation, 429 body + Retry-After, X-Forwarded-For, port strip,
  concurrency, reset)
- `proxy` — 7 cases (scheme validation, body/status passthrough,
  path+query, SSE chunk streaming, 502 on unreachable, auth
  forwarding on/off, hop-by-hop strip)
- `audit` — 7 cases (JSONL round-trip, auto-timestamp, concurrency,
  file perms 0o600, append-across-reopen, unwritable path, nil
  writer fallback)
- `server` — 5 integration cases (health open, 401 without key,
  end-to-end with echo + audit record, 429 enforcement, X-Request-ID
  echo)

## Benchmarking

`bench/passthrough.js` is a k6 scenario that hammers the gateway's
`/v1/chat/completions` endpoint against a fake upstream. Install k6
from <https://k6.io> or `apt install k6`, then run:

```bash
# spin up a mock upstream
go run ./cmd/director-gateway &  # with DIRECTOR_UPSTREAM_URL=http://localhost:9001

DIRECTOR_URL=http://localhost:8080 \
DIRECTOR_KEY=sk-test-1 \
DIRECTOR_VUS=100 \
DIRECTOR_DURATION=30s \
k6 run gateway/go/bench/passthrough.js
```

Thresholds in the script flag `http_req_duration` p95 > 500 ms and
failure rate > 1%. Numbers should **not** be quoted as fact until
they come from a run on representative hardware with a real
upstream; Gemini-style "8-12× speedup" claims are explicitly out of
scope for this phase.

## Not in this phase

- Scoring integration (Phase 3 — Python exposes a
  `CoherenceScoring` gRPC server; Go calls it per stream).
- Distributed rate limiting across replicas (Redis backend).
- Circuit breaker on upstream errors.
- Structured metrics (Prometheus or OpenTelemetry).
- TLS termination on the gateway itself (still recommend a
  reverse proxy / load balancer in front).
