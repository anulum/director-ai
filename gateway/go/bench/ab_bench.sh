#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — A/B benchmark: gateway with and without scoring
#
# Runs two k6 scenarios sequentially against a local Go gateway:
#
#   A) gateway only — ``DIRECTOR_SCORING_ADDR`` unset
#   B) gateway + Python gRPC scoring sidecar
#
# Results go to ``gateway/go/bench/out/`` as JSON plus a diff table.
#
# Prereqs:
#   - `go` and `python` on PATH
#   - `pip install director-ai` or an editable install so
#     `python -m director_ai.grpc_scoring` is importable
#   - `k6` installed (https://k6.io/docs/get-started/installation)
#   - `jq` (optional) for pretty-printing the diff
#
# Usage:
#   bash gateway/go/bench/ab_bench.sh [VUS] [DURATION]
# Defaults: VUS=50, DURATION=30s.

set -euo pipefail

VUS=${1:-50}
DURATION=${2:-30s}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
mkdir -p "${OUT_DIR}"

UPSTREAM_PORT=${UPSTREAM_PORT:-9901}
GATEWAY_PORT=${GATEWAY_PORT:-8081}
GRPC_PORT=${GRPC_PORT:-50052}
API_KEY=${API_KEY:-sk-ab-bench}

cleanup() {
  local signal=$?
  echo "cleaning up (exit=$signal)"
  for pid in "${UPSTREAM_PID:-}" "${GATEWAY_PID:-}" "${GRPC_PID:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "→ mock upstream on :${UPSTREAM_PORT}"
python - <<PY &
import http.server, json, socketserver
PORT = ${UPSTREAM_PORT}
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        ln = int(self.headers.get("Content-Length") or 0)
        _ = self.rfile.read(ln)
        body = json.dumps({
            "id": "chatcmpl-bench", "object": "chat.completion", "model": "mock",
            "choices": [{"index": 0, "message": {"role": "assistant",
                "content": "Paris is the capital of France."},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **kw): pass
with socketserver.ThreadingTCPServer(("", PORT), H) as s:
    s.serve_forever()
PY
UPSTREAM_PID=$!
sleep 1

echo "→ building gateway binary"
(cd "${REPO_ROOT}/gateway/go" && go build -o "${OUT_DIR}/director-gateway" ./cmd/director-gateway)

echo
echo "══ Scenario A: gateway without scoring ══"
DIRECTOR_LISTEN_ADDR=":${GATEWAY_PORT}" \
DIRECTOR_UPSTREAM_URL="http://127.0.0.1:${UPSTREAM_PORT}" \
DIRECTOR_ALLOW_HTTP_UPSTREAM=1 \
DIRECTOR_API_KEYS="${API_KEY}" \
DIRECTOR_AUDIT_SALT="bench-salt" \
DIRECTOR_RATE_LIMIT_RPM=100000 \
DIRECTOR_RATE_LIMIT_BURST=1000 \
"${OUT_DIR}/director-gateway" > "${OUT_DIR}/gateway_a.log" 2>&1 &
GATEWAY_PID=$!
sleep 1

DIRECTOR_URL="http://127.0.0.1:${GATEWAY_PORT}" \
DIRECTOR_KEY="${API_KEY}" \
DIRECTOR_VUS="${VUS}" \
DIRECTOR_DURATION="${DURATION}" \
k6 run --summary-export="${OUT_DIR}/summary_a.json" \
  "${SCRIPT_DIR}/passthrough.js"

kill "${GATEWAY_PID}"; wait "${GATEWAY_PID}" 2>/dev/null || true

echo
echo "══ Scenario B: gateway + Python gRPC scoring ══"

echo "→ starting Python gRPC scoring on :${GRPC_PORT}"
(cd "${REPO_ROOT}" && \
  PYTHONUNBUFFERED=1 python -u -m director_ai.grpc_scoring \
    --listen "[::]:${GRPC_PORT}") \
  > "${OUT_DIR}/grpc.log" 2>&1 &
GRPC_PID=$!
sleep 3

DIRECTOR_LISTEN_ADDR=":${GATEWAY_PORT}" \
DIRECTOR_UPSTREAM_URL="http://127.0.0.1:${UPSTREAM_PORT}" \
DIRECTOR_ALLOW_HTTP_UPSTREAM=1 \
DIRECTOR_API_KEYS="${API_KEY}" \
DIRECTOR_AUDIT_SALT="bench-salt" \
DIRECTOR_RATE_LIMIT_RPM=100000 \
DIRECTOR_RATE_LIMIT_BURST=1000 \
DIRECTOR_SCORING_ADDR="127.0.0.1:${GRPC_PORT}" \
DIRECTOR_SCORING_TIMEOUT_MS=500 \
"${OUT_DIR}/director-gateway" > "${OUT_DIR}/gateway_b.log" 2>&1 &
GATEWAY_PID=$!
sleep 1

DIRECTOR_URL="http://127.0.0.1:${GATEWAY_PORT}" \
DIRECTOR_KEY="${API_KEY}" \
DIRECTOR_VUS="${VUS}" \
DIRECTOR_DURATION="${DURATION}" \
k6 run --summary-export="${OUT_DIR}/summary_b.json" \
  "${SCRIPT_DIR}/passthrough.js"

kill "${GATEWAY_PID}"; wait "${GATEWAY_PID}" 2>/dev/null || true
kill "${GRPC_PID}"; wait "${GRPC_PID}" 2>/dev/null || true

echo
echo "══ Summary ══"
for f in summary_a summary_b; do
  echo "--- ${f} ---"
  if command -v jq >/dev/null 2>&1; then
    jq '{
      reqs: .metrics.http_reqs.count // .metrics.http_reqs.values.count,
      p95:  .metrics.http_req_duration.values["p(95)"],
      p99:  .metrics.http_req_duration.values["p(99)"],
      failed: .metrics.http_req_failed.values.fails
    }' "${OUT_DIR}/${f}.json" || cat "${OUT_DIR}/${f}.json"
  else
    cat "${OUT_DIR}/${f}.json"
  fi
  echo
done

echo "raw k6 output in ${OUT_DIR}/summary_{a,b}.json"
