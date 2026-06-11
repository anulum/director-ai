#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Director-AI SaaS entrypoint — starts FastAPI with middleware
set -euo pipefail

PORT="${DIRECTOR_PORT:-8080}"
WORKERS="${DIRECTOR_WORKERS:-2}"
PROFILE="${DIRECTOR_PROFILE:-thorough}"

exec director-ai serve \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS" \
    --profile "$PROFILE"
