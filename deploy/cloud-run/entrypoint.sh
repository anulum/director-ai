#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — authenticated Cloud Run entrypoint
set -euo pipefail

if [[ "${DIRECTOR_PROFILE:-default}" != "default" ]]; then
    echo "Cloud Run uses environment configuration; set DIRECTOR_PROFILE=default and individual DIRECTOR_* fields." >&2
    exit 1
fi

run_mode="$(python - <<'PY'
from director_ai.core.config import DirectorConfig

config = DirectorConfig.from_env()
if not config.api_keys and not config.api_key_tenant_map:
    raise SystemExit("Cloud Run requires DIRECTOR_API_KEYS or DIRECTOR_API_KEY_TENANT_MAP.")
print("--production" if config.production_mode else "")
PY
)"
mode_args=()
if [[ -n "$run_mode" ]]; then
    mode_args=("$run_mode")
fi

exec director-ai serve \
    --host 0.0.0.0 \
    --port "${DIRECTOR_PORT:-${PORT:-8080}}" \
    --workers "${DIRECTOR_WORKERS:-1}" \
    --profile default \
    "${mode_args[@]}"
