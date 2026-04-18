#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Director-AI — Vertex AI benchmark container entrypoint
#
# Runs the orchestrator inside the Vertex AI custom-job worker,
# collects the RunReport + any per-case artefacts, and uploads the
# whole output directory to GCS under the job-specific prefix.
#
# Environment variables (set by the submit script):
#   DIRECTOR_BENCH_BUCKET     gs://bucket name (no trailing slash)
#   DIRECTOR_BENCH_PREFIX     GCS prefix for this run (e.g. benchmarks/20260418-1323-abc123)
#   DIRECTOR_BENCH_ONLY       optional space-separated list of case names
#   DIRECTOR_BENCH_BASELINE   optional GCS path to baseline run_report.json
#   DIRECTOR_BENCH_STRICT     optional; "1" exits non-zero on failures
set -euo pipefail

OUTPUT_DIR=/workspace/output
mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${DIRECTOR_BENCH_ONLY:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=(--only ${DIRECTOR_BENCH_ONLY})
fi
if [[ -n "${DIRECTOR_BENCH_STRICT:-}" ]]; then
    EXTRA_ARGS+=(--strict)
fi

BASELINE_LOCAL=""
if [[ -n "${DIRECTOR_BENCH_BASELINE:-}" ]]; then
    BASELINE_LOCAL="${OUTPUT_DIR}/baseline.json"
    echo "Downloading baseline from ${DIRECTOR_BENCH_BASELINE}..."
    python - <<EOF
from google.cloud import storage
import sys
path = "${DIRECTOR_BENCH_BASELINE}".removeprefix("gs://")
bucket_name, _, blob_name = path.partition("/")
client = storage.Client()
client.bucket(bucket_name).blob(blob_name).download_to_filename(
    "${BASELINE_LOCAL}",
)
print(f"downloaded {blob_name}", file=sys.stderr)
EOF
    EXTRA_ARGS+=(--baseline "${BASELINE_LOCAL}")
fi

echo "=== environment ==="
python -c 'import sys; print("python", sys.version)'
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ""

echo "=== orchestrator run ==="
python -m benchmarks.orchestrator \
    --runner vertex \
    --output-dir "${OUTPUT_DIR}" \
    --report-name run_report.json \
    --verbose \
    "${EXTRA_ARGS[@]}"

ORCHESTRATOR_EXIT=$?

if [[ -n "${DIRECTOR_BENCH_BUCKET:-}" && -n "${DIRECTOR_BENCH_PREFIX:-}" ]]; then
    DEST="${DIRECTOR_BENCH_BUCKET}/${DIRECTOR_BENCH_PREFIX}"
    echo ""
    echo "=== uploading ${OUTPUT_DIR}/ → ${DEST} ==="
    python - <<EOF
import os
from pathlib import Path
from google.cloud import storage
bucket_url = "${DIRECTOR_BENCH_BUCKET}".removeprefix("gs://")
bucket = storage.Client().bucket(bucket_url)
prefix = "${DIRECTOR_BENCH_PREFIX}".strip("/")
root = Path("${OUTPUT_DIR}")
count = 0
for path in root.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(root).as_posix()
    blob = bucket.blob(f"{prefix}/{rel}")
    blob.upload_from_filename(str(path))
    count += 1
    print(f"uploaded gs://${DIRECTOR_BENCH_BUCKET#gs://}/{prefix}/{rel}")
print(f"total: {count} file(s)")
EOF
fi

exit "${ORCHESTRATOR_EXIT}"
