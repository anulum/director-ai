#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
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
DIRECTOR_OUTPUT_ALREADY_UPLOADED=0

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
if [[ "${DIRECTOR_REQUIRE_CUDA:-0}" == "1" ]]; then
    python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("DIRECTOR_REQUIRE_CUDA=1 but torch.cuda.is_available() is false")
probe = torch.ones(1, device="cuda")
_ = float(probe.cpu()[0])
print(f"cuda smoke ok: {torch.cuda.get_device_name(0)}", file=sys.stderr)
PY
fi
echo ""

if [[ "${DIRECTOR_MODEL_BENCHMARK:-}" == "1" ]]; then
    echo "=== managed model-choice benchmark ==="
    python -m training.vertex_model_benchmark --output-dir "${OUTPUT_DIR}"
    ORCHESTRATOR_EXIT=$?
elif [[ "${DIRECTOR_MODEL_PACKAGE_CAMPAIGN:-}" == "1" ]]; then
    echo "=== per-model benchmark package campaign ==="
    PACKAGE_ARGS=(
        --output-root "${OUTPUT_DIR}"
        --min-free-gb "${DIRECTOR_MODEL_PACKAGE_MIN_FREE_GB:-25}"
    )
    if [[ "${DIRECTOR_MODEL_PACKAGE_NO_UPLOAD:-0}" == "1" ]]; then
        PACKAGE_ARGS+=(--no-upload)
    else
        PACKAGE_ARGS+=(
            --bucket "${DIRECTOR_BENCH_BUCKET:?DIRECTOR_BENCH_BUCKET is required}"
            --prefix "${DIRECTOR_BENCH_PREFIX:?DIRECTOR_BENCH_PREFIX is required}"
        )
    fi
    if [[ -n "${DIRECTOR_MODEL_PACKAGE_ALIASES:-}" ]]; then
        PACKAGE_ARGS+=(--model-aliases "${DIRECTOR_MODEL_PACKAGE_ALIASES}")
    fi
    if [[ -n "${DIRECTOR_MODEL_PACKAGE_STAGE_IDS:-}" ]]; then
        PACKAGE_ARGS+=(--stage-ids "${DIRECTOR_MODEL_PACKAGE_STAGE_IDS}")
    fi
    python -m benchmarks.model_package_campaign "${PACKAGE_ARGS[@]}"
    ORCHESTRATOR_EXIT=$?
    DIRECTOR_OUTPUT_ALREADY_UPLOADED=1
else
    echo "=== orchestrator run ==="
    python -m benchmarks.orchestrator \
        --runner vertex \
        --output-dir "${OUTPUT_DIR}" \
        --report-name run_report.json \
        --verbose \
        "${EXTRA_ARGS[@]}"

    ORCHESTRATOR_EXIT=$?
fi

if [[ "${DIRECTOR_OUTPUT_ALREADY_UPLOADED:-0}" != "1" && -n "${DIRECTOR_BENCH_BUCKET:-}" && -n "${DIRECTOR_BENCH_PREFIX:-}" ]]; then
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
