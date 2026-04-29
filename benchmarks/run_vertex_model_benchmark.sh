#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Director-AI — Vertex AI managed model-choice benchmark runner
#
# Builds the benchmark image and submits a Vertex CustomJob that
# downloads trained model artefacts from GCS, runs the anti-regression
# benchmark in-cloud, and uploads model_benchmark_report.json.
set -euo pipefail

PROJECT="${PROJECT:-gotm-director-ai}"
REGION="${REGION:-europe-west4}"
BUCKET="${BUCKET:-gs://gotm-director-ai-training}"
REPO="${REPO:-director-ai-training}"
IMAGE_NAME="${IMAGE_NAME:-director-ai-benchmarks}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR="${ACCELERATOR:-NVIDIA_TESLA_T4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
MODEL_SPECS="${MODEL_SPECS:-}"
GENERAL_URI="${GENERAL_URI:-}"
EVAL_URI="${EVAL_URI:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SKIP_BUILD=0
DISPLAY_SUFFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-specs) MODEL_SPECS="$2"; shift 2 ;;
        --general-uri) GENERAL_URI="$2"; shift 2 ;;
        --eval-uri) EVAL_URI="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --accelerator) ACCELERATOR="$2"; shift 2 ;;
        --accelerator-count) ACCELERATOR_COUNT="$2"; shift 2 ;;
        --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
        --suffix) DISPLAY_SUFFIX="-$2"; shift 2 ;;
        -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL_SPECS}" || -z "${GENERAL_URI}" ]]; then
    echo "MODEL_SPECS and GENERAL_URI are required" >&2
    exit 2
fi

COMMIT_SHA="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
TIMESTAMP="$(date +%Y%m%dT%H%M)"
RUN_ID="${TIMESTAMP}-${COMMIT_SHA}${DISPLAY_SUFFIX}"
RUN_PREFIX="managed-training/benchmarks/${RUN_ID}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}:${COMMIT_SHA}"

echo "=== configuration ==="
echo "  project      = ${PROJECT}"
echo "  region       = ${REGION}"
echo "  image        = ${IMAGE_URI}"
echo "  result       = ${BUCKET}/${RUN_PREFIX}/model_benchmark_report.json"
echo "  machine      = ${MACHINE_TYPE}"
echo "  accelerator  = ${ACCELERATOR} x${ACCELERATOR_COUNT}"

gcloud config set project "${PROJECT}" >/dev/null

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    gcloud builds submit \
        --project="${PROJECT}" \
        --config=benchmarks/cloudbuild.yaml \
        --substitutions=_IMAGE_TAG="${COMMIT_SHA}",_REGION="${REGION}",_REPO="${REPO}",_IMAGE_NAME="${IMAGE_NAME}" \
        .
fi

JOB_NAME="model-bench-${RUN_ID}"
CONFIG_FILE="$(mktemp)"
export JOB_NAME MACHINE_TYPE ACCELERATOR ACCELERATOR_COUNT IMAGE_URI
export BUCKET RUN_PREFIX MODEL_SPECS GENERAL_URI EVAL_URI BATCH_SIZE CONFIG_FILE
python - <<'PY'
import json
import os

machine_spec = {"machineType": os.environ["MACHINE_TYPE"]}
accelerator = os.environ.get("ACCELERATOR", "")
if accelerator:
    machine_spec["acceleratorType"] = accelerator
    machine_spec["acceleratorCount"] = int(os.environ["ACCELERATOR_COUNT"])

env = [
    {"name": "DIRECTOR_MODEL_BENCHMARK", "value": "1"},
    {"name": "DIRECTOR_BENCH_BUCKET", "value": os.environ["BUCKET"]},
    {"name": "DIRECTOR_BENCH_PREFIX", "value": os.environ["RUN_PREFIX"]},
    {
        "name": "DIRECTOR_MODEL_BENCHMARK_MODELS",
        "value": os.environ["MODEL_SPECS"],
    },
    {
        "name": "DIRECTOR_MODEL_BENCHMARK_GENERAL_URI",
        "value": os.environ["GENERAL_URI"],
    },
    {
        "name": "DIRECTOR_MODEL_BENCHMARK_BATCH_SIZE",
        "value": os.environ["BATCH_SIZE"],
    },
]
if os.environ.get("EVAL_URI"):
    env.append(
        {
            "name": "DIRECTOR_MODEL_BENCHMARK_EVAL_URI",
            "value": os.environ["EVAL_URI"],
        }
    )

config = {
    "workerPoolSpecs": [
        {
            "machineSpec": machine_spec,
            "replicaCount": 1,
            "containerSpec": {
                "imageUri": os.environ["IMAGE_URI"],
                "env": env,
            },
        }
    ]
}
with open(os.environ["CONFIG_FILE"], "w", encoding="utf-8") as fh:
    json.dump(config, fh)
PY

gcloud ai custom-jobs create \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --config="${CONFIG_FILE}"

echo "Submitted ${JOB_NAME}"
echo "Monitor:"
echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT} --limit=5"
