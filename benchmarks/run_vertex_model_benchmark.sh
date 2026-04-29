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

COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
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

WORKER_POOL_SPEC="machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${IMAGE_URI}"
if [[ -n "${ACCELERATOR}" ]]; then
    WORKER_POOL_SPEC="${WORKER_POOL_SPEC},accelerator-type=${ACCELERATOR},accelerator-count=${ACCELERATOR_COUNT}"
fi
WORKER_POOL_SPEC+=",env=DIRECTOR_MODEL_BENCHMARK=1"
WORKER_POOL_SPEC+=",env=DIRECTOR_BENCH_BUCKET=${BUCKET}"
WORKER_POOL_SPEC+=",env=DIRECTOR_BENCH_PREFIX=${RUN_PREFIX}"
WORKER_POOL_SPEC+=",env=DIRECTOR_MODEL_BENCHMARK_MODELS=${MODEL_SPECS}"
WORKER_POOL_SPEC+=",env=DIRECTOR_MODEL_BENCHMARK_GENERAL_URI=${GENERAL_URI}"
if [[ -n "${EVAL_URI}" ]]; then
    WORKER_POOL_SPEC+=",env=DIRECTOR_MODEL_BENCHMARK_EVAL_URI=${EVAL_URI}"
fi
WORKER_POOL_SPEC+=",env=DIRECTOR_MODEL_BENCHMARK_BATCH_SIZE=${BATCH_SIZE}"

JOB_NAME="model-bench-${RUN_ID}"
gcloud ai custom-jobs create \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --worker-pool-spec="${WORKER_POOL_SPEC}"

echo "Submitted ${JOB_NAME}"
echo "Monitor:"
echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT} --limit=5"
