#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Director-AI — End-to-end Vertex AI benchmark runner
#
# Two phases:
#   1. Build the benchmark container remotely with Cloud Build.
#   2. Submit a Vertex AI custom job that runs the orchestrator
#      inside the container, exports the RunReport + regression
#      diff to gs://${BUCKET}/${PREFIX}/.
#
# Usage:
#   benchmarks/run_vertex_benchmarks.sh
#   benchmarks/run_vertex_benchmarks.sh --accelerator NVIDIA_TESLA_T4
#   benchmarks/run_vertex_benchmarks.sh --only rust_parity_safety
#   benchmarks/run_vertex_benchmarks.sh --baseline gs://bucket/path/baseline.json
#   benchmarks/run_vertex_benchmarks.sh --skip-build
#
# Environment defaults (override via env vars before invocation):
#   PROJECT               gotm-director-ai
#   REGION                europe-west4
#   BUCKET                gs://gotm-director-ai-training
#   REPO                  director-ai-training
#   IMAGE_NAME            director-ai-benchmarks
#   MACHINE_TYPE          n1-standard-8
#   ACCELERATOR           NVIDIA_TESLA_T4  (set to "" for CPU-only)
#   ACCELERATOR_COUNT     1
set -euo pipefail

PROJECT="${PROJECT:-gotm-director-ai}"
REGION="${REGION:-europe-west4}"
BUCKET="${BUCKET:-gs://gotm-director-ai-training}"
REPO="${REPO:-director-ai-training}"
IMAGE_NAME="${IMAGE_NAME:-director-ai-benchmarks}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR="${ACCELERATOR:-NVIDIA_TESLA_T4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"

ONLY=""
BASELINE=""
SKIP_BUILD=0
STRICT=0
DISPLAY_SUFFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)
            ONLY="$2"; shift 2 ;;
        --baseline)
            BASELINE="$2"; shift 2 ;;
        --skip-build)
            SKIP_BUILD=1; shift ;;
        --strict)
            STRICT=1; shift ;;
        --accelerator)
            ACCELERATOR="$2"; shift 2 ;;
        --accelerator-count)
            ACCELERATOR_COUNT="$2"; shift 2 ;;
        --machine-type)
            MACHINE_TYPE="$2"; shift 2 ;;
        --suffix)
            DISPLAY_SUFFIX="-$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2 ;;
    esac
done

COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
TIMESTAMP="$(date +%Y%m%dT%H%M)"
RUN_ID="${TIMESTAMP}-${COMMIT_SHA}${DISPLAY_SUFFIX}"
RUN_PREFIX="benchmarks/${RUN_ID}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}:${COMMIT_SHA}"

echo "=== configuration ==="
echo "  project           = ${PROJECT}"
echo "  region            = ${REGION}"
echo "  bucket            = ${BUCKET}"
echo "  image             = ${IMAGE_URI}"
echo "  run_prefix        = gs://${BUCKET#gs://}/${RUN_PREFIX}"
echo "  machine_type      = ${MACHINE_TYPE}"
echo "  accelerator       = ${ACCELERATOR} x${ACCELERATOR_COUNT}"
echo "  only              = ${ONLY:-<default suite>}"
echo "  baseline          = ${BASELINE:-<none>}"
echo "  strict            = ${STRICT}"

gcloud config set project "${PROJECT}" >/dev/null

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo ""
    echo "=== phase 1: Cloud Build (remote image build) ==="
    # No --region — use the default (global) Cloud Build pool so
    # the build is not bottlenecked by the regional quota that
    # is typically smaller than the global one.
    gcloud builds submit \
        --project="${PROJECT}" \
        --config=benchmarks/cloudbuild.yaml \
        --substitutions=_IMAGE_TAG="${COMMIT_SHA}",_REGION="${REGION}",_REPO="${REPO}",_IMAGE_NAME="${IMAGE_NAME}" \
        .
else
    echo ""
    echo "=== phase 1: SKIPPED (--skip-build) — reusing ${IMAGE_URI} ==="
fi

echo ""
echo "=== phase 2: Vertex AI custom job submission ==="

WORKER_POOL_SPEC="machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${IMAGE_URI}"
if [[ -n "${ACCELERATOR}" ]]; then
    WORKER_POOL_SPEC="${WORKER_POOL_SPEC},accelerator-type=${ACCELERATOR},accelerator-count=${ACCELERATOR_COUNT}"
fi

# ``gcloud ai custom-jobs create`` passes container env vars
# through repeated ``env=KEY=VALUE`` entries inside the single
# --worker-pool-spec flag. Build the suffix once; embed it in
# the full spec string below.
ENV_KV=",env=DIRECTOR_BENCH_BUCKET=${BUCKET}"
ENV_KV+=",env=DIRECTOR_BENCH_PREFIX=${RUN_PREFIX}"
if [[ -n "${ONLY}" ]]; then
    ENV_KV+=",env=DIRECTOR_BENCH_ONLY=${ONLY}"
fi
if [[ -n "${BASELINE}" ]]; then
    ENV_KV+=",env=DIRECTOR_BENCH_BASELINE=${BASELINE}"
fi
if [[ "${STRICT}" -eq 1 ]]; then
    ENV_KV+=",env=DIRECTOR_BENCH_STRICT=1"
fi
WORKER_POOL_SPEC_WITH_ENV="${WORKER_POOL_SPEC}${ENV_KV}"

JOB_NAME="bench-${RUN_ID}"
echo "  display-name      = ${JOB_NAME}"
echo "  result            = gs://${BUCKET#gs://}/${RUN_PREFIX}/run_report.json"

gcloud ai custom-jobs create \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --worker-pool-spec="${WORKER_POOL_SPEC_WITH_ENV}"

echo ""
echo "=== done ==="
echo "Monitor with:"
echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT} --limit=5"
echo ""
echo "After completion, fetch results:"
echo "  gcloud storage cp -r gs://${BUCKET#gs://}/${RUN_PREFIX}/ ./benchmarks/results/${RUN_ID}/"
