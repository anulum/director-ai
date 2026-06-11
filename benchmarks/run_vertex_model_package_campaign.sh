#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex model package campaign submitter

set -euo pipefail

PROJECT="${PROJECT:-gotm-director-ai}"
REGION="${REGION:-europe-west4}"
BUCKET="${BUCKET:-gs://gotm-director-ai-training}"
REPO="${REPO:-director-ai-training}"
IMAGE_NAME="${IMAGE_NAME:-director-ai-benchmarks}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR="${ACCELERATOR:-NVIDIA_TESLA_T4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-ssd}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-500}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
MODEL_ALIASES="${MODEL_ALIASES:-}"
STAGE_IDS="${STAGE_IDS:-}"
SKIP_BUILD=0
DRY_RUN=0
DISPLAY_SUFFIX=""
CUSTOM_PREFIX=""
CONFIG_FILE=""
CONFIG_OUT=""
CONFIG_IS_TEMP=0

cleanup() {
    if [[ "${CONFIG_IS_TEMP}" -eq 1 && -n "${CONFIG_FILE}" && -f "${CONFIG_FILE}" ]]; then
        rm -f "${CONFIG_FILE}"
    fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-aliases) MODEL_ALIASES="$2"; shift 2 ;;
        --stage-ids) STAGE_IDS="$2"; shift 2 ;;
        --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
        --boot-disk-type) BOOT_DISK_TYPE="$2"; shift 2 ;;
        --boot-disk-size) BOOT_DISK_SIZE_GB="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --config-out) CONFIG_OUT="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --accelerator) ACCELERATOR="$2"; shift 2 ;;
        --accelerator-count) ACCELERATOR_COUNT="$2"; shift 2 ;;
        --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
        --prefix) CUSTOM_PREFIX="$2"; shift 2 ;;
        --suffix) DISPLAY_SUFFIX="-$2"; shift 2 ;;
        -h|--help)
            sed -n '2,52p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

COMMIT_SHA="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
FULL_COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
GIT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
if [[ -z "${GIT_BRANCH}" ]]; then
    GIT_BRANCH="detached"
fi
TIMESTAMP="$(date +%Y%m%dT%H%M)"
RUN_ID="${TIMESTAMP}-${COMMIT_SHA}${DISPLAY_SUFFIX}"
RUN_PREFIX="${CUSTOM_PREFIX:-benchmarks/model-packages/campaigns/${RUN_ID}}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}:${COMMIT_SHA}"

export BOOT_DISK_SIZE_GB MIN_FREE_GB
python - <<'PY'
import os

boot_disk_size_gb = int(os.environ["BOOT_DISK_SIZE_GB"])
min_free_gb = float(os.environ["MIN_FREE_GB"])
if boot_disk_size_gb < 500:
    raise SystemExit("BOOT_DISK_SIZE_GB must be at least 500")
if min_free_gb < 25:
    raise SystemExit("MIN_FREE_GB must be at least 25")
PY

echo "=== configuration ==="
echo "  project             = ${PROJECT}"
echo "  region              = ${REGION}"
echo "  bucket              = ${BUCKET}"
echo "  image               = ${IMAGE_URI}"
echo "  run_prefix          = ${BUCKET}/${RUN_PREFIX}"
echo "  machine_type        = ${MACHINE_TYPE}"
echo "  accelerator         = ${ACCELERATOR} x${ACCELERATOR_COUNT}"
echo "  boot_disk           = ${BOOT_DISK_TYPE} ${BOOT_DISK_SIZE_GB} GiB"
echo "  min_free_gb         = ${MIN_FREE_GB}"
echo "  model_aliases       = ${MODEL_ALIASES:-<all stable>}"
echo "  stage_ids           = ${STAGE_IDS:-<all Vertex stages>}"
echo "  git_commit          = ${FULL_COMMIT_SHA}"
echo "  git_branch          = ${GIT_BRANCH}"

if [[ -n "${CONFIG_OUT}" ]]; then
    CONFIG_FILE="${CONFIG_OUT}"
else
    CONFIG_FILE="$(mktemp)"
    CONFIG_IS_TEMP=1
fi
export MACHINE_TYPE ACCELERATOR ACCELERATOR_COUNT IMAGE_URI
export BOOT_DISK_TYPE BOOT_DISK_SIZE_GB BUCKET RUN_PREFIX MIN_FREE_GB
export FULL_COMMIT_SHA GIT_BRANCH MODEL_ALIASES STAGE_IDS CONFIG_FILE
python - <<'PY'
import json
import os

machine_spec = {"machineType": os.environ["MACHINE_TYPE"]}
accelerator = os.environ.get("ACCELERATOR", "")
if accelerator:
    machine_spec["acceleratorType"] = accelerator
    machine_spec["acceleratorCount"] = int(os.environ["ACCELERATOR_COUNT"])

env = [
    {"name": "DIRECTOR_MODEL_PACKAGE_CAMPAIGN", "value": "1"},
    {"name": "DIRECTOR_REQUIRE_CUDA", "value": "1"},
    {"name": "DIRECTOR_BENCH_BUCKET", "value": os.environ["BUCKET"]},
    {"name": "DIRECTOR_BENCH_PREFIX", "value": os.environ["RUN_PREFIX"]},
    {
        "name": "DIRECTOR_MODEL_PACKAGE_MIN_FREE_GB",
        "value": os.environ["MIN_FREE_GB"],
    },
    {"name": "DIRECTOR_GIT_COMMIT", "value": os.environ["FULL_COMMIT_SHA"]},
    {"name": "DIRECTOR_GIT_BRANCH", "value": os.environ["GIT_BRANCH"]},
    {"name": "DIRECTOR_RUN_ENV", "value": "vertex"},
]
if os.environ.get("MODEL_ALIASES"):
    env.append(
        {
            "name": "DIRECTOR_MODEL_PACKAGE_ALIASES",
            "value": os.environ["MODEL_ALIASES"],
        }
    )
if os.environ.get("STAGE_IDS"):
    env.append(
        {
            "name": "DIRECTOR_MODEL_PACKAGE_STAGE_IDS",
            "value": os.environ["STAGE_IDS"],
        }
    )

config = {
    "workerPoolSpecs": [
        {
            "machineSpec": machine_spec,
            "diskSpec": {
                "bootDiskType": os.environ["BOOT_DISK_TYPE"],
                "bootDiskSizeGb": int(os.environ["BOOT_DISK_SIZE_GB"]),
            },
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

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "=== dry run ==="
    echo "  config              = ${CONFIG_FILE}"
    python -m json.tool "${CONFIG_FILE}"
    exit 0
fi

gcloud config set project "${PROJECT}" >/dev/null

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo ""
    echo "=== phase 1: Cloud Build (remote image build) ==="
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

JOB_NAME="modelpkg-campaign-${RUN_ID}"
echo "  display-name        = ${JOB_NAME}"
echo "  campaign_summary    = ${BUCKET}/${RUN_PREFIX}/campaign_summary.json"

gcloud ai custom-jobs create \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --config="${CONFIG_FILE}"

echo ""
echo "=== done ==="
echo "Monitor with:"
echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT} --limit=5"
echo ""
echo "After completion, fetch results:"
echo "  gcloud storage cp -r ${BUCKET}/${RUN_PREFIX}/ ./benchmarks/results/model-packages/${RUN_ID}/"
