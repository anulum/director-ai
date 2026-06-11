#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — local model package campaign runner

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-./benchmarks/results/model-packages/local}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
MODEL_ALIASES="${MODEL_ALIASES:-}"
STAGE_IDS="${STAGE_IDS:-}"
UPLOAD_URI="${UPLOAD_URI:-}"
PREFIX="${PREFIX:-model-packages/local}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --model-aliases) MODEL_ALIASES="$2"; shift 2 ;;
        --stage-ids) STAGE_IDS="$2"; shift 2 ;;
        --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
        --upload-uri) UPLOAD_URI="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,44p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

export DIRECTOR_GIT_COMMIT="${DIRECTOR_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export DIRECTOR_GIT_BRANCH="${DIRECTOR_GIT_BRANCH:-$(git branch --show-current 2>/dev/null || echo detached)}"
if [[ -z "${DIRECTOR_GIT_BRANCH}" ]]; then
    export DIRECTOR_GIT_BRANCH="detached"
fi
export DIRECTOR_RUN_ENV="${DIRECTOR_RUN_ENV:-local}"

COMMAND=(
    python
    -m
    benchmarks.model_package_campaign
    --output-root "${OUTPUT_ROOT}"
    --min-free-gb "${MIN_FREE_GB}"
)
if [[ -n "${UPLOAD_URI}" ]]; then
    COMMAND+=(--upload-uri "${UPLOAD_URI}" --prefix "${PREFIX}")
else
    COMMAND+=(--no-upload)
fi
if [[ -n "${MODEL_ALIASES}" ]]; then
    COMMAND+=(--model-aliases "${MODEL_ALIASES}")
fi
if [[ -n "${STAGE_IDS}" ]]; then
    COMMAND+=(--stage-ids "${STAGE_IDS}")
fi

echo "=== local model package campaign ==="
echo "  output_root = ${OUTPUT_ROOT}"
echo "  min_free_gb = ${MIN_FREE_GB}"
echo "  aliases     = ${MODEL_ALIASES:-<all stable>}"
echo "  stages      = ${STAGE_IDS:-<all managed package stages>}"
echo "  upload_uri  = ${UPLOAD_URI:-<disabled>}"
echo "  prefix      = ${PREFIX}"
echo "  git_commit  = ${DIRECTOR_GIT_COMMIT}"
echo "  git_branch  = ${DIRECTOR_GIT_BRANCH}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'command:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}"
