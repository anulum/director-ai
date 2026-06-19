#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — JarvisLabs managed-run entrypoint for RAGTruth token training

set -euo pipefail

cd "$(dirname "$0")"

export TRAINER="${TRAINER:-$PWD/train_ragtruth_token.py}"
export EVAL_SCRIPT="${EVAL_SCRIPT:-$PWD/eval_ragtruth_token.py}"
export SELECTOR="${SELECTOR:-$PWD/select_ragtruth_checkpoint.py}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/ragtruth-token-modernbert-l4}"
export TAR_PATH="${TAR_PATH:-/home/ragtruth-token-modernbert-l4.tar.gz}"
export DISK_CHECK_PATH="${DISK_CHECK_PATH:-/home}"
export MIN_FREE_GB="${MIN_FREE_GB:-80}"

bash "$PWD/jarvislabs_ragtruth_token_run.sh"
