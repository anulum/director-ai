#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — airgap full-stack install example

set -euo pipefail

WHEELHOUSE="${WHEELHOUSE:-wheelhouse}"
RUST_WHEELHOUSE="${RUST_WHEELHOUSE:-wheelhouse/rust}"
MODEL_ROOT="${MODEL_ROOT:-models}"
ONNX_DIR="${ONNX_DIR:-models/factcg-onnx}"
VENV_DIR="${VENV_DIR:-.venv-airgap}"

python -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"

UV_OFFLINE=1 uv sync --locked --offline --active \
  --extra server \
  --extra vector \
  --extra nli \
  --extra onnx \
  --extra ui

if [ -d "${RUST_WHEELHOUSE}" ]; then
  uv pip install --offline --find-links "${RUST_WHEELHOUSE}" backfire-kernel==0.1.0
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIRECTOR_SCORER_BACKEND=onnx
export DIRECTOR_ONNX_PATH="${ONNX_DIR}"
export DIRECTOR_MODEL_ROOT="${MODEL_ROOT}"

director-ai doctor
python -c "from pathlib import Path; required=['model.onnx','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']; missing=[p for p in required if not (Path('${ONNX_DIR}') / p).exists()]; raise SystemExit('missing ONNX files: '+','.join(missing) if missing else 0)"
