#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth token detector L4/A100 training (run ON a JarvisLabs instance)
#
# Runs the RAGTruth token detector on a rented GPU. Optional hard-negative mode:
# provide HARD_NEGATIVE_BOOTSTRAP_MODEL and EVAL_SCRIPT to score the train split,
# mine grounded false positives, then train the next candidate with those
# train-only hard-negative weights. Leaves the model + token_metrics.json in OUT
# and a tarball for download.
#
# IMPORTANT: run from the staged RAGTruth training bundle uploaded to the
# instance. The trainer is self-contained (no director_ai imports; base model and
# wandb/RAGTruth-processed dataset are pulled from the Hub). See the internal
# runbook: docs/internal/jarvislabs_ragtruth_token_runbook.md.
#
# This script carries no secrets. Tuned defaults target an A100 PCIE 40GB;
# override any setting via the environment.

set -euo pipefail

# Where the uploaded trainer lives on the instance (rsync target from the runbook).
# Jarvis CLI container runs use /home, while manual SSH runs often use /root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="${TRAINER:-/root/train_ragtruth_token.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/root/eval_ragtruth_token.py}"
SELECTOR="${SELECTOR:-/root/select_ragtruth_checkpoint.py}"
RAGTRUTH_REQUIREMENTS="${RAGTRUTH_REQUIREMENTS:-$SCRIPT_DIR/jarvislabs_ragtruth_token_requirements.txt}"
OUT="${OUTPUT_DIR:-/root/ragtruth-token-modernbert-l4}"
TAR_PATH="${TAR_PATH:-/root/ragtruth-token-modernbert-l4.tar.gz}"
DISK_CHECK_PATH="${DISK_CHECK_PATH:-$(dirname "$OUT")}"
HARD_NEGATIVE_DIR="${HARD_NEGATIVE_DIR:-/root/ragtruth-hard-negatives}"
HARD_NEGATIVE_BOOTSTRAP_MODEL="${HARD_NEGATIVE_BOOTSTRAP_MODEL:-}"
HARD_NEGATIVE_OUTPUT="${HARD_NEGATIVE_OUTPUT:-$HARD_NEGATIVE_DIR/train_hard_negatives.jsonl}"
HARD_NEGATIVE_EVAL_RESULT="${HARD_NEGATIVE_EVAL_RESULT:-$HARD_NEGATIVE_DIR/train_eval_result.json}"
HARD_NEGATIVE_CACHE="${HARD_NEGATIVE_CACHE:-$HARD_NEGATIVE_DIR/train_eval_probs.json}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-$HARD_NEGATIVE_DIR/checkpoint_selection.json}"

echo "=== RAGTruth token detector — ml=4096 escalation ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?!"

MIN_FREE_GB="${MIN_FREE_GB:-60}"
mkdir -p "$DISK_CHECK_PATH"
FREE_GB="$(df -BG "$DISK_CHECK_PATH" | awk 'NR==2{gsub("G",""); print $4}')"
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "ERROR: only ${FREE_GB}GB free under ${DISK_CHECK_PATH}; need at least ${MIN_FREE_GB}GB. Provision a larger JarvisLabs disk." >&2
  exit 1
fi
echo ">>> disk ok: ${FREE_GB}GB free under ${DISK_CHECK_PATH}"

if [ ! -f "$TRAINER" ]; then
  echo "ERROR: trainer not found at $TRAINER — rsync training/train_ragtruth_token.py first" >&2
  exit 1
fi
if [ -n "$HARD_NEGATIVE_BOOTSTRAP_MODEL" ] && [ ! -f "$EVAL_SCRIPT" ]; then
  echo "ERROR: hard-negative mode needs eval script at $EVAL_SCRIPT" >&2
  exit 1
fi
if [ ! -f "$RAGTRUTH_REQUIREMENTS" ]; then
  echo "ERROR: hash-pinned requirements not found at $RAGTRUTH_REQUIREMENTS — upload training/jarvislabs_ragtruth_token_requirements.txt with the wrapper" >&2
  exit 1
fi

# Dependencies (CUDA torch is preinstalled in the JarvisLabs PyTorch image).
#
# CRITICAL: install ONLY these third-party packages. NEVER `pip install director-ai`
# — the PyPI release is stale/broken. The trainer is self-contained (no director_ai
# import; base model + dataset come from the Hub), so the package is not needed at
# all. None of the packages below depend on director-ai, so pip cannot pull it.
pip install --quiet --upgrade --require-hashes --no-deps --requirement "$RAGTRUTH_REQUIREMENTS"

# Contamination guard: fail loudly if a director-ai (e.g. the stale PyPI one) is on
# the path. The trainer must run purely against the uploaded local file.
python - <<'GUARD'
import importlib.util, sys
if importlib.util.find_spec("director_ai") is not None:
    sys.exit("FATAL: director_ai is importable on this instance — refuse to train "
             "against a possibly-stale package. Uninstall it; the trainer needs none.")
import transformers, datasets, sklearn, torch, numpy
print(f">>> deps ok: transformers {transformers.__version__}, torch {torch.__version__}, "
      f"datasets {datasets.__version__}; director_ai absent (correct)")
GUARD

# A100 40GB has headroom at ml=4096 for ModernBERT-base, so grad checkpointing is
# off (faster) and the batch is larger than the 6 GB local run. On OOM, fall back
# to BATCH_SIZE=4 GRAD_ACCUM=4 GRAD_CHECKPOINT=1.
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export EPOCHS="${EPOCHS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-0}"
export LR="${LR:-3e-5}"
export POS_WEIGHT_SCALE="${POS_WEIGHT_SCALE:-1.0}"
export FOCAL_GAMMA="${FOCAL_GAMMA:-0.0}"
export HARD_NEGATIVE_MAX_WEIGHT="${HARD_NEGATIVE_MAX_WEIGHT:-5.0}"
export HARD_NEGATIVE_FP_PENALTY="${HARD_NEGATIVE_FP_PENALTY:-0.0}"
export OUTPUT_DIR="$OUT"
export PYTHONUNBUFFERED=1

if [ -n "$HARD_NEGATIVE_BOOTSTRAP_MODEL" ]; then
  if [ ! -d "$HARD_NEGATIVE_BOOTSTRAP_MODEL" ]; then
    echo "ERROR: HARD_NEGATIVE_BOOTSTRAP_MODEL is not a directory: $HARD_NEGATIVE_BOOTSTRAP_MODEL" >&2
    exit 1
  fi
  mkdir -p "$HARD_NEGATIVE_DIR"
  echo ">>> mining TRAIN-split hard negatives from $HARD_NEGATIVE_BOOTSTRAP_MODEL"
  DATASET_SPLIT=train \
  MODEL_DIR="$HARD_NEGATIVE_BOOTSTRAP_MODEL" \
  CACHE="$HARD_NEGATIVE_CACHE" \
  RESULT="$HARD_NEGATIVE_EVAL_RESULT" \
  HARD_NEGATIVES="$HARD_NEGATIVE_OUTPUT" \
  TOP_FALSE_POSITIVES="${TOP_FALSE_POSITIVES:-50}" \
  BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}" \
  MAX_LENGTH="$MAX_LENGTH" \
  python -u "$EVAL_SCRIPT" --recompute

  if ! [ -s "$HARD_NEGATIVE_OUTPUT" ]; then
    echo "ERROR: hard-negative mining produced no rows at $HARD_NEGATIVE_OUTPUT" >&2
    exit 1
  fi
  export TRAIN_HARD_NEGATIVES="$HARD_NEGATIVE_OUTPUT"
  echo ">>> TRAIN_HARD_NEGATIVES=$TRAIN_HARD_NEGATIVES"
fi

echo ">>> MAX_LENGTH=$MAX_LENGTH EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM GRAD_CHECKPOINT=$GRAD_CHECKPOINT"
echo ">>> POS_WEIGHT_SCALE=$POS_WEIGHT_SCALE FOCAL_GAMMA=$FOCAL_GAMMA HARD_NEGATIVE_MAX_WEIGHT=$HARD_NEGATIVE_MAX_WEIGHT HARD_NEGATIVE_FP_PENALTY=$HARD_NEGATIVE_FP_PENALTY"
python -u "$TRAINER"

echo ">>> final metrics:"
cat "$OUT/token_metrics.json" || true
if [ -f "$EVAL_SCRIPT" ]; then
  echo ">>> example-level eval on final model"
  DATASET_SPLIT=test \
  MODEL_DIR="$OUT" \
  CACHE="$OUT/token_eval_probs.json" \
  RESULT="$OUT/example_eval_result.json" \
  HARD_NEGATIVES="$OUT/test_hard_negatives_DO_NOT_TRAIN.jsonl" \
  BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}" \
  MAX_LENGTH="$MAX_LENGTH" \
  python -u "$EVAL_SCRIPT" --recompute

  if [ "${EVAL_SAVED_CHECKPOINTS:-1}" = "1" ]; then
    for checkpoint_dir in "$OUT"/checkpoint-*; do
      [ -d "$checkpoint_dir" ] || continue
      checkpoint_name="$(basename "$checkpoint_dir")"
      echo ">>> example-level eval on $checkpoint_name"
      DATASET_SPLIT=test \
      MODEL_DIR="$checkpoint_dir" \
      CACHE="$checkpoint_dir/token_eval_probs.json" \
      RESULT="$checkpoint_dir/example_eval_result.json" \
      HARD_NEGATIVES=0 \
      BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}" \
      MAX_LENGTH="$MAX_LENGTH" \
      python -u "$EVAL_SCRIPT" --recompute
    done
  fi
fi
if [ -f "$SELECTOR" ] && [ -f "$OUT/example_eval_result.json" ]; then
  selection_inputs=("$OUT/example_eval_result.json")
  for checkpoint_result in "$OUT"/checkpoint-*/example_eval_result.json; do
    [ -f "$checkpoint_result" ] || continue
    selection_inputs+=("$checkpoint_result")
  done
  python -u "$SELECTOR" "${selection_inputs[@]}" --output "$SELECTION_OUTPUT"
  cp "$SELECTION_OUTPUT" "$OUT/checkpoint_selection.json"
fi
tar -czf "$TAR_PATH" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo ">>> DONE. Download $TAR_PATH"
