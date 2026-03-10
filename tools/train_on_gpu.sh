#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — GPU Training Script (UpCloud L40S)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
#
# Run on UpCloud GPU-8xCPU-64GB-1xL40S (€1.11/h, fi-hel2).
# Total estimated time: ~2.75 hours ≈ €3.05
#
# Prerequisites:
#   - Python 3.11+
#   - HF_TOKEN env var (for gated LLM-AggreFact)
#
# Usage:
#   ssh user@gpu-server 'bash -s' < tools/train_on_gpu.sh
#   # or copy repo and run directly:
#   bash tools/train_on_gpu.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

MODELS_DIR="$REPO_DIR/models"
DATA_DIR="$REPO_DIR/data"

echo "=== Director-AI GPU Training Pipeline ==="
echo "Repo:   $REPO_DIR"
echo "Models: $MODELS_DIR"
echo "Data:   $DATA_DIR"
echo ""

# 1. Install
echo ">>> Installing Director-AI with finetune extras..."
pip install -e ".[finetune,nli]" --quiet
pip install datasets nltk --quiet

# 2. Prepare data
echo ">>> Preparing training data..."
python tools/prepare_finetune_data.py --dataset all --eval-ratio 0.1

# Verify data files exist
for f in aggrefact_train.jsonl aggrefact_eval.jsonl \
         medical_train.jsonl medical_eval.jsonl \
         legal_train.jsonl legal_eval.jsonl; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "ERROR: $DATA_DIR/$f not found"
        exit 1
    fi
    echo "  ✓ $f ($(wc -l < "$DATA_DIR/$f") lines)"
done

# 3. Run 1: General factuality (AggreFact)
echo ""
echo "=== Run 1: General Factuality (AggreFact) ==="
director-ai finetune "$DATA_DIR/aggrefact_train.jsonl" \
    --eval "$DATA_DIR/aggrefact_eval.jsonl" \
    --output "$MODELS_DIR/factcg-aggrefact" \
    --epochs 3 \
    --lr 2e-5 \
    --batch-size 16

# 4. Run 2: Medical domain (MedNLI + PubMedQA)
echo ""
echo "=== Run 2: Medical Domain (MedNLI + PubMedQA) ==="
director-ai finetune "$DATA_DIR/medical_train.jsonl" \
    --eval "$DATA_DIR/medical_eval.jsonl" \
    --output "$MODELS_DIR/factcg-medical" \
    --epochs 3 \
    --lr 2e-5 \
    --batch-size 16

# 5. Run 3: Legal domain (ContractNLI)
echo ""
echo "=== Run 3: Legal Domain (ContractNLI) ==="
director-ai finetune "$DATA_DIR/legal_train.jsonl" \
    --eval "$DATA_DIR/legal_eval.jsonl" \
    --output "$MODELS_DIR/factcg-legal" \
    --epochs 3 \
    --lr 2e-5 \
    --batch-size 16

# 6. Benchmark with fine-tuned models
echo ""
echo "=== Benchmarking Fine-tuned Models ==="

echo ">>> AggreFact sweep (general model)..."
python -m benchmarks.aggrefact_eval \
    --model "$MODELS_DIR/factcg-aggrefact" --sweep 2>&1 | tee "$MODELS_DIR/aggrefact_bench.log"

echo ">>> MedNLI (medical model)..."
python -m benchmarks.medical_eval \
    --dataset mednli --model "$MODELS_DIR/factcg-medical" 2>&1 | tee "$MODELS_DIR/medical_bench.log"

echo ">>> ContractNLI (legal model)..."
python -m benchmarks.legal_eval \
    --dataset contractnli --model "$MODELS_DIR/factcg-legal" 2>&1 | tee "$MODELS_DIR/legal_bench.log"

# 7. Also benchmark baseline for comparison
echo ""
echo ">>> Baseline AggreFact (original FactCG)..."
python -m benchmarks.aggrefact_eval --sweep 2>&1 | tee "$MODELS_DIR/baseline_bench.log"

echo ""
echo "=== Training Complete ==="
echo "Models saved in $MODELS_DIR/"
ls -lh "$MODELS_DIR/"
echo ""
echo "Download models, then shut down GPU server to stop billing."
echo "scp -r user@gpu-server:$MODELS_DIR/ ./models/"
