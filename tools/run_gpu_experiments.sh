#!/bin/bash
# GPU experiment runner: CB-lowLR redo + NCA synthetic NLI
#
# Usage (on JarvisLabs or UpCloud GPU instance):
#   bash tools/run_gpu_experiments.sh
#
# Prerequisites:
#   - pip install transformers datasets scikit-learn torch accelerate sentencepiece protobuf
#   - benchmarks/aggrefact_test.jsonl must exist in $WORKDIR
#   - CUDA GPU with >= 16GB VRAM (24GB recommended)
#
# Total time: ~3h (15 min CB + 30 min gen + 2h NCA train + 40 min scoring x2)

set -euo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True
export DIRECTOR_WORKDIR="${DIRECTOR_WORKDIR:-/home/user/director-ai}"

cd "$DIRECTOR_WORKDIR"
ln -sf "$DIRECTOR_WORKDIR" /home/director-ai 2>/dev/null || true

echo "================================================"
echo "GPU Experiments — $(date -u)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Working dir: $DIRECTOR_WORKDIR"
echo "================================================"

# --- Experiment 1: CB-lowLR redo ---
echo ""
echo "=== [1/2] CB-lowLR redo (LR=5e-6, 20 epochs) ==="
python tools/run_cb_lowlr_redo.py 2>&1 | tee /tmp/cb_lowlr.log

# --- Experiment 2: NCA synthetic NLI ---
echo ""
echo "=== [2/2] NCA synthetic NLI (50K samples, LR=5e-6, 3 epochs) ==="
python tools/run_nca_synthetic_nli.py --n-samples 50000 2>&1 | tee /tmp/nca_synthetic.log

echo ""
echo "================================================"
echo "ALL EXPERIMENTS COMPLETE — $(date -u)"
echo ""
echo "Download:"
echo "  scp -P PORT root@HOST:$DIRECTOR_WORKDIR/scores/factcg-cb-lowlr.json ./benchmarks/scores/"
echo "  scp -P PORT root@HOST:$DIRECTOR_WORKDIR/scores/factcg-nca-synthetic.json ./benchmarks/scores/"
echo "  scp -P PORT root@HOST:$DIRECTOR_WORKDIR/models/factcg-cb-lowlr/training_result.json ./benchmarks/models/"
echo "  scp -P PORT root@HOST:$DIRECTOR_WORKDIR/models/factcg-nca-synthetic/training_result.json ./benchmarks/models/"
echo "================================================"
