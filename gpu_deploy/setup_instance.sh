#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — GPU Instance Setup
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
# Run once after SSH into the GPU instance.
#
# Prerequisites:
#   - director_ai_gpu.tar.gz uploaded to /root/
#   - HF_TOKEN env var set (for gated AggreFact dataset)
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash setup_instance.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

WORK_DIR="/root/director-ai"

echo "=== Director-AI GPU Setup ==="
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Unpack project
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
if [ -f /root/director_ai_gpu.tar.gz ]; then
    tar xzf /root/director_ai_gpu.tar.gz
    echo "Project unpacked to $WORK_DIR"
else
    echo "ERROR: /root/director_ai_gpu.tar.gz not found"
    exit 1
fi

# System packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq screen htop python3-pip python3-venv 2>/dev/null || true

# Python venv
python3 -m venv /opt/director-venv
source /opt/director-venv/bin/activate

pip install --upgrade pip setuptools wheel -q

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# Project dependencies
pip install -r gpu_deploy/requirements_gpu.txt -q
pip install -e ".[nli]" -q

# NLTK tokenizer data
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Create working directories
mkdir -p models features labels results logs benchmarks/results

# Verify GPU
echo ""
echo "=== GPU Verification ==="
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_mem / 1e9:.1f} GB')
"

# Verify model loading
echo ""
echo "=== Model Verification ==="
python3 -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained('yaxili96/FactCG-DeBERTa-v3-Large')
mdl = AutoModelForSequenceClassification.from_pretrained('yaxili96/FactCG-DeBERTa-v3-Large')
print(f'FactCG loaded: {sum(p.numel() for p in mdl.parameters())/1e6:.0f}M params')
"

# Verify AggreFact access
echo ""
echo "=== Dataset Verification ==="
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set — AggreFact download will fail"
    echo "Set it: export HF_TOKEN=hf_..."
else
    python3 -c "
from datasets import load_dataset
import os
ds = load_dataset('lytang/LLM-AggreFact', split='test', token=os.environ['HF_TOKEN'])
print(f'AggreFact loaded: {len(ds)} samples')
" || echo "WARNING: AggreFact load failed — check HF_TOKEN"
fi

# Disk space check
echo ""
echo "=== Disk Space ==="
df -h / | tail -1

echo ""
echo "=== Setup Complete ==="
echo "Next: screen -S train bash gpu_deploy/master_runner.sh"
echo "Detach: Ctrl+A, D"
echo "Reattach: screen -r train"
