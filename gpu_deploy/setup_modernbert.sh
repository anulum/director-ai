#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — ModernBERT GPU Instance Setup
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
# Run once after uploading director_ai_modernbert.tar.gz to /root/.
# No HF_TOKEN required — eval uses local JSONL.
#
# Usage:
#   bash gpu_deploy/setup_modernbert.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

WORK_DIR="/root/director-ai"

echo "=== Director-AI ModernBERT Setup ==="
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Unpack project
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
if [ -f /root/director_ai_modernbert.tar.gz ]; then
    tar xzf /root/director_ai_modernbert.tar.gz
    echo "Project unpacked to $WORK_DIR"
else
    echo "ERROR: /root/director_ai_modernbert.tar.gz not found"
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
pip install -r gpu_deploy/requirements_modernbert.txt -q
pip install -e ".[nli]" -q

# Flash Attention 2 (optional — sdpa fallback is fine)
echo ""
echo "=== Flash Attention Install (may take 5-10 min) ==="
if pip install flash-attn --no-build-isolation -q 2>/dev/null; then
    echo "flash-attn installed"
else
    echo "flash-attn build failed — using sdpa (no performance impact on L40S)"
fi

# NLTK tokenizer data
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Working directories
mkdir -p models results logs data

# Copy eval data to expected location
if [ -f benchmarks/aggrefact_test.jsonl ] && [ ! -f data/aggrefact_test.jsonl ]; then
    cp benchmarks/aggrefact_test.jsonl data/aggrefact_test.jsonl
fi

# Verify GPU + bf16
echo ""
echo "=== GPU Verification ==="
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_mem / 1e9:.1f} GB')
    print(f'bf16: {torch.cuda.is_bf16_supported()}')
    assert torch.cuda.is_bf16_supported(), 'bf16 required for ModernBERT'
"

# Verify transformers version
echo ""
echo "=== Package Versions ==="
python3 -c "
import transformers, torch, datasets
v = transformers.__version__
major, minor = int(v.split('.')[0]), int(v.split('.')[1])
assert major >= 5 or (major == 4 and minor >= 48), f'transformers >= 4.48 required, got {v}'
print(f'transformers {v}')
print(f'torch {torch.__version__}')
print(f'datasets {datasets.__version__}')
try:
    import flash_attn
    print(f'flash_attn {flash_attn.__version__}')
except ImportError:
    print('flash_attn: not installed (using sdpa)')
"

# Pre-download ModernBERT checkpoint
echo ""
echo "=== Pre-downloading ModernBERT-large ==="
python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-large')
mdl = AutoModelForSequenceClassification.from_pretrained(
    'answerdotai/ModernBERT-large', num_labels=2, attn_implementation='sdpa',
)
n = sum(p.numel() for p in mdl.parameters())
print(f'ModernBERT-large: {n/1e6:.0f}M params')

inputs = tok('Forward pass test', return_tensors='pt', truncation=True, max_length=512)
with torch.no_grad():
    out = mdl(**inputs)
assert out.logits.shape == (1, 2), f'Expected (1,2), got {out.logits.shape}'
print('Forward pass: OK')
"

# Probe NLI checkpoint candidate
echo ""
echo "=== Probing NLI Checkpoint ==="
python3 -c "
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained('MoritzLaurer/ModernBERT-large-zeroshot-v2.0')
    nl = getattr(cfg, 'num_labels', '?')
    i2l = getattr(cfg, 'id2label', '?')
    print(f'MoritzLaurer/ModernBERT-large-zeroshot-v2.0: num_labels={nl}, id2label={i2l}')
except Exception as e:
    print(f'Could not probe checkpoint: {e}')
    print('Will fall back to full Stage 1 training')
" || true

# Verify eval data
echo ""
echo "=== Data Verification ==="
python3 -c "
import os
for p in ('data/aggrefact_test.jsonl', 'benchmarks/aggrefact_test.jsonl'):
    if os.path.exists(p):
        n = sum(1 for _ in open(p))
        print(f'{p}: {n} samples')
        assert n > 25000, f'Expected 25K+ samples, got {n}'
        break
else:
    raise FileNotFoundError('aggrefact_test.jsonl not found')
"

# Disk space
echo ""
echo "=== Disk Space ==="
df -h / | tail -1

echo ""
echo "=== Setup Complete ==="
echo "Launch: nohup bash gpu_deploy/modernbert_master.sh > logs/master.log 2>&1 &"
echo "Monitor: cat STATUS | tail -20"
