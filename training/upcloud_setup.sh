#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# UpCloud L40S Setup — Director AI DeBERTa-v3-large Training
# Run this ONCE after SSH-ing into the fresh UpCloud GPU server.
# Total runtime: ~5 min setup + ~3-4 hours training
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "=== [1/5] System packages ==="
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git wget > /dev/null 2>&1

echo "=== [2/5] Python venv ==="
python3 -m venv /opt/director-train
source /opt/director-train/bin/activate

echo "=== [3/5] PyTorch + dependencies ==="
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
pip install --quiet transformers datasets accelerate scikit-learn

echo "=== [4/5] Verify GPU ==="
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'GPU: {name} ({vram:.0f} GB)')
"

echo "=== [5/5] Create working directory ==="
mkdir -p /opt/director-train/work
cd /opt/director-train/work

echo ""
echo "============================================"
echo "  Setup complete. Next steps:"
echo "  1. Upload train_cloud_large.py to /opt/director-train/work/"
echo "  2. Activate: source /opt/director-train/bin/activate"
echo "  3. Run:      cd /opt/director-train/work && python3 train_cloud_large.py"
echo "============================================"
