#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# UpCloud L40S Setup — Director-AI GPU Benchmarks
# Run this ONCE after SSH-ing into the fresh UpCloud GPU server.
# Total: ~5 min setup, ~2-3 hours benchmarks
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "=== [1/6] System packages ==="
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git wget > /dev/null 2>&1

echo "=== [2/6] Python venv ==="
python3 -m venv /opt/director-bench
source /opt/director-bench/bin/activate

echo "=== [3/6] PyTorch + dependencies ==="
pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install --quiet transformers==4.49.0 datasets==3.3.2 accelerate==1.4.0 scikit-learn==1.6.1
pip install --quiet sentence-transformers==4.0.2 chromadb==0.6.3
pip install --quiet openai==1.68.2 anthropic==0.49.0 requests==2.32.3 numpy==2.2.3

echo "=== [4/6] Verify GPU ==="
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'GPU: {name} ({vram:.0f} GB)')
"

echo "=== [5/6] Clone repo ==="
cd /opt/director-bench
git clone --depth 1 https://github.com/anulum/director-ai.git work/director-ai
cd work/director-ai
pip install --quiet -e ".[nli,vector,dev]"

echo "=== [6/6] Set API keys ==="
# Keys are passed via environment — see provision script
echo "ANTHROPIC_API_KEY set: ${ANTHROPIC_API_KEY:+yes}"
echo "OPENAI_API_KEY set: ${OPENAI_API_KEY:+yes}"

echo ""
echo "============================================"
echo "  Setup complete. Run benchmarks:"
echo "  source /opt/director-bench/bin/activate"
echo "  cd /opt/director-bench/work/director-ai"
echo "  python benchmarks/run_cloud_benchmarks.py 2>&1 | tee /tmp/bench.log"
echo "============================================"
