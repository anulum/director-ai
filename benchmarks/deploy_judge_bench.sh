#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-AI — UpCloud Local Judge Benchmark Deployment
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
#
# Deploys an UpCloud L40S GPU server, uploads the trained judge model,
# runs NLI-only vs local judge comparison on HaluEval, collects results.
#
# Prerequisites:
#   - UPCLOUD_TOKEN env var (UpCloud API token)
#   - SSH key ~/.ssh/id_ed25519.pub registered with UpCloud
#   - Judge model at training/output/deberta-v3-base-judge/
#
# Usage:
#   export UPCLOUD_TOKEN="ucat_..."
#   bash benchmarks/deploy_judge_bench.sh [--samples 500]
#
# Cost estimate:
#   - 500 samples/task: ~1.5h on L40S → ~€1.80
#   - 1000 samples/task: ~3h on L40S → ~€3.60
#   - 10000 samples/task: ~15h on L40S → ~€18
#
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SAMPLES="${1:-500}"
if [[ "$1" == "--samples" ]]; then
    SAMPLES="${2:-500}"
fi

: "${UPCLOUD_TOKEN:?Set UPCLOUD_TOKEN}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JUDGE_DIR="$REPO_DIR/training/output/deberta-v3-base-judge"
SSH_KEY=$(cat ~/.ssh/id_ed25519.pub)
ZONE="fi-hel2"
PLAN="GPU-8xCPU-64GB-1xL40S"
TEMPLATE="01000000-0000-4000-8000-000030700200"  # Ubuntu 24.04 + NVIDIA

# Verify judge model exists locally
if [ ! -f "$JUDGE_DIR/model.safetensors" ]; then
    echo "ERROR: Judge model not found at $JUDGE_DIR/model.safetensors"
    echo "Train it first: python training/train_judge.py"
    exit 1
fi

echo "=== Director-AI Local Judge Benchmark ==="
echo "  Samples/task: $SAMPLES"
echo "  Judge model:  $JUDGE_DIR ($(du -sh "$JUDGE_DIR/model.safetensors" | cut -f1))"
echo ""

echo "=== [1/8] Provisioning $PLAN in $ZONE ==="
RESPONSE=$(curl -s -X POST \
  -H "Authorization: Bearer $UPCLOUD_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.upcloud.com/1.3/server" \
  -d "{
    \"server\": {
      \"zone\": \"$ZONE\",
      \"title\": \"director-ai-judge-bench\",
      \"hostname\": \"judge-bench\",
      \"plan\": \"$PLAN\",
      \"metadata\": \"yes\",
      \"storage_devices\": {
        \"storage_device\": [{
          \"action\": \"clone\",
          \"storage\": \"$TEMPLATE\",
          \"title\": \"judge-bench-os\",
          \"size\": 80,
          \"tier\": \"maxiops\"
        }]
      },
      \"login_user\": {
        \"username\": \"root\",
        \"ssh_keys\": {\"ssh_key\": [\"$SSH_KEY\"]}
      },
      \"networking\": {
        \"interfaces\": {
          \"interface\": [{
            \"ip_addresses\": {\"ip_address\": [{\"family\": \"IPv4\"}]},
            \"type\": \"public\"
          }]
        }
      }
    }
  }")

SERVER_UUID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['server']['uuid'])")
SERVER_IP=$(echo "$RESPONSE" | python3 -c "
import sys,json
s=json.loads(sys.stdin.read())['server']
for ip in s['ip_addresses']['ip_address']:
    if ip['family']=='IPv4' and ip['access']=='public':
        print(ip['address'])
        break
")
echo "  UUID: $SERVER_UUID"
echo "  IP:   $SERVER_IP"

# Save server info for cleanup
echo "$SERVER_UUID" > /tmp/judge_bench_uuid.txt
echo "$SERVER_IP" > /tmp/judge_bench_ip.txt

echo "=== [2/8] Waiting for server to start ==="
for i in $(seq 1 20); do
  STATE=$(curl -s -H "Authorization: Bearer $UPCLOUD_TOKEN" \
    "https://api.upcloud.com/1.3/server/$SERVER_UUID" | \
    python3 -c "import sys,json; print(json.loads(sys.stdin.read())['server']['state'])")
  echo "  $(date +%H:%M:%S) State: $STATE"
  [ "$STATE" = "started" ] && break
  sleep 15
done

echo "=== [3/8] Waiting for SSH ==="
for i in $(seq 1 12); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@"$SERVER_IP" "echo SSH OK" 2>/dev/null && break
  sleep 10
done

echo "=== [4/8] Installing Python env + director-ai ==="
ssh -o StrictHostKeyChecking=no root@"$SERVER_IP" bash -s <<'SETUP'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git wget > /dev/null 2>&1
python3 -m venv /opt/director-bench
source /opt/director-bench/bin/activate
pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install --quiet transformers==4.49.0 datasets==3.3.2 accelerate==1.4.0 scikit-learn==1.6.1
pip install --quiet sentence-transformers==4.0.2 chromadb==0.6.3
pip install --quiet requests==2.32.3 numpy==2.2.3
cd /opt/director-bench
git clone --depth 1 https://github.com/anulum/director-ai.git work/director-ai
cd work/director-ai
pip install --quiet -e ".[nli,vector,dev]"
python3 -c "from director_ai import __version__; print(f'director-ai v{__version__}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory//1024**3}GB)')"
echo "SETUP COMPLETE"
SETUP

echo "=== [5/8] Uploading judge model (~707MB) ==="
# Upload only the files needed for inference (skip checkpoint subdirs)
ssh -o StrictHostKeyChecking=no root@"$SERVER_IP" \
  "mkdir -p /opt/director-bench/work/director-ai/training/output/deberta-v3-base-judge"

for f in model.safetensors config.json tokenizer_config.json \
         tokenizer.json special_tokens_map.json spm.model; do
    if [ -f "$JUDGE_DIR/$f" ]; then
        echo "  Uploading $f..."
        scp -o StrictHostKeyChecking=no \
            "$JUDGE_DIR/$f" \
            root@"$SERVER_IP":/opt/director-bench/work/director-ai/training/output/deberta-v3-base-judge/
    fi
done

echo "=== [6/8] Uploading benchmark script ==="
scp -o StrictHostKeyChecking=no \
  "$SCRIPT_DIR/run_judge_benchmark.py" \
  root@"$SERVER_IP":/opt/director-bench/work/director-ai/benchmarks/

echo "=== [7/8] Launching benchmark ==="
ssh -o StrictHostKeyChecking=no root@"$SERVER_IP" bash -s <<LAUNCH
set -euo pipefail
source /opt/director-bench/bin/activate
cd /opt/director-bench/work/director-ai

# Verify judge model loaded
python3 -c "
from transformers import AutoModelForSequenceClassification
m = AutoModelForSequenceClassification.from_pretrained('training/output/deberta-v3-base-judge')
print(f'Judge model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params')
"

nohup python3 benchmarks/run_judge_benchmark.py --samples $SAMPLES > /tmp/judge_bench.log 2>&1 &
echo "PID: \$!"
echo "LAUNCHED"
LAUNCH

echo "=== [8/8] Deployment complete ==="
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Server: $SERVER_IP  (UUID: $SERVER_UUID)"
echo "║  Samples: $SAMPLES per task"
echo "║"
echo "║  Monitor:"
echo "║    ssh root@$SERVER_IP tail -f /tmp/judge_bench.log"
echo "║"
echo "║  Collect results when done:"
echo "║    scp root@$SERVER_IP:/opt/director-bench/work/director-ai/benchmarks/results/judge_bench_*.json benchmarks/results/"
echo "║"
echo "║  Delete server:"
echo "║    curl -s -X POST -H 'Authorization: Bearer $UPCLOUD_TOKEN' \\"
echo "║      https://api.upcloud.com/1.3/server/$SERVER_UUID/stop \\"
echo "║      -d '{\"stop_server\":{\"stop_type\":\"soft\"}}'"
echo "║    sleep 30"
echo "║    curl -s -X DELETE -H 'Authorization: Bearer $UPCLOUD_TOKEN' \\"
echo "║      'https://api.upcloud.com/1.3/server/$SERVER_UUID?storages=1&backups=delete'"
echo "╚══════════════════════════════════════════════════════════════╝"
