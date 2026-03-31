#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-AI — One-Command UpCloud GPU Benchmark Deployment
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
#
# Prerequisites:
#   - UpCloud API token in UPCLOUD_TOKEN env var
#   - ANTHROPIC_API_KEY and OPENAI_API_KEY set
#   - SSH key ~/.ssh/id_ed25519.pub registered
#   - GPU slot available (account limit: 1)
#
# Usage:
#   export UPCLOUD_TOKEN="ucat_..."
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export OPENAI_API_KEY="sk-proj-..."
#   bash benchmarks/deploy_upcloud_bench.sh
#
# What it does:
#   1. Provisions GPU-8xCPU-64GB-1xL40S in fi-hel2
#   2. Waits for server to boot (~2 min)
#   3. SSHs in, installs Python env + director-ai
#   4. Uploads benchmark scripts
#   5. Launches benchmarks via nohup
#   6. Prints monitoring instructions
#
# Estimated runtime:
#   - Setup: ~5 min
#   - Hybrid-Claude (300 samples): ~15 min
#   - Hybrid-OpenAI (300 samples): ~15 min
#   - AggreFact (29K samples): ~30 min
#   - RAGTruth NLI: ~20 min
#   - FreshQA NLI: ~10 min
#   - Total: ~1.5 hours + setup
#
# Cost: GPU-8xCPU-64GB-1xL40S ~€2.50/hr → ~€5 total
#
# After benchmarks complete:
#   1. scp results back: scp root@<IP>:/opt/director-bench/work/director-ai/benchmarks/results/*.json benchmarks/results/
#   2. Delete server: curl -X DELETE -H "Authorization: Bearer $UPCLOUD_TOKEN" https://api.upcloud.com/1.3/server/<UUID>?storages=1&backups=delete
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# Validate env
: "${UPCLOUD_TOKEN:?Set UPCLOUD_TOKEN}"
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY=$(cat ~/.ssh/id_ed25519.pub)
ZONE="fi-hel2"
PLAN="GPU-8xCPU-64GB-1xL40S"
TEMPLATE="01000000-0000-4000-8000-000030700200"  # Ubuntu 24.04 + NVIDIA

echo "=== [1/7] Provisioning $PLAN in $ZONE ==="
RESPONSE=$(curl -s -X POST \
  -H "Authorization: Bearer $UPCLOUD_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.upcloud.com/1.3/server" \
  -d "{
    \"server\": {
      \"zone\": \"$ZONE\",
      \"title\": \"director-ai-bench\",
      \"hostname\": \"director-bench\",
      \"plan\": \"$PLAN\",
      \"metadata\": \"yes\",
      \"storage_devices\": {
        \"storage_device\": [{
          \"action\": \"clone\",
          \"storage\": \"$TEMPLATE\",
          \"title\": \"director-bench-os\",
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

echo "=== [2/7] Waiting for server to start ==="
for i in $(seq 1 20); do
  STATE=$(curl -s -H "Authorization: Bearer $UPCLOUD_TOKEN" \
    "https://api.upcloud.com/1.3/server/$SERVER_UUID" | \
    python3 -c "import sys,json; print(json.loads(sys.stdin.read())['server']['state'])")
  echo "  $(date +%H:%M:%S) State: $STATE"
  [ "$STATE" = "started" ] && break
  sleep 15
done

echo "=== [3/7] Waiting for SSH ==="
for i in $(seq 1 12); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@"$SERVER_IP" "echo SSH OK" 2>/dev/null && break
  sleep 10
done

echo "=== [4/7] Installing Python env + director-ai ==="
ssh -o StrictHostKeyChecking=no root@"$SERVER_IP" bash -s <<'SETUP'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git wget > /dev/null 2>&1
python3 -m venv /opt/director-bench
source /opt/director-bench/bin/activate
pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install --quiet transformers==4.49.0 datasets==3.3.2 accelerate==1.4.0 scikit-learn==1.6.1
pip install --quiet sentence-transformers==4.0.2 chromadb==0.6.3
pip install --quiet openai==1.68.2 anthropic==0.49.0 requests==2.32.3 numpy==2.2.3
cd /opt/director-bench
git clone --depth 1 https://github.com/anulum/director-ai.git work/director-ai
cd work/director-ai
pip install --quiet -e ".[nli,vector,dev]"
python3 -c "from director_ai import __version__; print(f'director-ai v{__version__}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory//1024**3}GB)')"
echo "SETUP COMPLETE"
SETUP

echo "=== [5/7] Uploading benchmark scripts ==="
scp -o StrictHostKeyChecking=no \
  "$SCRIPT_DIR/run_cloud_benchmarks.py" \
  "$SCRIPT_DIR/run_ragtruth_freshqa.py" \
  root@"$SERVER_IP":/opt/director-bench/work/director-ai/benchmarks/

echo "=== [6/7] Launching benchmarks ==="
ssh -o StrictHostKeyChecking=no root@"$SERVER_IP" bash -s <<LAUNCH
set -euo pipefail
source /opt/director-bench/bin/activate
cd /opt/director-bench/work/director-ai
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export OPENAI_API_KEY="$OPENAI_API_KEY"

# Run hybrid + aggrefact first
nohup python3 benchmarks/run_cloud_benchmarks.py > /tmp/bench_hybrid.log 2>&1 &
echo "Hybrid PID: \$!"

# Run ragtruth + freshqa in parallel
nohup python3 benchmarks/run_ragtruth_freshqa.py > /tmp/bench_ragfresh.log 2>&1 &
echo "RAGTruth+FreshQA PID: \$!"

echo "LAUNCHED"
LAUNCH

echo "=== [7/7] Deployment complete ==="
echo ""
echo "Server: $SERVER_IP (UUID: $SERVER_UUID)"
echo ""
echo "Monitor:"
echo "  ssh root@$SERVER_IP tail -f /tmp/bench_hybrid.log"
echo "  ssh root@$SERVER_IP tail -f /tmp/bench_ragfresh.log"
echo ""
echo "Collect results:"
echo "  scp root@$SERVER_IP:/opt/director-bench/work/director-ai/benchmarks/results/*.json benchmarks/results/"
echo ""
echo "Delete server when done:"
echo "  curl -s -X POST -H 'Authorization: Bearer $UPCLOUD_TOKEN' https://api.upcloud.com/1.3/server/$SERVER_UUID/stop -d '{\"stop_server\":{\"stop_type\":\"soft\"}}'"
echo "  sleep 30"
echo "  curl -s -X DELETE -H 'Authorization: Bearer $UPCLOUD_TOKEN' 'https://api.upcloud.com/1.3/server/$SERVER_UUID?storages=1&backups=delete'"
