#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Provision UpCloud L40S for Distillation Pipeline
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
#
# Provisions UpCloud L40S (48GB VRAM), uploads code, starts pipeline.
# Requires: UPCLOUD_API_TOKEN, SSH key at ~/.ssh/id_ed25519_upcloud
#
set -euo pipefail

API_TOKEN="${UPCLOUD_API_TOKEN:-ucat_01KKD8KM5KBB5RRZXYAR7HP6KM}"
SSH_KEY=$(cat ~/.ssh/id_ed25519_upcloud.pub)
ZONE="fi-hel2"
PLAN="GPU-8xCPU-64GB-1xL40S"
OS_TEMPLATE="01000000-0000-4000-8000-000030700200"
HOSTNAME="director-distill-$(date +%Y%m%d)"

echo "Provisioning $PLAN in $ZONE..."

RESPONSE=$(curl -s -H "Authorization: Bearer $API_TOKEN" \
  -X POST "https://api.upcloud.com/1.3/server" \
  -H "Content-Type: application/json" \
  -d "{
    \"server\": {
      \"zone\": \"$ZONE\",
      \"title\": \"$HOSTNAME\",
      \"hostname\": \"$HOSTNAME\",
      \"plan\": \"$PLAN\",
      \"metadata\": \"yes\",
      \"storage_devices\": {
        \"storage_device\": [{
          \"action\": \"clone\",
          \"storage\": \"$OS_TEMPLATE\",
          \"title\": \"OS disk\",
          \"size\": 80,
          \"tier\": \"maxiops\"
        }]
      },
      \"login_user\": {
        \"username\": \"root\",
        \"ssh_keys\": {\"ssh_key\": [\"$SSH_KEY\"]}
      }
    }
  }")

UUID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['uuid'])")
IP=$(echo "$RESPONSE" | python3 -c "
import sys,json
s=json.load(sys.stdin)['server']
for iface in s.get('ip_addresses',{}).get('ip_address',[]):
    if iface.get('access')=='public' and iface.get('family')=='IPv4':
        print(iface['address']); break
")

echo "UUID: $UUID"
echo "IP:   $IP"
echo "Waiting for server to start..."

for i in $(seq 1 60); do
    STATE=$(curl -s -H "Authorization: Bearer $API_TOKEN" \
      "https://api.upcloud.com/1.3/server/$UUID" | \
      python3 -c "import sys,json; print(json.load(sys.stdin)['server']['state'])")
    if [ "$STATE" = "started" ]; then
        echo "Server started."
        break
    fi
    sleep 10
done

echo "Waiting for SSH..."
for i in $(seq 1 30); do
    if ssh -i ~/.ssh/id_ed25519_upcloud -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@"$IP" 'echo OK' 2>/dev/null; then
        break
    fi
    sleep 10
done

echo "Setting up environment..."
ssh -i ~/.ssh/id_ed25519_upcloud -o StrictHostKeyChecking=no root@"$IP" bash -s <<'SETUP'
set -e
python3 -m venv /opt/director-venv
source /opt/director-venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets scikit-learn peft bitsandbytes accelerate
pip install -e /root/director-ai
SETUP

echo "Uploading codebase..."
rsync -az --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='node_modules' --exclude='.venv' --exclude='gpu_results' \
  --exclude='models' --exclude='*.pt' --exclude='*.safetensors' \
  -e "ssh -i ~/.ssh/id_ed25519_upcloud -o StrictHostKeyChecking=no" \
  ./ root@"$IP":/root/director-ai/

echo "Starting distillation pipeline..."
ssh -i ~/.ssh/id_ed25519_upcloud -o StrictHostKeyChecking=no root@"$IP" \
  "cd /root/director-ai && source /opt/director-venv/bin/activate && \
   nohup bash gpu_deploy/distillation_master.sh > /root/director-ai/logs/master_console.log 2>&1 &"

echo ""
echo "=== PROVISIONED ==="
echo "UUID: $UUID"
echo "IP:   $IP"
echo "Monitor: ssh -i ~/.ssh/id_ed25519_upcloud root@$IP 'cat /root/director-ai/STATUS'"
echo "Destroy: curl -s -H \"Authorization: Bearer $API_TOKEN\" -X POST https://api.upcloud.com/1.3/server/$UUID/stop -H 'Content-Type: application/json' -d '{\"stop_server\":{\"stop_type\":\"soft\",\"timeout\":\"60\"}}'"
