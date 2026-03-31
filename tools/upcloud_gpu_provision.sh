#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — UpCloud GPU Server Provisioner (retry loop)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
#
# Polls UpCloud API for GPU availability. When a slot opens, creates
# the server, outputs the IP, and exits.
#
# Usage:
#   export UPCLOUD_TOKEN=ucat_...
#   bash tools/upcloud_gpu_provision.sh
#   # Polls every 5 minutes until GPU available (up to 24h)
#
set -euo pipefail

: "${UPCLOUD_TOKEN:?Set UPCLOUD_TOKEN}"
: "${SSH_PUBKEY_FILE:=/tmp/upcloud_training_key.pub}"

if [ ! -f "$SSH_PUBKEY_FILE" ]; then
    echo "Generating SSH keypair..."
    ssh-keygen -t ed25519 -f /tmp/upcloud_training_key -N "" -C "director-ai-training"
fi
SSH_PUBKEY=$(cat "$SSH_PUBKEY_FILE")

PLANS=("GPU-8xCPU-64GB-1xL40S" "GPU-12xCPU-128GB-1xL40S")
ZONES=("fi-hel2" "fi-hel1" "de-fra1" "nl-ams1" "se-sto1" "uk-lon1")
POLL_INTERVAL=300  # 5 minutes
MAX_ATTEMPTS=288   # 24 hours

attempt=0
while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    echo "[$(date +%H:%M:%S)] Attempt $attempt/$MAX_ATTEMPTS"

    for plan in "${PLANS[@]}"; do
        for zone in "${ZONES[@]}"; do
            result=$(curl -s -X POST \
                -H "Authorization: Bearer $UPCLOUD_TOKEN" \
                -H "Content-Type: application/json" \
                "https://api.upcloud.com/1.3/server" \
                -d "{
                    \"server\": {
                        \"zone\": \"$zone\",
                        \"title\": \"director-ai-training\",
                        \"hostname\": \"director-train\",
                        \"plan\": \"$plan\",
                        \"metadata\": \"yes\",
                        \"storage_devices\": {
                            \"storage_device\": [{
                                \"action\": \"clone\",
                                \"storage\": \"01000000-0000-4000-8000-000030700200\",
                                \"title\": \"director-ai-os\",
                                \"size\": 80,
                                \"tier\": \"maxiops\"
                            }]
                        },
                        \"networking\": {
                            \"interfaces\": {
                                \"interface\": [{
                                    \"ip_addresses\": {\"ip_address\": [{\"family\": \"IPv4\"}]},
                                    \"type\": \"public\"
                                }]
                            }
                        },
                        \"login_user\": {
                            \"username\": \"root\",
                            \"ssh_keys\": {\"ssh_key\": [\"$SSH_PUBKEY\"]}
                        }
                    }
                }" 2>&1)

            if echo "$result" | grep -q '"uuid"'; then
                echo ""
                echo "=== GPU SERVER CREATED ==="
                echo "$result" | python3 -c "
import json, sys
s = json.load(sys.stdin).get('server', {})
ips = []
for iface in s.get('networking', {}).get('interfaces', {}).get('interface', []):
    for ip in iface.get('ip_addresses', {}).get('ip_address', []):
        ips.append(ip.get('address', ''))
print(f'UUID:  {s.get(\"uuid\")}')
print(f'Plan:  {s.get(\"plan\")}')
print(f'Zone:  {s.get(\"zone\")}')
print(f'State: {s.get(\"state\")}')
print(f'IPs:   {\" \".join(ips)}')
with open('/tmp/upcloud_server_uuid.txt', 'w') as f:
    f.write(s.get('uuid', ''))
with open('/tmp/upcloud_server_ip.txt', 'w') as f:
    f.write(' '.join(ips))
"
                echo ""
                echo "SSH key: /tmp/upcloud_training_key"
                echo "Connect: ssh -i /tmp/upcloud_training_key root@\$(cat /tmp/upcloud_server_ip.txt)"
                exit 0
            fi
        done
    done

    echo "  No GPU available. Retrying in ${POLL_INTERVAL}s..."
    sleep $POLL_INTERVAL
done

echo "Timed out after $MAX_ATTEMPTS attempts."
exit 1
