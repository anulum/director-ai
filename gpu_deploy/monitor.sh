#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Remote Training Monitor
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
# Usage: bash gpu_deploy/monitor.sh <server-ip>
# ─────────────────────────────────────────────────────────────────────
IP="${1:?Usage: monitor.sh <server-ip>}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_upcloud}"

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "root@${IP}" '
echo "=== STATUS ==="
tail -10 /root/director-ai/STATUS 2>/dev/null || echo "No status file"
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "No GPU"
echo ""
echo "=== DISK ==="
df -h / | tail -1
echo ""
echo "=== SCREEN ==="
screen -ls 2>/dev/null || echo "No screen sessions"
echo ""
echo "=== LATEST LOG (last 5 lines) ==="
LATEST=$(ls -t /root/director-ai/logs/*.log 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "File: $LATEST"
    tail -5 "$LATEST"
else
    echo "No logs yet"
fi
'
