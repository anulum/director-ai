#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Download GPU Results
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
# Usage: bash gpu_deploy/download_results.sh <server-ip>
#
# Downloads results_package.tar.gz (adapters + results + logs).
# For merged models (large), use --full flag.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

IP="${1:?Usage: download_results.sh <server-ip> [--full]}"
FULL="${2:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_upcloud}"
LOCAL_DIR="gpu_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOCAL_DIR"
echo "Downloading from $IP to $LOCAL_DIR/"

# Always download the light package first
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "root@${IP}:/root/director-ai/results_package.tar.gz" \
    "$LOCAL_DIR/" 2>/dev/null && echo "  results_package.tar.gz downloaded" || echo "  results_package.tar.gz not found"

# Download STATUS and summary
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "root@${IP}:/root/director-ai/STATUS" \
    "$LOCAL_DIR/" 2>/dev/null || true

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "root@${IP}:/root/director-ai/results/RESULTS_SUMMARY.md" \
    "$LOCAL_DIR/" 2>/dev/null || true

# Full download: includes merged models (~1.5GB each)
if [ "$FULL" = "--full" ]; then
    echo "Full download: including merged models..."
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r \
        "root@${IP}:/root/director-ai/models/" \
        "$LOCAL_DIR/models/" 2>/dev/null || true
fi

echo ""
echo "=== Downloaded ==="
ls -lh "$LOCAL_DIR/"
echo ""

# Unpack and show summary
if [ -f "$LOCAL_DIR/results_package.tar.gz" ]; then
    cd "$LOCAL_DIR"
    tar xzf results_package.tar.gz
    echo "=== RESULTS SUMMARY ==="
    cat results/RESULTS_SUMMARY.md 2>/dev/null || echo "(no summary)"
fi
