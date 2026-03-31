#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Pack deployment tarball
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
# Creates director_ai_gpu.tar.gz with code only (no models/data).
# Run from the DIRECTOR_AI root directory.
set -euo pipefail

cd "$(dirname "$0")/.."

tar czf director_ai_gpu.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='models' \
    --exclude='node_modules' \
    --exclude='*.egg-info' \
    --exclude='backfire-kernel' \
    --exclude='docs' \
    --exclude='docs-site' \
    --exclude='notebooks' \
    --exclude='demo' \
    --exclude='.github' \
    --exclude='benchmarks/results' \
    --exclude='benchmarks/models' \
    --exclude='benchmarks/scores' \
    --exclude='benchmarks/.cache' \
    --exclude='*.jsonl' \
    --exclude='*.pkl' \
    --exclude='*.pt' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    src/ benchmarks/ tools/ gpu_deploy/ pyproject.toml

SIZE=$(du -h director_ai_gpu.tar.gz | cut -f1)
echo "Created director_ai_gpu.tar.gz ($SIZE)"
echo "Upload: scp director_ai_gpu.tar.gz root@<IP>:/root/"
