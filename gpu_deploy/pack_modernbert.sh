#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Pack ModernBERT deployment tarball
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
# Creates director_ai_modernbert.tar.gz with code + eval data.
# Unlike pack.sh, this INCLUDES aggrefact_test.jsonl for offline eval.
# Run from the DIRECTOR_AI root directory.
set -euo pipefail

cd "$(dirname "$0")/.."

# Verify eval data exists before packing
if [ ! -f benchmarks/aggrefact_test.jsonl ]; then
    echo "ERROR: benchmarks/aggrefact_test.jsonl not found"
    echo "This file is required for GPU evaluation."
    exit 1
fi

tar czf director_ai_modernbert.tar.gz \
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
    --exclude='*.pkl' \
    --exclude='*.pt' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    --exclude='gpu_results' \
    src/ \
    benchmarks/aggrefact_test.jsonl \
    benchmarks/_common.py \
    benchmarks/aggrefact_eval.py \
    tools/ \
    gpu_deploy/ \
    pyproject.toml

SIZE=$(du -h director_ai_modernbert.tar.gz | cut -f1)
echo "Created director_ai_modernbert.tar.gz ($SIZE)"

# Verify critical files are in the tarball
echo ""
echo "Verifying tarball contents..."
tar tzf director_ai_modernbert.tar.gz | grep -q 'benchmarks/aggrefact_test.jsonl' && echo "  aggrefact_test.jsonl: OK" || echo "  aggrefact_test.jsonl: MISSING!"
tar tzf director_ai_modernbert.tar.gz | grep -q 'tools/train_modernbert.py' && echo "  train_modernbert.py: OK" || echo "  train_modernbert.py: MISSING!"
tar tzf director_ai_modernbert.tar.gz | grep -q 'tools/eval_aggrefact.py' && echo "  eval_aggrefact.py: OK" || echo "  eval_aggrefact.py: MISSING!"
tar tzf director_ai_modernbert.tar.gz | grep -q 'gpu_deploy/modernbert_master.sh' && echo "  modernbert_master.sh: OK" || echo "  modernbert_master.sh: MISSING!"
tar tzf director_ai_modernbert.tar.gz | grep -q 'gpu_deploy/requirements_modernbert.txt' && echo "  requirements_modernbert.txt: OK" || echo "  requirements_modernbert.txt: MISSING!"

echo ""
echo "Upload: scp director_ai_modernbert.tar.gz root@<IP>:/root/"
