#!/usr/bin/env bash
# UpCloud Batch 2: MultiNLI (after SNLI+MedNLI+Finance complete)
# Run on L40S 48GB. Est. ~10h for 433K samples.
# IMPORTANT: Download batch 1 models BEFORE starting this.

set -euo pipefail
cd /home/director-ai

echo "=== UPCLOUD BATCH 2 START: $(date -u) ==="

# Verify batch 1 results exist (safety check)
for m in factcg-snli factcg-mednli factcg-finance; do
    if [ ! -f "models/$m/training_result.json" ]; then
        echo "ERROR: $m not complete — do NOT start batch 2 yet"
        exit 1
    fi
    echo "OK: $m complete"
done

# MultiNLI (~433K binary, ~10h)
echo ">>> MultiNLI starting: $(date -u)"
python run_multinli_training.py > training_multinli.log 2>&1
echo ">>> MultiNLI done: $(date -u)"
tar czf factcg-multinli.tar.gz -C models factcg-multinli
echo ">>> Packaged factcg-multinli.tar.gz"

echo "=== UPCLOUD BATCH 2 COMPLETE: $(date -u) ==="
ls -lh /home/director-ai/factcg-*.tar.gz
