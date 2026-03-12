#!/usr/bin/env bash
# Sequential follow-up domain training: ANLI → SciTail → PAWS
# Run inside tmux on the GPU instance after batch 1 (SNLI/MedNLI/Finance) completes.
#
# Usage:
#   tmux new-session -d -s followup "bash /home/director-ai/run_followup_training.sh 2>&1 | tee -a /home/director-ai/training_followup.log"
set -euo pipefail

cd /home/director-ai
source venv/bin/activate

echo "=== Follow-up training batch: ANLI → SciTail → PAWS ==="
echo "Started: $(date -u)"

# 1. ANLI (~170K binary samples, ~4-6h on L40S)
echo "[$(date -u)] Starting ANLI training..."
python run_anli_training.py 2>&1 | tee training_anli.log
echo "[$(date -u)] ANLI done. Packaging..."
tar czf factcg-anli.tar.gz -C models factcg-anli

# 2. SciTail (~27K samples, ~1h)
echo "[$(date -u)] Starting SciTail training..."
python run_scitail_training.py 2>&1 | tee training_scitail.log
echo "[$(date -u)] SciTail done. Packaging..."
tar czf factcg-scitail.tar.gz -C models factcg-scitail

# 3. PAWS (~49K samples, ~1.5h)
echo "[$(date -u)] Starting PAWS training..."
python run_paws_training.py 2>&1 | tee training_paws.log
echo "[$(date -u)] PAWS done. Packaging..."
tar czf factcg-paws.tar.gz -C models factcg-paws

echo ""
echo "============================================"
echo "  ALL FOLLOW-UP TRAINING COMPLETE"
echo "  $(date -u)"
echo "============================================"
echo ""
echo "Results:"
for m in anli scitail paws; do
    echo "--- factcg-$m ---"
    cat models/factcg-$m/training_result.json 2>/dev/null || echo "MISSING"
done
echo ""
echo "Download with:"
echo "  scp -i ~/.ssh/id_ed25519_upcloud root@\$(hostname -I | awk '{print \$1}'):/home/director-ai/factcg-{anli,scitail,paws}.tar.gz ."
