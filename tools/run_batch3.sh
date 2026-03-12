#!/usr/bin/env bash
# Batch 3: DialogueNLI → HealthVer → ClimateFEVER → WANLI (JarvisLabs)
# Sequential pipeline with tar packaging after each model.
# Est. total: ~6h on RTX 6000 24GB

set -euo pipefail
cd /home/director-ai

echo "=== BATCH 3 START: $(date -u) ==="

# 1. DialogueNLI (~31K, ~1h)
echo ">>> DialogueNLI starting: $(date -u)"
python run_dialogue_nli_training.py > training_dialogue_nli.log 2>&1
echo ">>> DialogueNLI done: $(date -u)"
tar czf factcg-dialogue-nli.tar.gz -C models factcg-dialogue-nli
echo ">>> Packaged factcg-dialogue-nli.tar.gz"

# 2. HealthVer/PUBHEALTH (~12K, ~30min)
echo ">>> HealthVer starting: $(date -u)"
python run_healthver_training.py > training_healthver.log 2>&1
echo ">>> HealthVer done: $(date -u)"
tar czf factcg-healthver.tar.gz -C models factcg-healthver
echo ">>> Packaged factcg-healthver.tar.gz"

# 3. ClimateFEVER (~1.5K, ~10min)
echo ">>> ClimateFEVER starting: $(date -u)"
python run_climatefever_training.py > training_climatefever.log 2>&1
echo ">>> ClimateFEVER done: $(date -u)"
tar czf factcg-climatefever.tar.gz -C models factcg-climatefever
echo ">>> Packaged factcg-climatefever.tar.gz"

# 4. WANLI (~108K, ~4h)
echo ">>> WANLI starting: $(date -u)"
python run_wanli_training.py > training_wanli.log 2>&1
echo ">>> WANLI done: $(date -u)"
tar czf factcg-wanli.tar.gz -C models factcg-wanli
echo ">>> Packaged factcg-wanli.tar.gz"

echo "=== BATCH 3 COMPLETE: $(date -u) ==="
echo "Models ready for download:"
ls -lh /home/director-ai/factcg-*.tar.gz
