#!/usr/bin/env bash
# JarvisLabs: DocNLI (100K subset) + CB-lowLR
# SummaC skipped: mteb/summac dataset unavailable on Hub
# DeBERTa-v3-large batch size: 8 (24GB VRAM limit), grad_accum 4 (effective 32)
set -euo pipefail

WORKDIR=/home/user/director-ai
MODELS=$WORKDIR/models
SCORES=$WORKDIR/scores

echo "[$(date -u +%H:%M:%S)] === PRE-FLIGHT ==="
FREE_GB=$(df -BG / | awk 'NR==2{gsub("G",""); print $4}')
if [ "$FREE_GB" -lt 40 ]; then echo "ERROR: Only ${FREE_GB}GB free"; exit 1; fi
echo "  Disk: ${FREE_GB} GB free"
echo "  RAM: $(free -g | awk '/^Mem:/{print $7}') GB available"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f "$WORKDIR/benchmarks/aggrefact_test.jsonl" ]; then
    echo "ERROR: aggrefact_test.jsonl missing"; exit 1
fi
echo "  AggreFact: $(wc -l < $WORKDIR/benchmarks/aggrefact_test.jsonl) rows"

mkdir -p "$MODELS" "$SCORES"

bench_model() {
    local name="$1" path="$2" out="$SCORES/${name}.json"
    if [ -f "$out" ]; then echo "  [skip] $name already scored"; return; fi
    echo "[$(date -u +%H:%M:%S)] Benchmarking $name"
    python3.12 /home/user/director-ai/benchmarks/_bench_model.py "$name" "$path" "$WORKDIR" "$out"
}

# RUN 1: DocNLI (100K sample subset, 3 epochs)
echo "[$(date -u +%H:%M:%S)] === RUN 1/2: DocNLI (100K subset, 3 epochs) ==="
echo "  Expected: ~5h, ~\$2.50"
python3.12 tools/run_docnli_training.py
bench_model "factcg-docnli" "$MODELS/factcg-docnli"

# RUN 2: CommitmentBank at LR=5e-6 (catastrophic-forgetting test)
echo "[$(date -u +%H:%M:%S)] === RUN 2/2: CB-lowLR (LR=5e-6 vs original 1e-5) ==="
echo "  Tests: lower LR reduces catastrophic forgetting on small dataset"
python3.12 /home/user/director-ai/benchmarks/_cb_lowlr_train.py
bench_model "factcg-cb-lowlr" "$MODELS/factcg-cb-lowlr"

echo "[$(date -u +%H:%M:%S)] === ALL RUNS COMPLETE ==="
echo "Disk: $(df -BG / | awk 'NR==2{gsub("G",""); print $4}') GB free"

python3.12 - << 'PYEOF'
import json, glob, numpy as np

scores_dir = '/home/user/director-ai/scores'

def macro_ba(data, t):
    ds_bas = []
    for rows in data.values():
        labels = np.array([int(r[0]) for r in rows])
        scores = np.array([float(r[1]) for r in rows])
        if len(np.unique(labels)) < 2: continue
        preds = (scores >= t).astype(int)
        recalls = [(preds[labels==c]==c).mean() for c in np.unique(labels)]
        ds_bas.append(np.mean(recalls))
    return np.mean(ds_bas)

def best_t(data):
    best_ba, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        ba = macro_ba(data, t)
        if ba > best_ba:
            best_ba, best_thresh = ba, t
    return best_thresh, best_ba

upcloud_base = 0.7586
print('\nModel                         BestBA    Thresh   vs_base')
print('-'*60)
for f in sorted(glob.glob(f'{scores_dir}/*.json')):
    name = f.split('/')[-1].replace('.json','')
    data = json.load(open(f))
    t, ba = best_t(data)
    diff = (ba - upcloud_base)*100
    sign = ('+' if diff >= 0 else '') + f'{diff:.2f}'
    tag = '  BEATS BASE' if diff > 0.5 else ('  hurts' if diff < -0.5 else '')
    print(f'  {name:<28s}  {ba*100:.2f}%   {t:.2f}   {sign}%{tag}')
PYEOF

echo ""
echo "=== DOWNLOAD (run from local machine) ==="
echo "scp -P 11114 root@sshc.jarvislabs.ai:/home/user/director-ai/scores/factcg-docnli.json benchmarks/scores/"
echo "scp -P 11114 root@sshc.jarvislabs.ai:/home/user/director-ai/scores/factcg-cb-lowlr.json benchmarks/scores/"
echo "scp -r -P 11114 root@sshc.jarvislabs.ai:/home/user/director-ai/models/factcg-docnli/ benchmarks/models/"
echo "scp -r -P 11114 root@sshc.jarvislabs.ai:/home/user/director-ai/models/factcg-cb-lowlr/ benchmarks/models/"
echo ""
echo "Then DESTROY the instance to stop billing."
