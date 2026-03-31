#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — ModernBERT Autonomous Training Pipeline
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
#
# Autonomous pipeline: probe checkpoint → train → eval at 3 lengths.
# Designed for UpCloud L40S (48GB VRAM).
#
# Usage: nohup bash gpu_deploy/modernbert_master.sh > logs/master.log 2>&1 &
# Monitor: cat STATUS | tail -20
#
set -o pipefail

WORKDIR="/root/director-ai"
LOGDIR="$WORKDIR/logs"
RESULTDIR="$WORKDIR/results"
STATUS="$WORKDIR/STATUS"
VENV="/opt/director-venv"

mkdir -p "$LOGDIR" "$RESULTDIR" "$WORKDIR/models"

log_status() {
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') | $1" >> "$STATUS"
}

run_phase() {
    local name="$1"
    shift
    log_status "START $name"
    local t0
    t0=$(date +%s)
    if "$@" > "$LOGDIR/${name}.log" 2>&1; then
        local t1
        t1=$(date +%s)
        log_status "DONE  $name ($((($t1 - $t0) / 60))m)"
        return 0
    else
        local rc=$?
        local t1
        t1=$(date +%s)
        log_status "FAIL  $name (exit $rc, $((($t1 - $t0) / 60))m)"
        return $rc
    fi
}

source "$VENV/bin/activate"
cd "$WORKDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== ModernBERT Pipeline ===" > "$STATUS"
log_status "Pipeline started"

# ── Phase 0: Environment verification ──────────────────────────────
run_phase "env_check" python3 -c "
import torch, transformers, datasets, sklearn
print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
print(f'transformers {transformers.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'bf16: {torch.cuda.is_bf16_supported()}')
v = transformers.__version__
major, minor = int(v.split('.')[0]), int(v.split('.')[1])
assert major >= 5 or (major == 4 and minor >= 48), \
    f'transformers >= 4.48 required, got {v}'
import os
for p in ('data/aggrefact_test.jsonl', 'benchmarks/aggrefact_test.jsonl'):
    if os.path.exists(p):
        n = sum(1 for _ in open(p))
        assert n > 25000, f'AggreFact incomplete: {n}'
        print(f'AggreFact: {n} samples ({p})')
        break
else:
    raise FileNotFoundError('aggrefact_test.jsonl missing')
print('All checks passed')
"

# ── Phase 1: Probe pre-trained NLI checkpoint ─────────────────────
NLI_CHECKPOINT="MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
USE_PRETRAINED=false
SKIP_FLAG=""

run_phase "probe_checkpoint" python3 -c "
from transformers import AutoConfig, AutoTokenizer
import json

try:
    cfg = AutoConfig.from_pretrained('$NLI_CHECKPOINT')
    tok = AutoTokenizer.from_pretrained('$NLI_CHECKPOINT')
    num_labels = getattr(cfg, 'num_labels', None)
    id2label = getattr(cfg, 'id2label', {})
    result = {
        'checkpoint': '$NLI_CHECKPOINT',
        'num_labels': num_labels,
        'id2label': {str(k): v for k, v in id2label.items()},
        'usable': num_labels in (2, 3),
    }
    json.dump(result, open('$RESULTDIR/checkpoint_probe.json', 'w'), indent=2)
    print(f'Checkpoint: num_labels={num_labels}, id2label={id2label}')
    if num_labels in (2, 3):
        print('USABLE: will skip Stage 1')
    else:
        print('NOT USABLE: will run full Stage 1')
except Exception as e:
    json.dump({'checkpoint': '$NLI_CHECKPOINT', 'error': str(e), 'usable': False},
              open('$RESULTDIR/checkpoint_probe.json', 'w'), indent=2)
    print(f'Probe failed: {e}')
    print('Will run full Stage 1')
" || true

if [ -f "$RESULTDIR/checkpoint_probe.json" ]; then
    usable=$(python3 -c "import json; print(json.load(open('$RESULTDIR/checkpoint_probe.json')).get('usable', False))")
    if [ "$usable" = "True" ]; then
        USE_PRETRAINED=true
        SKIP_FLAG="--skip-stage1"
        log_status "Using pre-trained checkpoint: $NLI_CHECKPOINT (skipping Stage 1)"
    else
        log_status "No usable checkpoint — will run full Stage 1"
    fi
fi

# ── Phase 2: Baseline FactCG-DeBERTa eval ─────────────────────────
run_phase "eval_baseline" python3 tools/eval_aggrefact.py \
    baseline "" --max-length 512 --batch-size 32

# ── Phase 3: Train ModernBERT ──────────────────────────────────────
if [ "$USE_PRETRAINED" = true ]; then
    BASE_MODEL="$NLI_CHECKPOINT"
    log_status "Skipping Stage 1 (using $NLI_CHECKPOINT)"
else
    run_phase "stage1_nli" python3 tools/train_modernbert.py --stage 1 \
        --output-dir models/modernbert-nli-stage1 \
        --max-length 2048 \
        --lr 2e-5 \
        --epochs 3 \
        --batch-size 8 \
        --grad-accum 4 \
        --gradient-checkpointing
    BASE_MODEL="models/modernbert-nli-stage1"
fi

run_phase "stage2_halueval" python3 tools/train_modernbert.py --stage 2 \
    --base-model "$BASE_MODEL" \
    $SKIP_FLAG \
    --output-dir models/modernbert-factcg \
    --max-length 4096 \
    --lr 5e-6 \
    --epochs 5 \
    --batch-size 4 \
    --grad-accum 8 \
    --gradient-checkpointing

# ── Phase 4-6: Multi-length evaluation ────────────────────────────
run_phase "eval_512" python3 tools/eval_aggrefact.py \
    modernbert_512 models/modernbert-factcg --max-length 512 --batch-size 16

run_phase "eval_2048" python3 tools/eval_aggrefact.py \
    modernbert_2048 models/modernbert-factcg --max-length 2048 --batch-size 8

run_phase "eval_4096" python3 tools/eval_aggrefact.py \
    modernbert_4096 models/modernbert-factcg --max-length 4096 --batch-size 4

# ── Phase 7: Package results ──────────────────────────────────────
log_status "Packaging results"

tar czf "$WORKDIR/results_package.tar.gz" \
    --exclude='*.tar.gz' \
    --exclude='checkpoint-*' \
    -C "$WORKDIR" \
    results/ \
    models/modernbert-factcg/ \
    logs/ \
    STATUS

PACKAGE_SIZE=$(du -h "$WORKDIR/results_package.tar.gz" | cut -f1)
log_status "Package created: $PACKAGE_SIZE"

# ── Summary ────────────────────────────────────────────────────────
log_status "Pipeline complete"

echo "" >> "$STATUS"
echo "=== ALL RESULTS ===" >> "$STATUS"
for f in "$RESULTDIR"/eval_*.json "$RESULTDIR"/checkpoint_probe.json; do
    [ -f "$f" ] && python3 -c "
import json
d = json.load(open('$f'))
if 'macro_ba' in d:
    print(f\"{d['model']}: {d['macro_ba']*100:.2f}% BA (t={d['threshold']}, len={d.get('max_length','?')})\")
elif 'usable' in d:
    print(f\"checkpoint: usable={d['usable']}, num_labels={d.get('num_labels','?')}\")
" >> "$STATUS" 2>/dev/null || true
done

echo "" >> "$STATUS"
echo "Download: scp root@\$(hostname -I | awk '{print \$1}'):/root/director-ai/results_package.tar.gz ." >> "$STATUS"

cat "$STATUS"
