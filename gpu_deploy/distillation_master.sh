#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Distillation Master Runner (Phase 4A)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
#
# Runs on UpCloud L40S (48GB VRAM) or equivalent.
# Three distillation experiments + full AggreFact eval.
#
# Usage: bash gpu_deploy/distillation_master.sh
#
set -euo pipefail

WORKDIR="/root/director-ai"
LOGDIR="$WORKDIR/logs"
RESULTDIR="$WORKDIR/results"
STATUS="$WORKDIR/STATUS"
VENV="/opt/director-venv"

mkdir -p "$LOGDIR" "$RESULTDIR" "$WORKDIR/labels" "$WORKDIR/models"

log_status() {
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') | $1" >> "$STATUS"
}

run_step() {
    local name="$1"
    shift
    log_status "START $name"
    local t0
    t0=$(date +%s)
    if "$@" > "$LOGDIR/${name}.log" 2>&1; then
        local t1
        t1=$(date +%s)
        log_status "DONE  $name ($((($t1 - $t0) / 60))m)"
    else
        local rc=$?
        local t1
        t1=$(date +%s)
        log_status "FAIL  $name (exit $rc, $((($t1 - $t0) / 60))m)"
    fi
}

source "$VENV/bin/activate"
cd "$WORKDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Distillation Pipeline ===" > "$STATUS"
log_status "Pipeline started"

# ── Step 0: Install extra deps ────────────────────────────────────────
pip install peft==0.14.0 bitsandbytes==0.45.1 accelerate==1.3.0 > "$LOGDIR/pip_install.log" 2>&1

# ── Step 1: Generate teacher soft labels from Bespoke-MiniCheck-7B ────
# 7B model at FP16 needs ~14GB VRAM. L40S can handle it.
# Score HaluEval training set (10K QA pairs) for distillation.
run_step "teacher_labels" python tools/run_distillation.py \
    --generate-labels \
    --teacher bespoke-minicheck \
    --dataset halueval \
    --max-samples 10000 \
    --output labels/minicheck7b_halueval_10k.json

# ── Step 2a: Distillation — head-only (safest, no encoder modification) ─
run_step "distill_head_only" python tools/run_distillation.py \
    --train \
    --freeze-encoder \
    --labels labels/minicheck7b_halueval_10k.json \
    --output-dir models/distill-head-only \
    --temperature 2.0 \
    --alpha 0.7 \
    --lr 1e-3 \
    --epochs 5 \
    --batch-size 16

# ── Step 2b: Distillation — LoRA r=4 (conservative) ──────────────────
run_step "distill_lora_r4" python tools/run_distillation.py \
    --train \
    --labels labels/minicheck7b_halueval_10k.json \
    --output-dir models/distill-lora-r4 \
    --temperature 2.0 \
    --alpha 0.7 \
    --lr 1e-5 \
    --epochs 3 \
    --batch-size 4 \
    --lora-rank 4

# ── Step 2c: Distillation — LoRA r=8 (as planned) ────────────────────
run_step "distill_lora_r8" python tools/run_distillation.py \
    --train \
    --labels labels/minicheck7b_halueval_10k.json \
    --output-dir models/distill-lora-r8 \
    --temperature 2.0 \
    --alpha 0.5 \
    --lr 5e-5 \
    --epochs 3 \
    --batch-size 4 \
    --lora-rank 8

# ── Step 3: Evaluate all variants on full AggreFact (29K) ────────────

# 3a: Baseline (unmodified FactCG)
run_step "eval_baseline" python -c "
import json, time, os, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import balanced_accuracy_score

model_name = 'yaxili96/FactCG-DeBERTa-v3-Large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda().eval()

ds = load_dataset('lytang/LLM-AggreFact', split='test')
rows = list(ds)

TEMPLATE = '{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that \"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is '

def score_batch(pairs, batch_size=32):
    texts = [TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, 1].cpu().numpy().tolist())
    return scores

t0 = time.perf_counter()
by_dataset = {}
for idx, row in enumerate(rows):
    doc, claim, label, ds_name = row.get('doc',''), row.get('claim',''), row.get('label'), row.get('dataset','unknown')
    if label is None or not doc or not claim: continue
    score = score_batch([(doc, claim)])[0]
    by_dataset.setdefault(ds_name, []).append((int(label), score))
    if idx % 5000 == 0: print(f'  {idx}/{len(rows)}', flush=True)

elapsed = time.perf_counter() - t0
best_thresh, best_avg = 0.5, 0.0
for t_int in range(10, 91):
    t = t_int / 100.0
    accs = [balanced_accuracy_score([p[0] for p in v], [1 if p[1]>=t else 0 for p in v]) for v in by_dataset.values()]
    avg = float(np.mean(accs))
    if avg > best_avg: best_avg, best_thresh = avg, t

per_ds = {}
for ds_name in sorted(by_dataset):
    v = by_dataset[ds_name]
    ba = balanced_accuracy_score([p[0] for p in v], [1 if p[1]>=best_thresh else 0 for p in v])
    per_ds[ds_name] = {'ba': float(ba), 'n': len(v)}

json.dump({'model': 'baseline', 'macro_ba': float(best_avg), 'threshold': best_thresh, 'elapsed': elapsed, 'per_dataset': per_ds}, open('results/distill_eval_baseline.json','w'), indent=2)
print(f'Baseline: {best_avg*100:.2f}% BA (t={best_thresh})')
"

# Evaluation function for distilled models
eval_model() {
    local model_dir="$1"
    local tag="$2"
    run_step "eval_${tag}" python -c "
import json, time, os, sys, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import balanced_accuracy_score

model_dir = '$model_dir'
tag = '$tag'

# Check if it's a LoRA model (has adapter_config.json)
is_lora = os.path.exists(os.path.join(model_dir, 'adapter_config.json'))
if is_lora:
    from peft import PeftModel
    base = AutoModelForSequenceClassification.from_pretrained('yaxili96/FactCG-DeBERTa-v3-Large')
    model = PeftModel.from_pretrained(base, model_dir)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

model = model.cuda().eval()

# Mark as factcg for template detection
model.config.factcg = True

ds = load_dataset('lytang/LLM-AggreFact', split='test')
rows = list(ds)

TEMPLATE = '{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that \"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is '

def score_batch(pairs, batch_size=32):
    texts = [TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, 1].cpu().numpy().tolist())
    return scores

t0 = time.perf_counter()
by_dataset = {}
for idx, row in enumerate(rows):
    doc, claim, label, ds_name = row.get('doc',''), row.get('claim',''), row.get('label'), row.get('dataset','unknown')
    if label is None or not doc or not claim: continue
    score = score_batch([(doc, claim)])[0]
    by_dataset.setdefault(ds_name, []).append((int(label), score))
    if idx % 5000 == 0: print(f'  {idx}/{len(rows)}', flush=True)

elapsed = time.perf_counter() - t0
best_thresh, best_avg = 0.5, 0.0
for t_int in range(10, 91):
    t = t_int / 100.0
    accs = [balanced_accuracy_score([p[0] for p in v], [1 if p[1]>=t else 0 for p in v]) for v in by_dataset.values()]
    avg = float(np.mean(accs))
    if avg > best_avg: best_avg, best_thresh = avg, t

per_ds = {}
for ds_name in sorted(by_dataset):
    v = by_dataset[ds_name]
    ba = balanced_accuracy_score([p[0] for p in v], [1 if p[1]>=best_thresh else 0 for p in v])
    per_ds[ds_name] = {'ba': float(ba), 'n': len(v)}

json.dump({'model': tag, 'macro_ba': float(best_avg), 'threshold': best_thresh, 'elapsed': elapsed, 'per_dataset': per_ds}, open(f'results/distill_eval_{tag}.json','w'), indent=2)
print(f'{tag}: {best_avg*100:.2f}% BA (t={best_thresh})')
"
}

# 3b-d: Evaluate each distilled variant
eval_model "models/distill-head-only" "head_only"
eval_model "models/distill-lora-r4" "lora_r4"
eval_model "models/distill-lora-r8" "lora_r8"

# ── Step 4: Summary ──────────────────────────────────────────────────
log_status "Pipeline complete"

echo "" >> "$STATUS"
echo "=== ALL RESULTS ===" >> "$STATUS"
for f in results/distill_eval_*.json; do
    tag=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['model']}: {d['macro_ba']*100:.2f}% BA (t={d['threshold']})\")")
    echo "$tag" >> "$STATUS"
done

cat "$STATUS"
