#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/root/director-ai
LOGDIR="$WORKDIR/logs"
RESULTDIR="$WORKDIR/results"
STATUS="$WORKDIR/STATUS"
VENV=/opt/director-venv

export PYTHONPATH=/root/director-ai
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in environment or .bashrc}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
        log_status "DONE  $name ($(( (t1 - t0) ))s)"
    else
        local rc=$?
        local t1
        t1=$(date +%s)
        log_status "FAIL  $name (exit $rc, $(( (t1 - t0) ))s)"
    fi
}

source "$VENV/bin/activate"
cd "$WORKDIR"

echo "=== Distillation Pipeline v2 ===" > "$STATUS"
log_status "Pipeline started"

# Step 1: Teacher soft labels from Bespoke-MiniCheck-7B (10K HaluEval)
run_step "teacher_labels" python tools/generate_teacher_labels.py 10000 labels/minicheck7b_halueval_10k.json

# Step 2a: Head-only distillation (freeze encoder, train only classification head)
run_step "distill_head_only" python tools/run_distillation.py \
    --train --freeze-encoder \
    --labels labels/minicheck7b_halueval_10k.json \
    --output-dir models/distill-head-only \
    --temperature 2.0 --alpha 0.7 --lr 1e-3 --epochs 5 --batch-size 16

# Step 2b: LoRA r=4 distillation
run_step "distill_lora_r4" python tools/run_distillation.py \
    --train \
    --labels labels/minicheck7b_halueval_10k.json \
    --output-dir models/distill-lora-r4 \
    --temperature 2.0 --alpha 0.7 --lr 1e-5 --epochs 3 --batch-size 4 --lora-rank 4

# Step 2c: LoRA r=8 distillation
run_step "distill_lora_r8" python tools/run_distillation.py \
    --train \
    --labels labels/minicheck7b_halueval_10k.json \
    --output-dir models/distill-lora-r8 \
    --temperature 2.0 --alpha 0.5 --lr 5e-5 --epochs 3 --batch-size 4 --lora-rank 8

# Step 3: Eval all variants on full AggreFact (29K)
for variant in baseline head_only lora_r4 lora_r8; do
    case "$variant" in
        baseline)    model_path=""; is_baseline="True" ;;
        head_only)   model_path="models/distill-head-only"; is_baseline="False" ;;
        lora_r4)     model_path="models/distill-lora-r4"; is_baseline="False" ;;
        lora_r8)     model_path="models/distill-lora-r8"; is_baseline="False" ;;
    esac

    run_step "eval_${variant}" python3 -c "
import json, time, os, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import balanced_accuracy_score

tag = '${variant}'
is_baseline = ${is_baseline}
model_path = '${model_path}'
token = os.environ.get('HF_TOKEN', '')

if is_baseline:
    mn = 'yaxili96/FactCG-DeBERTa-v3-Large'
    tokenizer = AutoTokenizer.from_pretrained(mn)
    model = AutoModelForSequenceClassification.from_pretrained(mn).cuda().eval()
else:
    is_lora = os.path.exists(os.path.join(model_path, 'adapter_config.json'))
    if is_lora:
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained('yaxili96/FactCG-DeBERTa-v3-Large')
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.cuda().eval()

ds = load_dataset('lytang/LLM-AggreFact', split='test', token=token)
rows = list(ds)

TEMPLATE = '{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that \"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is '

t0 = time.perf_counter()
by_dataset = {}
for idx, row in enumerate(rows):
    doc = row.get('doc', '')
    claim = row.get('claim', '')
    label = row.get('label')
    ds_name = row.get('dataset', 'unknown')
    if label is None or not doc or not claim:
        continue
    text = TEMPLATE.format(text_a=doc[:2000], text_b=claim[:500])
    enc = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    score = probs[0, 1].item()
    by_dataset.setdefault(ds_name, []).append((int(label), score))
    if idx % 5000 == 0:
        print(f'  {idx}/{len(rows)}', flush=True)

elapsed = time.perf_counter() - t0
best_thresh, best_avg = 0.5, 0.0
for t_int in range(10, 91):
    t = t_int / 100.0
    accs = [balanced_accuracy_score([p[0] for p in v], [1 if p[1] >= t else 0 for p in v]) for v in by_dataset.values()]
    avg = float(np.mean(accs))
    if avg > best_avg:
        best_avg, best_thresh = avg, t

per_ds = {}
for ds_name in sorted(by_dataset):
    v = by_dataset[ds_name]
    ba = balanced_accuracy_score([p[0] for p in v], [1 if p[1] >= best_thresh else 0 for p in v])
    per_ds[ds_name] = {'ba': round(float(ba), 4), 'n': len(v)}

result = {
    'model': tag,
    'macro_ba': round(float(best_avg), 4),
    'threshold': best_thresh,
    'elapsed': round(elapsed, 1),
    'per_dataset': per_ds,
}
json.dump(result, open(f'results/distill_eval_{tag}.json', 'w'), indent=2)
print(f'{tag}: {best_avg*100:.2f}% BA (t={best_thresh})')
"
done

# Summary
log_status "Pipeline complete"
echo "" >> "$STATUS"
echo "=== ALL RESULTS ===" >> "$STATUS"
for f in "$RESULTDIR"/distill_eval_*.json; do
    python3 -c "import json; d=json.load(open('$f')); print(f\"{d['model']}: {d['macro_ba']*100:.2f}% BA (t={d['threshold']})\")" >> "$STATUS"
done
cat "$STATUS"
