#!/usr/bin/env bash
# JarvisLabs run: DocNLI + SummaC + CB-lowLR fine-tuning on FactCG-DeBERTa-v3-Large
#
# INSTANCE REQUIREMENTS (mandatory — do not cut corners):
#   GPU:     RTX 5000 Ada (16 GB VRAM) minimum.  A100 preferred.
#   Storage: 80 GB (models ~1.7 GB each, HF cache ~2 GB, JSONL 86 MB, venv ~4 GB)
#   RAM:     32 GB system (DeBERTa-v3-large peaks ~18 GB resident during training)
#
# BEFORE RUNNING — transfer AggreFact test data from UpCloud to this instance:
#   scp root@212.147.240.110:/home/director-ai/benchmarks/aggrefact_test.jsonl \
#       /home/user/director-ai/benchmarks/
#
# THEN run in tmux:
#   tmux new-session -d -s train
#   tmux send-keys -t train "cd /home/user/director-ai && bash benchmarks/jarvislabs_run_large.sh 2>&1 | tee /home/user/run.log" Enter
#
# EXPECTED RUNS (priority order):
#   1. factcg-docnli  — DocNLI 900K doc-level NLI,  most relevant for AggreFact
#   2. factcg-summac  — SummaC summarization consistency, directly on-domain
#   3. factcg-cb-lowlr — CB re-run at LR=5e-6 (vs 1e-5 in original) to test forgetting

set -euo pipefail

WORKDIR=/home/user/director-ai
MODELS=$WORKDIR/models
SCORES=$WORKDIR/scores

# ── 0. Headroom checks ───────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] === PRE-FLIGHT CHECKS ==="

FREE_GB=$(df -BG / | awk 'NR==2{gsub("G",""); print $4}')
if [ "$FREE_GB" -lt 40 ]; then
    echo "ERROR: Only ${FREE_GB} GB free. Need 40 GB minimum. Abort." >&2
    exit 1
fi
echo "  Disk: ${FREE_GB} GB free — OK"

FREE_RAM_GB=$(free -g | awk '/^Mem:/{print $7}')
if [ "$FREE_RAM_GB" -lt 20 ]; then
    echo "WARNING: Only ${FREE_RAM_GB} GB free RAM. May OOM on DocNLI."
fi
echo "  RAM: ${FREE_RAM_GB} GB available"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  GPU: check failed (non-fatal)"

# ── 1. Verify data is present ─────────────────────────────────────────
JSONL=$WORKDIR/benchmarks/aggrefact_test.jsonl
if [ ! -f "$JSONL" ]; then
    echo "ERROR: $JSONL not found."
    echo "Transfer it first: scp root@212.147.240.110:/home/director-ai/benchmarks/aggrefact_test.jsonl $JSONL"
    exit 1
fi
echo "  AggreFact data: $(wc -l < "$JSONL") rows — OK"

# ── 2. Install deps ────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] Installing dependencies"
cd "$WORKDIR"
pip install -e ".[finetune,nli]" --quiet 2>&1 | tail -3
echo "  Disk after install: $(df -BG / | awk 'NR==2{gsub("G",""); print $4}') GB free"

# ── 3. Benchmark helper ─────────────────────────────────────────────────
# Reuses existing run_ensemble_seq.py infrastructure.
# Accepts a single model name — skips already-scored models.
bench_model() {
    local name="$1"
    local path="$2"
    local out="$SCORES/${name}.json"

    if [ -f "$out" ]; then
        echo "  [skip] $name already scored"
        return
    fi

    echo "[$(date -u +%H:%M:%S)] Benchmarking $name"
    python3 - << PYEOF
import json, sys, time, logging
sys.path.insert(0, '$WORKDIR')
from benchmarks._load_aggrefact_patch import _load_aggrefact_local
import benchmarks.aggrefact_eval as ae
ae._load_aggrefact = _load_aggrefact_local
from benchmarks.aggrefact_eval import _BinaryNLIPredictor
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
rows = _load_aggrefact_local()

pred = _BinaryNLIPredictor(model_name='$path')
by_ds = {}
t0 = time.perf_counter()
for i, row in enumerate(rows):
    doc = row.get('doc', '')
    claim = row.get('claim', '')
    lbl = row.get('label')
    ds = row.get('dataset', 'unknown')
    if lbl is None or not doc or not claim:
        continue
    prob = pred.score(doc, claim)
    by_ds.setdefault(ds, []).append((int(lbl), float(prob)))
    if (i+1) % 2000 == 0:
        elapsed = time.perf_counter() - t0
        eta = (len(rows)-i-1)*elapsed/(i+1)/60
        logging.info('  $name: %d/%d (%.0f min remaining)', i+1, len(rows), eta)

import pathlib
pathlib.Path('$SCORES').mkdir(exist_ok=True)
pathlib.Path('$out').write_text(json.dumps(by_ds))
elapsed = time.perf_counter() - t0
logging.info('Saved $name (%.1f min)', elapsed/60)
del pred.model, pred
torch.cuda.empty_cache()
PYEOF
}

mkdir -p "$MODELS" "$SCORES"

# ── 4. RUN 1: DocNLI ────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] === RUN 1/3: DocNLI (900K doc-level NLI) ==="
echo "  Most relevant for AggreFact: document-hypothesis pairs from summarization + QA"
echo "  Disk before: $(df -BG / | awk 'NR==2{gsub("G",""); print $4}') GB free"

python3 tools/run_docnli_training.py
echo "  Disk after DocNLI train: $(df -BG / | awk 'NR==2{gsub("G",""); print $4}') GB free"

bench_model "factcg-docnli" "$MODELS/factcg-docnli"
echo "  Disk after DocNLI bench: $(df -BG / | awk 'NR==2{gsub("G",""); print $4}') GB free"

# ── 5. RUN 2: SummaC ────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] === RUN 2/3: SummaC (summarization consistency) ==="
echo "  Disk before: $(df -BG / | awk 'NR==2{gsub("G",""); print $4}') GB free"

python3 tools/run_summac_training.py
echo "  Disk after SummaC train: $(df -BG / | awk 'NR==2{gsub("G",""); print $4}') GB free"

bench_model "factcg-summac" "$MODELS/factcg-summac"

# ── 6. RUN 3: CB at LR=5e-6 (catastrophic-forgetting experiment) ────────
echo "[$(date -u +%H:%M:%S)] === RUN 3/3: CB at LR=5e-6 vs original 1e-5 ==="
echo "  Testing: lower LR → less catastrophic forgetting → better AggreFact transfer"

python3 - << 'PYEOF'
import json, os, time, numpy as np
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments,
)

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
OUTPUT_DIR = "/home/user/director-ai/models/factcg-cb-lowlr"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

ds_raw = load_dataset("super_glue", "cb")

def convert(ex):
    label = 1 if ex["label"] == 0 else 0  # entailment=1, other=0
    text = TEMPLATE.format(text_a=ex["premise"], text_b=ex["hypothesis"])
    return {"text": text, "label": label}

train = ds_raw["train"].map(convert, remove_columns=ds_raw["train"].column_names)
val_split = ds_raw["validation"].train_test_split(test_size=0.5, seed=42)
val = val_split["train"].map(convert, remove_columns=val_split["train"].column_names)
test = val_split["test"].map(convert, remove_columns=val_split["test"].column_names)
print(f"CB: train={len(train)}, val={len(val)}, test={len(test)}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

def tok_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512, padding=False)

for split in [train, val, test]:
    split = split.map(tok_fn, batched=True, remove_columns=["text"])

train = train.map(tok_fn, batched=True, remove_columns=["text"])
val = val.map(tok_fn, batched=True, remove_columns=["text"])
test = test.map(tok_fn, batched=True, remove_columns=["text"])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    learning_rate=5e-6,         # key difference vs original 1e-5
    weight_decay=0.01,
    warmup_ratio=0.2,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="balanced_accuracy",
    greater_is_better=True,
    fp16=True,
    logging_steps=10,
    report_to="none",
)

t0 = time.time()
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                  tokenizer=tokenizer, compute_metrics=compute_metrics)
trainer.train()
test_result = trainer.evaluate(test)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

result = {
    "dataset": "cb_superglue",
    "base_model": BASE_MODEL,
    "learning_rate": 5e-6,
    "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
    "training_time_minutes": round((time.time()-t0)/60, 1),
}
with open(os.path.join(OUTPUT_DIR, "training_result.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"CB-lowLR COMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")
PYEOF

bench_model "factcg-cb-lowlr" "$MODELS/factcg-cb-lowlr"

# ── 7. Summary ────────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] === ALL RUNS COMPLETE ==="
echo "Disk final: $(df -BG / | awk 'NR==2{gsub("G",""); print $4}') GB free"
echo "RAM final: $(free -g | awk '/^Mem:/{print $7}') GB available"

source /home/user/director-ai/venv/bin/activate 2>/dev/null || true
python3 - << 'PYEOF'
import json, glob, numpy as np

scores_dir = '/home/user/director-ai/scores'

def macro_ba(data, t):
    ds_bas = []
    for ds_name, rows in data.items():
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

results = {}
for f in sorted(glob.glob(f'{scores_dir}/*.json')):
    name = f.split('/')[-1].replace('.json','')
    data = json.load(open(f))
    t, ba = best_t(data)
    results[name] = (ba, t)

upcloud_base = 0.7586  # reference from UpCloud run
print('\nModel                         BestBA    Thresh   vs_upcloud_base')
print('-'*65)
for k,(ba,t) in sorted(results.items(), key=lambda x:-x[1][0]):
    diff = (ba - upcloud_base)*100
    sign = ('+' if diff >= 0 else '') + f'{diff:.2f}'
    tag = '  BEATS' if diff > 0.5 else ('  hurts' if diff < -0.5 else '')
    print(f'  {k:<28s}  {ba*100:.2f}%   {t:.2f}   {sign}%{tag}')
PYEOF

echo ""
echo "=== DOWNLOAD COMMANDS (run from local machine) ==="
echo "scp <user>@<host>:/home/user/director-ai/scores/factcg-docnli.json benchmarks/scores/"
echo "scp <user>@<host>:/home/user/director-ai/scores/factcg-summac.json benchmarks/scores/"
echo "scp <user>@<host>:/home/user/director-ai/scores/factcg-cb-lowlr.json benchmarks/scores/"
echo "scp -r <user>@<host>:/home/user/director-ai/models/factcg-docnli/ benchmarks/models/"
echo "scp -r <user>@<host>:/home/user/director-ai/models/factcg-summac/ benchmarks/models/"
echo "scp -r <user>@<host>:/home/user/director-ai/models/factcg-cb-lowlr/ benchmarks/models/"
echo ""
echo "Then DESTROY the instance to stop billing."
