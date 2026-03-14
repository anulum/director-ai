#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Accuracy Improvement GPU Master Runner
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
# Autonomous pipeline: Phases 1A → 6A-feat → 3A (×4) → 4A → 6A-train
# Run inside screen: screen -S train bash gpu_deploy/master_runner.sh
#
# Survives SSH disconnect. All results saved after each phase.
# STATUS file tracks progress. RESULTS_SUMMARY.md has all numbers.
# ─────────────────────────────────────────────────────────────────────
set -o pipefail

PROJ_ROOT="/root/director-ai"
cd "$PROJ_ROOT"

source /opt/director-venv/bin/activate
export PYTHONPATH="$PROJ_ROOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p models features labels results logs benchmarks/results

STATUS="$PROJ_ROOT/STATUS"
SUMMARY="$PROJ_ROOT/results/RESULTS_SUMMARY.md"

# ── Helpers ───────────────────────────────────────────────────────

ts() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

log_status() {
    echo "$(ts) | $1" | tee -a "$STATUS"
}

run_phase() {
    local name="$1"
    shift
    local logfile="logs/${name}.log"

    log_status "START $name"
    echo "" >> "$SUMMARY"
    echo "### $name" >> "$SUMMARY"
    echo "Started: $(ts)" >> "$SUMMARY"

    local t0=$(date +%s)
    if "$@" > "$logfile" 2>&1; then
        local rc=0
    else
        local rc=$?
    fi
    local elapsed=$(( $(date +%s) - t0 ))
    local mins=$(( elapsed / 60 ))

    if [ "$rc" -eq 0 ]; then
        log_status "DONE  $name (${mins}m)"
        echo "Result: PASS (${mins}m)" >> "$SUMMARY"
    else
        log_status "FAIL  $name (exit $rc, ${mins}m)"
        echo "Result: FAIL (exit $rc, ${mins}m)" >> "$SUMMARY"
        echo "Last 20 lines:" >> "$SUMMARY"
        echo '```' >> "$SUMMARY"
        tail -20 "$logfile" >> "$SUMMARY" 2>/dev/null
        echo '```' >> "$SUMMARY"
    fi

    # Checkpoint: tar current results after each phase
    # --exclude prevents recursive inclusion of previous checkpoint tarballs
    tar czf "results/checkpoint_${name}.tar.gz" \
        --exclude='*.tar.gz' \
        results/ logs/ STATUS 2>/dev/null || true

    return 0
}

extract_ba() {
    # Extract macro-avg BA from an aggrefact JSON result file
    local f="$1"
    if [ -f "$f" ]; then
        python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
ba = d.get('avg_balanced_accuracy_pct', d.get('avg_balanced_accuracy', '?'))
print(f'  Macro-avg BA: {ba}%')
" 2>/dev/null || echo "  (could not parse)"
    fi
}

# ── Initialize ────────────────────────────────────────────────────

echo "# Director-AI Accuracy Improvement Results" > "$SUMMARY"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)" >> "$SUMMARY"
echo "Started: $(ts)" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "## Baseline" >> "$SUMMARY"

log_status "PIPELINE START"

# ── PHASE 0: Baseline eval (t=0.50, global) ──────────────────────

echo "## Phase 0: Baseline" >> "$SUMMARY"

run_phase "phase0_baseline" \
    python -m benchmarks.aggrefact_eval --sweep

# Find the baseline result file
BASELINE_FILE=$(ls -t benchmarks/results/aggrefact_*.json 2>/dev/null | head -1)
if [ -n "$BASELINE_FILE" ]; then
    cp "$BASELINE_FILE" results/phase0_baseline.json
    extract_ba results/phase0_baseline.json >> "$SUMMARY"
fi

# ── PHASE 1A: Threshold Analysis ─────────────────────────────────

echo "" >> "$SUMMARY"
echo "## Phase 1A: Threshold Analysis" >> "$SUMMARY"

run_phase "phase1a_threshold" \
    python -m benchmarks.threshold_analysis --compare

if [ -f benchmarks/results/threshold_analysis.json ]; then
    cp benchmarks/results/threshold_analysis.json results/phase1a_threshold.json
    python3 -c "
import json
with open('results/phase1a_threshold.json') as f:
    d = json.load(f)
for mode in ['global', 'per_dataset', 'task_type']:
    if mode in d:
        print(f'  {mode}: {d[mode].get(\"macro_avg_ba\", \"?\")}% BA')
" >> "$SUMMARY" 2>/dev/null
fi

# ── PHASE 6A Step 1: Meta-Classifier Feature Generation ──────────

echo "" >> "$SUMMARY"
echo "## Phase 6A-feat: Meta-Classifier Features" >> "$SUMMARY"

run_phase "phase6a_features" \
    python tools/train_meta_classifier.py --generate-features \
        --output features/aggrefact_meta_features.json

if [ -f features/aggrefact_meta_features.json ]; then
    N=$(python3 -c "import json; print(len(json.load(open('features/aggrefact_meta_features.json'))))" 2>/dev/null || echo "?")
    echo "  Features extracted: $N samples" >> "$SUMMARY"
fi

# ── PHASE 3A: LoRA Experiments ────────────────────────────────────

echo "" >> "$SUMMARY"
echo "## Phase 3A: LoRA Fine-Tuning" >> "$SUMMARY"

# --- Exp 1: HaluEval, rank=8, lr=1e-4 ---
run_phase "phase3a_exp1_train" \
    python tools/run_lora_training.py \
        --dataset halueval \
        --output-dir models/lora-halueval-r8-lr1e4 \
        --rank 8 --alpha 16 --lr 1e-4 --epochs 5 \
        --batch-size 16 --grad-accum 2

run_phase "phase3a_exp1_merge" \
    python gpu_deploy/merge_lora.py \
        --adapter models/lora-halueval-r8-lr1e4 \
        --output models/merged-halueval-r8-lr1e4

run_phase "phase3a_exp1_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/merged-halueval-r8-lr1e4

EVAL1=$(ls -t benchmarks/results/aggrefact_merged-halueval-r8-lr1e4*.json 2>/dev/null | head -1)
[ -n "$EVAL1" ] && cp "$EVAL1" results/phase3a_exp1.json && extract_ba results/phase3a_exp1.json >> "$SUMMARY"

# --- Exp 2: HaluEval, rank=8, lr=5e-5 (lower LR) ---
run_phase "phase3a_exp2_train" \
    python tools/run_lora_training.py \
        --dataset halueval \
        --output-dir models/lora-halueval-r8-lr5e5 \
        --rank 8 --alpha 16 --lr 5e-5 --epochs 5 \
        --batch-size 16 --grad-accum 2

run_phase "phase3a_exp2_merge" \
    python gpu_deploy/merge_lora.py \
        --adapter models/lora-halueval-r8-lr5e5 \
        --output models/merged-halueval-r8-lr5e5

run_phase "phase3a_exp2_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/merged-halueval-r8-lr5e5

EVAL2=$(ls -t benchmarks/results/aggrefact_merged-halueval-r8-lr5e5*.json 2>/dev/null | head -1)
[ -n "$EVAL2" ] && cp "$EVAL2" results/phase3a_exp2.json && extract_ba results/phase3a_exp2.json >> "$SUMMARY"

# --- Exp 3: HaluEval + Sieving ---
run_phase "phase3a_exp3_train" \
    python tools/run_lora_training.py \
        --dataset halueval \
        --output-dir models/lora-halueval-sieved \
        --rank 8 --alpha 16 --lr 1e-4 --epochs 5 \
        --batch-size 16 --grad-accum 2 \
        --sieving --sieve-rate 0.1

run_phase "phase3a_exp3_merge" \
    python gpu_deploy/merge_lora.py \
        --adapter models/lora-halueval-sieved \
        --output models/merged-halueval-sieved

run_phase "phase3a_exp3_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/merged-halueval-sieved

EVAL3=$(ls -t benchmarks/results/aggrefact_merged-halueval-sieved*.json 2>/dev/null | head -1)
[ -n "$EVAL3" ] && cp "$EVAL3" results/phase3a_exp3.json && extract_ba results/phase3a_exp3.json >> "$SUMMARY"

# --- Exp 4: VitaminC 50K subset ---
run_phase "phase3a_exp4_train" \
    python tools/run_lora_training.py \
        --dataset vitaminc \
        --max-samples 50000 \
        --output-dir models/lora-vitaminc-50k \
        --rank 8 --alpha 16 --lr 1e-4 --epochs 3 \
        --batch-size 16 --grad-accum 2

run_phase "phase3a_exp4_merge" \
    python gpu_deploy/merge_lora.py \
        --adapter models/lora-vitaminc-50k \
        --output models/merged-vitaminc-50k

run_phase "phase3a_exp4_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/merged-vitaminc-50k

EVAL4=$(ls -t benchmarks/results/aggrefact_merged-vitaminc-50k*.json 2>/dev/null | head -1)
[ -n "$EVAL4" ] && cp "$EVAL4" results/phase3a_exp4.json && extract_ba results/phase3a_exp4.json >> "$SUMMARY"

# ── PHASE 4A: Distillation from 7B Teacher ───────────────────────

echo "" >> "$SUMMARY"
echo "## Phase 4A: Knowledge Distillation" >> "$SUMMARY"

# Step 1: Generate teacher soft labels (Bespoke-MiniCheck-7B)
run_phase "phase4a_teacher" \
    python tools/run_distillation.py --generate-labels \
        --teacher bespoke-minicheck \
        --dataset halueval \
        --max-samples 35000 \
        --output labels/teacher_soft_labels.json

if [ -f labels/teacher_soft_labels.json ]; then
    N=$(python3 -c "import json; print(len(json.load(open('labels/teacher_soft_labels.json'))))" 2>/dev/null || echo "?")
    echo "  Teacher labels: $N pairs" >> "$SUMMARY"
fi

# Step 2: Train distilled student
run_phase "phase4a_student" \
    python tools/run_distillation.py --train \
        --labels labels/teacher_soft_labels.json \
        --output-dir models/distilled-student \
        --temperature 2.0 --alpha 0.5 \
        --lr 5e-5 --epochs 3 --batch-size 8

# Evaluate distilled student
run_phase "phase4a_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/distilled-student

EVAL_D=$(ls -t benchmarks/results/aggrefact_distilled-student*.json 2>/dev/null | head -1)
[ -n "$EVAL_D" ] && cp "$EVAL_D" results/phase4a_eval.json && extract_ba results/phase4a_eval.json >> "$SUMMARY"

# ── PHASE 6A Step 2: Train Meta-Classifier ────────────────────────

echo "" >> "$SUMMARY"
echo "## Phase 6A-train: Meta-Classifier" >> "$SUMMARY"

run_phase "phase6a_train" \
    python tools/train_meta_classifier.py --train \
        --features features/aggrefact_meta_features.json \
        --output models/meta_classifier.pkl

if [ -f models/meta_classifier.pkl ]; then
    echo "  Meta-classifier saved" >> "$SUMMARY"
    if [ -f models/meta_classifier_metrics.json ]; then
        python3 -c "
import json
with open('models/meta_classifier_metrics.json') as f:
    d = json.load(f)
print(f'  Test BA: {d[\"test_balanced_acc\"]*100:.1f}%')
print(f'  Test F1: {d[\"test_f1\"]*100:.1f}%')
" >> "$SUMMARY" 2>/dev/null
    fi
fi

# ── Final: Package Results ────────────────────────────────────────

echo "" >> "$SUMMARY"
echo "## Completion" >> "$SUMMARY"
echo "Finished: $(ts)" >> "$SUMMARY"
echo "" >> "$SUMMARY"

# Summary comparison table
echo "## Comparison Table" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
python3 -c "
import json, glob, os
results = {}
for f in sorted(glob.glob('results/phase*.json')):
    tag = os.path.basename(f).replace('.json', '')
    try:
        with open(f) as fh:
            d = json.load(fh)
        ba = d.get('avg_balanced_accuracy_pct', d.get('avg_balanced_accuracy', 0))
        if isinstance(ba, float) and ba < 1:
            ba = ba * 100
        results[tag] = ba
    except Exception:
        pass
if results:
    baseline = results.get('phase0_baseline', 0)
    print(f'  {\"Experiment\":<35} {\"BA%\":>7} {\"vs Base\":>8}')
    print(f'  {\"-\"*50}')
    for tag, ba in sorted(results.items()):
        delta = ba - baseline if baseline else 0
        print(f'  {tag:<35} {ba:>6.1f}% {delta:>+7.1f}pp')
" >> "$SUMMARY" 2>/dev/null
echo '```' >> "$SUMMARY"

# Create lightweight results package (adapters, not merged models)
log_status "Packaging results"
tar czf results_package.tar.gz \
    results/ logs/ features/ labels/ STATUS \
    benchmarks/results/ \
    models/meta_classifier.pkl \
    models/meta_classifier_metrics.json \
    models/lora-halueval-r8-lr1e4/ \
    models/lora-halueval-r8-lr5e5/ \
    models/lora-halueval-sieved/ \
    models/lora-vitaminc-50k/ \
    models/distilled-student/ \
    2>/dev/null || true

PKGSIZE=$(du -h results_package.tar.gz 2>/dev/null | cut -f1 || echo "?")

log_status "ALL PHASES COMPLETE — results_package.tar.gz ($PKGSIZE)"

echo ""
echo "==================================================================="
echo "  ALL PHASES COMPLETE"
echo "  Results:  results/RESULTS_SUMMARY.md"
echo "  Package:  results_package.tar.gz ($PKGSIZE)"
echo "  Download: scp -i ~/.ssh/id_ed25519_upcloud root@<IP>:~/director-ai/results_package.tar.gz ."
echo "==================================================================="
cat results/RESULTS_SUMMARY.md
