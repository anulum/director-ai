#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — GPU Runner v2: LoRA (frozen head) + Overlap Eval
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
# Task 1: LoRA re-run with modules_to_save=[] (5 experiments)
# Task 2: AggreFact re-eval with overlap chunking (2 evals)
#
# screen -S train bash gpu_deploy/master_runner_v2.sh
# ─────────────────────────────────────────────────────────────────────
set -o pipefail

PROJ_ROOT="/root/director-ai"
cd "$PROJ_ROOT"

source /opt/director-venv/bin/activate
export PYTHONPATH="$PROJ_ROOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p models results logs benchmarks/results

STATUS="$PROJ_ROOT/STATUS"
SUMMARY="$PROJ_ROOT/results/RESULTS_SUMMARY_V2.md"

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
        echo '```' >> "$SUMMARY"
        tail -20 "$logfile" >> "$SUMMARY" 2>/dev/null
        echo '```' >> "$SUMMARY"
    fi

    tar czf "results/checkpoint_${name}.tar.gz" \
        --exclude='*.tar.gz' \
        results/ logs/ STATUS 2>/dev/null || true

    return 0
}

extract_ba() {
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

echo "# Director-AI Accuracy Improvement v2" > "$SUMMARY"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)" >> "$SUMMARY"
echo "Started: $(ts)" >> "$SUMMARY"
echo "Focus: LoRA with frozen classification head + overlap chunking eval" >> "$SUMMARY"

log_status "PIPELINE V2 START"

# ── PHASE 0: Baseline eval ────────────────────────────────────────

echo "" >> "$SUMMARY"
echo "## Phase 0: Baseline" >> "$SUMMARY"

run_phase "p0_baseline" \
    python -m benchmarks.aggrefact_eval --sweep

BASELINE_FILE=$(ls -t benchmarks/results/aggrefact_*.json 2>/dev/null | head -1)
if [ -n "$BASELINE_FILE" ]; then
    cp "$BASELINE_FILE" results/p0_baseline.json
    extract_ba results/p0_baseline.json >> "$SUMMARY"
fi

# ── TASK 1: LoRA Experiments (frozen head) ────────────────────────

echo "" >> "$SUMMARY"
echo "## Task 1: LoRA (modules_to_save=[])" >> "$SUMMARY"

# Sanity check: verify modules_to_save=[] is in the training script
if grep -q 'modules_to_save=\[\]' tools/run_lora_training.py; then
    echo "  Verified: modules_to_save=[] in run_lora_training.py" >> "$SUMMARY"
else
    echo "  WARNING: modules_to_save=[] NOT FOUND — aborting LoRA" >> "$SUMMARY"
    log_status "ABORT: modules_to_save fix not present"
    # Skip to Task 2
    SKIP_LORA=1
fi

if [ -z "${SKIP_LORA:-}" ]; then

# --- Exp 1: HaluEval, r=8, lr=1e-4 ---
run_phase "lora_exp1_train" \
    python tools/run_lora_training.py \
        --dataset halueval \
        --output-dir models/lora-exp1-halueval-r8-lr1e4 \
        --rank 8 --alpha 16 --lr 1e-4 --epochs 5 \
        --batch-size 8 --grad-accum 4

run_phase "lora_exp1_merge" \
    python gpu_deploy/merge_lora.py \
        --adapter models/lora-exp1-halueval-r8-lr1e4 \
        --output models/merged-exp1

run_phase "lora_exp1_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/merged-exp1

EVAL=$(ls -t benchmarks/results/aggrefact_merged-exp1*.json 2>/dev/null | head -1)
[ -n "$EVAL" ] && cp "$EVAL" results/lora_exp1.json && extract_ba results/lora_exp1.json >> "$SUMMARY"

# --- Exp 2: HaluEval, r=8, lr=5e-5 ---
run_phase "lora_exp2_train" \
    python tools/run_lora_training.py \
        --dataset halueval \
        --output-dir models/lora-exp2-halueval-r8-lr5e5 \
        --rank 8 --alpha 16 --lr 5e-5 --epochs 5 \
        --batch-size 8 --grad-accum 4

run_phase "lora_exp2_merge" \
    python gpu_deploy/merge_lora.py \
        --adapter models/lora-exp2-halueval-r8-lr5e5 \
        --output models/merged-exp2

run_phase "lora_exp2_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/merged-exp2

EVAL=$(ls -t benchmarks/results/aggrefact_merged-exp2*.json 2>/dev/null | head -1)
[ -n "$EVAL" ] && cp "$EVAL" results/lora_exp2.json && extract_ba results/lora_exp2.json >> "$SUMMARY"

# --- Exp 3: HaluEval, r=4, lr=5e-5 (conservative rank) ---
run_phase "lora_exp3_train" \
    python tools/run_lora_training.py \
        --dataset halueval \
        --output-dir models/lora-exp3-halueval-r4-lr5e5 \
        --rank 4 --alpha 8 --lr 5e-5 --epochs 5 \
        --batch-size 8 --grad-accum 4

run_phase "lora_exp3_merge" \
    python gpu_deploy/merge_lora.py \
        --adapter models/lora-exp3-halueval-r4-lr5e5 \
        --output models/merged-exp3

run_phase "lora_exp3_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/merged-exp3

EVAL=$(ls -t benchmarks/results/aggrefact_merged-exp3*.json 2>/dev/null | head -1)
[ -n "$EVAL" ] && cp "$EVAL" results/lora_exp3.json && extract_ba results/lora_exp3.json >> "$SUMMARY"

# --- Exp 4: HaluEval + Sieving, r=8, lr=1e-4 ---
run_phase "lora_exp4_train" \
    python tools/run_lora_training.py \
        --dataset halueval \
        --output-dir models/lora-exp4-halueval-sieved \
        --rank 8 --alpha 16 --lr 1e-4 --epochs 5 \
        --batch-size 8 --grad-accum 4 \
        --sieving --sieve-rate 0.1

run_phase "lora_exp4_merge" \
    python gpu_deploy/merge_lora.py \
        --adapter models/lora-exp4-halueval-sieved \
        --output models/merged-exp4

run_phase "lora_exp4_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/merged-exp4

EVAL=$(ls -t benchmarks/results/aggrefact_merged-exp4*.json 2>/dev/null | head -1)
[ -n "$EVAL" ] && cp "$EVAL" results/lora_exp4.json && extract_ba results/lora_exp4.json >> "$SUMMARY"

# --- Exp 5: VitaminC 50K, r=8, lr=1e-4 ---
run_phase "lora_exp5_train" \
    python tools/run_lora_training.py \
        --dataset vitaminc \
        --max-samples 50000 \
        --output-dir models/lora-exp5-vitaminc-50k \
        --rank 8 --alpha 16 --lr 1e-4 --epochs 3 \
        --batch-size 8 --grad-accum 4

run_phase "lora_exp5_merge" \
    python gpu_deploy/merge_lora.py \
        --adapter models/lora-exp5-vitaminc-50k \
        --output models/merged-exp5

run_phase "lora_exp5_eval" \
    python -m benchmarks.aggrefact_eval --sweep \
        --model models/merged-exp5

EVAL=$(ls -t benchmarks/results/aggrefact_merged-exp5*.json 2>/dev/null | head -1)
[ -n "$EVAL" ] && cp "$EVAL" results/lora_exp5.json && extract_ba results/lora_exp5.json >> "$SUMMARY"

fi  # end SKIP_LORA

# ── TASK 2: Overlap Chunking Eval ────────────────────────────────

echo "" >> "$SUMMARY"
echo "## Task 2: Overlap Chunking Eval" >> "$SUMMARY"

# 2A: Bidirectional, no overlap (baseline for comparison)
run_phase "overlap_baseline" \
    python -m benchmarks.aggrefact_eval \
        --bidirectional --overlap 0.0

OVAL=$(ls -t benchmarks/results/aggrefact_*.json 2>/dev/null | head -1)
[ -n "$OVAL" ] && cp "$OVAL" results/overlap_baseline.json && extract_ba results/overlap_baseline.json >> "$SUMMARY"

# 2B: Bidirectional, 50% overlap
run_phase "overlap_50pct" \
    python -m benchmarks.aggrefact_eval \
        --bidirectional --overlap 0.5

OVAL=$(ls -t benchmarks/results/aggrefact_*.json 2>/dev/null | head -1)
[ -n "$OVAL" ] && cp "$OVAL" results/overlap_50pct.json && extract_ba results/overlap_50pct.json >> "$SUMMARY"

# ── Final: Package ────────────────────────────────────────────────

echo "" >> "$SUMMARY"
echo "## Comparison Table" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
python3 -c "
import json, glob, os
results = {}
for f in sorted(glob.glob('results/*.json')):
    tag = os.path.basename(f).replace('.json', '')
    if tag.startswith('checkpoint'):
        continue
    try:
        with open(f) as fh:
            d = json.load(fh)
        ba = d.get('avg_balanced_accuracy_pct', d.get('avg_balanced_accuracy', 0))
        if isinstance(ba, float) and ba < 1:
            ba = ba * 100
        t = d.get('threshold', '?')
        results[tag] = (ba, t)
    except Exception:
        pass
if results:
    baseline_ba = results.get('p0_baseline', (0, 0))[0]
    print(f'  {\"Experiment\":<30} {\"BA%\":>7} {\"Thresh\":>7} {\"vs Base\":>8}')
    print(f'  {\"-\"*55}')
    for tag, (ba, t) in sorted(results.items()):
        delta = ba - baseline_ba if baseline_ba else 0
        print(f'  {tag:<30} {ba:>6.1f}% {t:>7} {delta:>+7.1f}pp')
" >> "$SUMMARY" 2>/dev/null
echo '```' >> "$SUMMARY"

echo "" >> "$SUMMARY"
echo "Finished: $(ts)" >> "$SUMMARY"

log_status "Packaging results"
tar czf results_package_v2.tar.gz \
    --exclude='*.tar.gz' \
    results/ logs/ STATUS benchmarks/results/ \
    models/lora-exp*/training_metrics.json \
    2>/dev/null || true

PKGSIZE=$(du -h results_package_v2.tar.gz 2>/dev/null | cut -f1 || echo "?")
log_status "ALL PHASES COMPLETE — results_package_v2.tar.gz ($PKGSIZE)"

echo ""
echo "==================================================================="
echo "  ALL PHASES COMPLETE"
echo "  Results:  results/RESULTS_SUMMARY_V2.md"
echo "  Package:  results_package_v2.tar.gz ($PKGSIZE)"
echo "==================================================================="
cat results/RESULTS_SUMMARY_V2.md
