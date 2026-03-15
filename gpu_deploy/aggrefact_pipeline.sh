#!/bin/bash
# Director-AI AggreFact Pipeline Evaluation — UpCloud L40S
# Phases: save-scores (timing), per-dataset sweep, agg-sweep, bidirectional
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

R="/root/results"
mkdir -p "$R"
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$R/run.log"; }

log "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$R/run.log"

log "=== DEPS ==="
pip install -q torch==2.5.1 transformers==4.48.1 datasets==3.2.0 scikit-learn==1.6.1 nltk==3.9.1 accelerate==1.3.0 2>&1 | tail -3

log "=== UNPACK CODE ==="
cd /root/director-ai
# tarball already extracted by cloud-init

log "=== PHASE 1: SAVE SCORES (L40S timing) ==="
START=$(date +%s)
python -m benchmarks.aggrefact_eval --save-scores "$R/cached_scores_l40s.json" 2>&1 | tee -a "$R/run.log"
DUR=$(($(date +%s) - START))
log "Phase 1 done: ${DUR}s"
echo "save_scores_seconds=$DUR" > "$R/timing.txt"

log "=== PHASE 2: PER-DATASET SWEEP (cached, instant) ==="
python -m benchmarks.aggrefact_eval --load-scores "$R/cached_scores_l40s.json" --per-dataset 2>&1 | tee "$R/per_dataset_sweep.txt" | tee -a "$R/run.log"

log "=== PHASE 3: AGG SWEEP (3 strategies × 29K) ==="
START=$(date +%s)
python -m benchmarks.aggrefact_eval --agg-sweep 2>&1 | tee "$R/agg_sweep.txt" | tee -a "$R/run.log"
DUR=$(($(date +%s) - START))
log "Phase 3 done: ${DUR}s"
echo "agg_sweep_seconds=$DUR" >> "$R/timing.txt"

log "=== PHASE 4: BIDIRECTIONAL ==="
START=$(date +%s)
python -m benchmarks.aggrefact_eval --bidirectional --threshold 0.5 2>&1 | tee "$R/bidirectional.txt" | tee -a "$R/run.log"
DUR=$(($(date +%s) - START))
log "Phase 4 done: ${DUR}s"
echo "bidirectional_seconds=$DUR" >> "$R/timing.txt"

log "=== PACK ==="
cd /root
tar czf /root/aggrefact_results.tar.gz results/
echo "PIPELINE_COMPLETE" > "$R/STATUS"
log "=== ALL DONE ==="
