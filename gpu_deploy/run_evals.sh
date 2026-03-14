#!/usr/bin/env bash
set -euo pipefail
source /opt/director-venv/bin/activate
export PYTHONPATH=/root/director-ai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/director-ai
mkdir -p results

STATUS=/root/director-ai/EVAL_STATUS
echo "=== Eval Pipeline v3 (local JSONL) ===" > "$STATUS"
echo "$(date -u '+%H:%M:%S') Started" >> "$STATUS"

eval_variant() {
    local tag="$1"
    shift
    echo "$(date -u '+%H:%M:%S') START $tag" >> "$STATUS"
    local t0
    t0=$(date +%s)
    if python tools/eval_aggrefact.py "$tag" "$@" > "logs/eval_${tag}.log" 2>&1; then
        local t1
        t1=$(date +%s)
        local result
        result=$(tail -1 "logs/eval_${tag}.log")
        echo "$(date -u '+%H:%M:%S') DONE  $tag ($(( t1 - t0 ))s) $result" >> "$STATUS"
    else
        local rc=$?
        local t1
        t1=$(date +%s)
        echo "$(date -u '+%H:%M:%S') FAIL  $tag (exit $rc, $(( t1 - t0 ))s)" >> "$STATUS"
        tail -3 "logs/eval_${tag}.log" >> "$STATUS"
    fi
}

eval_variant baseline
eval_variant head_only models/distill-head-only
eval_variant lora_r4 models/distill-lora-r4
eval_variant lora_r8 models/distill-lora-r8

echo "" >> "$STATUS"
echo "=== FINAL RESULTS ===" >> "$STATUS"
for f in results/distill_eval_*.json; do
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(f\"{d['model']:12s} {d['macro_ba']*100:6.2f}% BA  (t={d['threshold']})\")
" "$f" >> "$STATUS" 2>/dev/null || true
done
cat "$STATUS"
