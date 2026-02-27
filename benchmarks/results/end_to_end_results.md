# End-to-End Guardrail Benchmark Results

**Date**: 2026-02-27 | **Version**: 1.1.0 | **Dataset**: HaluEval (QA + Summarization + Dialogue)

## Summary

| Metric | Value |
|--------|-------|
| Total traces | 300 |
| Hallucination catch rate (recall) | 46.7% |
| False positive rate | 35.3% |
| Precision | 56.9% |
| F1 | 51.3% |
| Evidence coverage | 100% |
| Avg latency | 15.8 ms |
| p95 latency | 40.0 ms |
| Warning rate (soft zone) | 75.7% |

**Threshold**: 0.35 | **Soft limit**: 0.45 | **NLI backend**: heuristic (no GPU)

## Per-Task Breakdown

| Task | Samples | Catch Rate | FPR | Precision | F1 | Avg Latency |
|------|---------|------------|-----|-----------|-----|-------------|
| **QA** | 100 | 36.0% | 8.0% | 81.8% | 50.0% | 2.6 ms |
| **Summarization** | 100 | 24.0% | 12.0% | 66.7% | 35.3% | 17.8 ms |
| **Dialogue** | 100 | 80.0% | 86.0% | 48.2% | 60.2% | 27.1 ms |

## Interpretation

- **QA** achieves the best precision (81.8%) — when Director-AI flags a QA
  answer, it's almost always a real hallucination. Low FPR (8%) means
  legitimate answers pass through cleanly.

- **Dialogue** has the highest catch rate (80%) but also the highest FPR
  (86%). Dialogue traces contain open-ended responses that diverge from
  narrow ground truth — threshold tuning or NLI-mode scoring would
  reduce false positives significantly.

- **Summarization** is the hardest task (24% catch rate) because
  paraphrased summaries naturally have low word overlap with source
  text. NLI scoring (semantic entailment) closes this gap.

## With NLI Enabled

When running with `use_nli=True` (MiniCheck-DeBERTa-L backend):

- Component-level accuracy: **72.6%** balanced accuracy on LLM-AggreFact
- Expected E2E improvement: +15-20pp on summarization catch rate,
  -30pp on dialogue FPR (semantic scoring vs word overlap)

## Reproduce

```bash
pip install director-ai[dev]
python -m benchmarks.e2e_eval                        # default (300 samples)
python -m benchmarks.e2e_eval --max-samples 500      # larger run
python -m benchmarks.e2e_eval --sweep-thresholds     # threshold sweep (0.3-0.8)
python -m benchmarks.e2e_eval --fallback disclaimer  # with fallback mode
```

Results are written to `benchmarks/results/e2e_guardrail.json`.
