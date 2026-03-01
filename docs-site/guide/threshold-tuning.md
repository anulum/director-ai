# Threshold Tuning Guide

## When to Use Heuristic vs NLI

| Mode | Latency | Accuracy | Best For |
|------|---------|----------|----------|
| Heuristic only (`use_nli=False`) | < 0.1 ms | Moderate — catches obvious off-topic responses | High-throughput, cost-sensitive, or when KB coverage is strong |
| NLI (`use_nli=True`) | 15-200 ms (GPU/CPU) | High — catches subtle contradictions | Medical, legal, finance, or any domain where factual precision matters |
| Chunked NLI (`score_chunked`) | 30-400 ms | Highest — catches localized hallucinations | Long responses where a single hallucinated sentence hides in correct text |

**Rule of thumb**: start with heuristic for development. Switch to NLI for production if your domain has high factual stakes.

## Score Components

The coherence score is a weighted combination:

```
score = (1 - w_logic) * (1 - w_fact) * heuristic_base
      + w_logic * (1 - h_logical)
      + w_fact * (1 - h_factual)
```

Where:

- `heuristic_base` — word-overlap between response and KB facts (0-1)
- `h_logical` — NLI-derived logical divergence (0 = entailed, 1 = contradicted)
- `h_factual` — NLI-derived factual divergence from KB retrieval
- `w_logic`, `w_fact` — configurable weights (default 0.3, 0.3)

**Adjusting weights:**

- High `w_logic` → penalizes logical contradictions more (good for reasoning tasks)
- High `w_fact` → penalizes factual divergence from KB more (good for RAG pipelines)
- Both low → relies primarily on heuristic word overlap

## Running a Threshold Sweep

```bash
python -m benchmarks.e2e_eval --sweep-thresholds --max-samples 200
```

This scores all samples once, then evaluates catch rate / FPR at thresholds from 0.30 to 0.80:

```
 Threshold    Catch      FPR     Prec       F1
      0.30   89.2%    45.1%    66.3%    76.1%
      0.35   82.4%    32.0%    72.0%    76.9%
      0.40   74.1%    21.3%    77.8%    75.9%
      0.45   65.2%    14.5%    81.8%    72.5%
      0.50   55.0%     8.7%    86.3%    67.2%
      ...
```

Pick the threshold where F1 is maximized for your risk tolerance.

## Decision Matrix

| Symptom | Action |
|---------|--------|
| High false-positive rate (correct responses rejected) | Lower `coherence_threshold` by 0.05-0.10 |
| Missing hallucinations (low catch rate) | Raise `coherence_threshold` by 0.05-0.10 |
| Good catch rate but noisy warnings | Raise `soft_limit` closer to threshold |
| Streaming halts too aggressively | Increase `window_size` or lower `trend_threshold` |
| Streaming misses gradual degradation | Decrease `window_size` or raise `trend_threshold` |
| NLI scores all cluster near 0.5 | Check KB coverage — scorer needs grounding facts to differentiate |

## Domain-Specific Presets

### Medical

```python
scorer = CoherenceScorer(
    threshold=0.6,
    soft_limit=0.7,
    use_nli=True,
    ground_truth_store=medical_kb,
)
```

See [Medical Cookbook](../cookbook/medical.md) and `examples/medical_guard.py`.

### Customer Support

```python
scorer = CoherenceScorer(
    threshold=0.5,
    soft_limit=0.6,
    use_nli=True,
    ground_truth_store=support_kb,
)
```

See [Customer Support Cookbook](../cookbook/customer-support.md) and `examples/customer_support_guard.py`.

### Finance

```python
scorer = CoherenceScorer(
    threshold=0.55,
    soft_limit=0.65,
    use_nli=True,
)
```

See [Finance Cookbook](../cookbook/finance.md).

## Streaming Threshold Tuning

For streaming workloads, you tune 4 parameters:

| Parameter | Default | Effect of Raising | Effect of Lowering |
|-----------|---------|-------------------|--------------------|
| `hard_limit` | 0.5 | More immediate halts | Tolerates brief dips |
| `window_threshold` | 0.55 | Stricter sustained quality | Allows temporary degradation |
| `trend_threshold` | 0.15 | More sensitive to coherence drops | Ignores gradual decline |
| `window_size` | 10 | Smooths noise (less reactive) | Faster response to changes |

Use `streaming_debug=True` to inspect per-token scores and identify which mechanism triggers. See [Streaming Halt](streaming.md#debug-mode).
