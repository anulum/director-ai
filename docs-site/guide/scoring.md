# Scoring

## How Coherence Scoring Works

Director-AI computes a composite coherence score from two signals:

- **H_logical** (weight 0.6): NLI contradiction probability between prompt and response
- **H_factual** (weight 0.4): ground-truth deviation via RAG retrieval

```
coherence = 1.0 - (0.6 * H_logical + 0.4 * H_factual)
```

## Thresholds

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `threshold` / `hard_limit` | 0.5 | Below this = rejected |
| `soft_limit` | 0.6 | Between hard and soft = warning zone |

## NLI Backends

### Heuristic (default, no GPU)
Word-overlap scoring. Fast (<1ms) but limited to vocabulary-level detection.

### FactCG-DeBERTa-v3-Large (default)
```python
scorer = CoherenceScorer(use_nli=True)
```
75.8% balanced accuracy on AggreFact (4th on leaderboard). Uses instruction
template + SummaC source chunking. ~575 ms CPU, ~50-80 ms GPU.

### MiniCheck (alternative)
```python
scorer = CoherenceScorer(
    use_nli=True,
    nli_model="lytang/MiniCheck-DeBERTa-L",
)
```
72.6% balanced accuracy. Faster on CPU (~120 ms) but lower accuracy.

## Score Caching

For streaming workloads, enable caching to avoid redundant scoring:

```python
scorer = CoherenceScorer(
    cache_size=1024,  # max entries
    cache_ttl=300.0,  # seconds
)
```

Access cache stats:
```python
print(f"Hit rate: {scorer.cache.hit_rate:.1%}")
print(f"Size: {scorer.cache.size}")
```

## Soft Warning Zone

Scores between `threshold` and `soft_limit` trigger a warning but still pass:

```python
scorer = CoherenceScorer(threshold=0.5, soft_limit=0.65)
approved, score = scorer.review(query, response)
if score.warning:
    print("Low confidence — consider verification")
```
