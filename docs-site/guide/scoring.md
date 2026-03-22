# Scoring

## How Coherence Scoring Works

Director-AI computes a composite coherence score from two independent signals:

```
coherence = 1.0 - (W_LOGIC × H_logical + W_FACT × H_factual)
```

| Signal | Weight | Source | Measures |
|--------|--------|--------|----------|
| **H_logical** | 0.6 | NLI model (DeBERTa) | Contradiction probability between prompt and response |
| **H_factual** | 0.4 | RAG retrieval | Deviation from ground-truth knowledge base |

The score is in [0.0, 1.0]. Higher = more coherent.

## Thresholds

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `threshold` | 0.5 | Below this = rejected |
| `soft_limit` | `threshold + 0.1` | Between threshold and soft_limit = warning zone |

```python
scorer = CoherenceScorer(threshold=0.5, soft_limit=0.65)
approved, score = scorer.review(query, response)

if not approved:
    print("Rejected — below threshold")
elif score.warning:
    print("Warning — low confidence, consider verification")
else:
    print("Approved")
```

## NLI Backends

### Heuristic (default, no GPU)

Word-overlap scoring. Fast (<1ms) but limited to vocabulary-level detection.

```python
scorer = CoherenceScorer(use_nli=False)
```

### FactCG-DeBERTa-v3-Large (recommended)

75.8% balanced accuracy on AggreFact. Uses instruction template + SummaC source chunking.

```python
scorer = CoherenceScorer(use_nli=True)
```

| Backend | Latency | Accuracy |
|---------|---------|----------|
| ONNX GPU batch | 14.6 ms/pair | 75.8% BA |
| PyTorch GPU batch | 19 ms/pair | 75.8% BA |
| PyTorch GPU sequential | 197 ms/pair | 75.8% BA |
| ONNX CPU batch | 383 ms/pair | 75.8% BA |

### MiniCheck (lighter alternative)

72.6% balanced accuracy. Lower VRAM (~400MB vs ~1.5GB).

```python
scorer = CoherenceScorer(
    use_nli=True,
    nli_model="lytang/MiniCheck-DeBERTa-L",
)
```

### LiteScorer (CPU-only, ~65% accuracy)

Word overlap + length ratio + negation heuristics. <0.5 ms/pair, no dependencies.

```python
scorer = CoherenceScorer(scorer_backend="lite")
```

## Customizing Weights

Adjust the balance between logical and factual signals:

```python
# Fact-heavy (for KB-grounded use cases)
scorer = CoherenceScorer(w_logic=0.3, w_fact=0.7)

# Logic-heavy (for free-form reasoning)
scorer = CoherenceScorer(w_logic=0.8, w_fact=0.2)

# Summarization (factual only, no logic duplication)
scorer = CoherenceScorer(w_logic=0.0, w_fact=1.0)
```

Constraint: `w_logic + w_fact` must equal 1.0.

## Score Caching

Enable caching to avoid redundant NLI inference (60-80% cost reduction in streaming):

```python
scorer = CoherenceScorer(
    cache_size=2048,
    cache_ttl=300.0,
)

# Monitor cache
print(f"Hit rate: {scorer.cache.hit_rate:.1%}")
print(f"Size: {scorer.cache.size}")
```

## Batch Scoring

Score multiple pairs in 2 GPU forward passes (when NLI is available):

```python
items = [
    ("What is 2+2?", "The answer is 4."),
    ("Capital of France?", "Paris is in Germany."),
]
results = scorer.review_batch(items)
```

## Chunked NLI

For long documents, sentence-level scoring catches localized hallucinations:

```python
divergence = scorer._nli.score_chunked(
    premise="Paris is the capital of France. The Eiffel Tower is in Paris.",
    hypothesis="Berlin is the capital of France. The Eiffel Tower is in Berlin.",
)
```

Max-aggregation: the worst per-sentence contradiction drives the final score.

## Next Steps

- [Threshold Tuning](threshold-tuning.md) — domain-specific calibration
- [Streaming Halt](streaming.md) — token-level oversight
- [KB Ingestion](kb-ingestion.md) — populate the factual signal
