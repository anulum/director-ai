# Good First Issues

Starter tasks for new contributors. Each is self-contained and well-scoped.

## Documentation

1. **Add docstrings to `integrations/crewai.py`** — the public methods need
   NumPy-style docstrings matching the rest of the codebase.

2. **Add a "Quickstart" Colab notebook** — convert `examples/quickstart.py`
   into a Jupyter notebook with markdown explanations.

3. **Improve cookbook examples** — add a customer support domain cookbook
   in `docs-site/cookbook/support.md` following the legal/medical/finance pattern.

## Testing

4. **Add edge-case tests for `ScoreCache`** — concurrent access from multiple
   threads, unicode keys, very long query strings.

5. **Add tests for `SentenceTransformerBackend`** — mock the model to test
   add/query/count without downloading weights.

## Features

6. **Add `__repr__` to `CoherenceScore`** — human-readable representation
   showing score, approved, and warning status.

7. **Add `cache_stats` property to `CoherenceScorer`** — return a dict with
   hits, misses, hit_rate, and size when caching is enabled.

8. **Chunked NLI scoring** — split long documents into overlapping chunks
   before NLI scoring, aggregate with max or mean.

## Integrations

9. **Add Guardrails AI integration** — implement a `DirectorAIValidator`
   compatible with the Guardrails AI framework.

10. **Add DSPy integration** — implement a `DirectorAIAssert` module
    for DSPy assertion-based optimization.
