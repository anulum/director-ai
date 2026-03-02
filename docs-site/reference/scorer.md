# CoherenceScorer

Dual-entropy scorer. Coherence = `1 - (w_logic * H_logical + w_fact * H_factual)`.

## Constructor

```python
from director_ai import CoherenceScorer

scorer = CoherenceScorer(
    threshold=0.5,            # minimum coherence to approve
    soft_limit=None,          # warning zone floor (default: threshold + 0.1)
    w_logic=0.6,              # logical divergence weight
    w_fact=0.4,               # factual divergence weight (must sum to 1.0)
    strict_mode=False,        # disable heuristic fallbacks when NLI unavailable
    use_nli=None,             # True/False/None (auto-detect)
    ground_truth_store=None,  # GroundTruthStore for RAG factual checks
    cache_size=0,             # LRU cache entries (0 = disabled)
    scorer_backend="deberta", # "deberta" or "onnx"
    onnx_path=None,           # ONNX model path
)
```

## Methods

### `review(prompt, response) -> tuple[bool, CoherenceScore]`

Score a response. Returns `(approved, CoherenceScore)`.

### `areview(prompt, response) -> tuple[bool, CoherenceScore]`

Async version. Offloads NLI to a thread pool.

### `compute_divergence(prompt, action) -> float`

Raw composite divergence in `[0, 1]` (lower is better).

## CoherenceScore

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Composite coherence (0 = incoherent, 1 = perfect) |
| `approved` | `bool` | Passes threshold gate |
| `h_logical` | `float` | Logical divergence component |
| `h_factual` | `float` | Factual divergence component |
| `evidence` | `ScoringEvidence\|None` | RAG chunks + NLI details |
| `warning` | `bool` | True when score is in `[threshold, soft_limit)` |

## Example

```python
from director_ai import CoherenceScorer
from director_ai.core.vector_store import VectorGroundTruthStore

store = VectorGroundTruthStore()
store.add("photosynthesis", "Plants convert CO2 into glucose using sunlight.")

scorer = CoherenceScorer(threshold=0.6, use_nli=True, ground_truth_store=store)

approved, cs = scorer.review(
    "How do plants make food?",
    "Plants use sunlight to convert CO2 into glucose.",
)
print(f"approved={approved}  score={cs.score:.3f}  warning={cs.warning}")
```
