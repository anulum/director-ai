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
    scorer_backend="deberta", # "deberta", "onnx", or "hybrid"
    onnx_path=None,           # ONNX model path
)
```

## Methods

### `review(prompt, response) -> tuple[bool, CoherenceScore]`

Score a response. Returns `(approved, CoherenceScore)`.

### `areview(prompt, response) -> tuple[bool, CoherenceScore]`

Async version. Offloads NLI to a thread pool.

### `review_batch(items) -> list[tuple[bool, CoherenceScore]]`

Coalesced batch review. Accepts `list[tuple[str, str]]` of (prompt, response)
pairs. Runs 2 GPU kernel calls total (one for H_logical, one for H_factual)
instead of 2*N per-item calls.

```python
items = [
    ("What is 2+2?", "The answer is 4."),
    ("Capital of France?", "Paris is in Germany."),
]
results = scorer.review_batch(items)
for approved, score in results:
    print(f"approved={approved}  score={score.score:.3f}")
```

Falls back to per-item `review()` if the batch path raises an exception.

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

store = VectorGroundTruthStore()  # empty — add your own facts
store.add("photosynthesis", "Plants convert CO2 into glucose using sunlight.")

scorer = CoherenceScorer(threshold=0.6, use_nli=True, ground_truth_store=store)

approved, cs = scorer.review(
    "How do plants make food?",
    "Plants use sunlight to convert CO2 into glucose.",
)
print(f"approved={approved}  score={cs.score:.3f}  warning={cs.warning}")
```

## Validation Rules (v2.2.0+)

- `threshold` must be in [0, 1]
- `soft_limit` defaults to `min(threshold + 0.1, 1.0)`, must be in [0, 1] and >= threshold
- `w_logic` and `w_fact` must each be in [0, 1] and sum to 1.0
- `scorer_backend` accepts `"deberta"`, `"onnx"`, or `"hybrid"`

Invalid values raise `ValueError` at construction time.
