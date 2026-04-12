# NLI Backends

Natural Language Inference scorer using FactCG-DeBERTa-v3-Large (75.6% per-dataset mean BA on AggreFact). Used internally by `CoherenceScorer` — direct use is only needed for custom pipelines or benchmarking.

## Usage

```python
from director_ai.core.nli import NLIScorer, nli_available

if nli_available():
    nli = NLIScorer()
    divergence = nli.score("Paris is the capital of France.", "Berlin is the capital of France.")
    print(f"Divergence: {divergence:.3f}")  # ~0.85 (high contradiction)
```

## NLIScorer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"yaxili96/FactCG-DeBERTa-v3-Large"` | HuggingFace model ID |
| `device` | `str \| None` | `None` | Torch device (`"cuda"`, `"cpu"`) |
| `quantize_8bit` | `bool` | `False` | 8-bit quantization |
| `torch_dtype` | `str \| None` | `None` | `"float16"`, `"bfloat16"` |
| `backend` | `str` | `"deberta"` | `"deberta"`, `"onnx"`, `"minicheck"`, `"lite"` |
| `use_model` | `bool` | `True` | `False` = heuristic-only mode |

### Methods

- `score(premise, hypothesis) -> float` — NLI divergence in [0, 1]
- `score_batch(pairs) -> list[float]` — batch inference (2 GPU kernels)
- `score_chunked(premise, hypothesis) -> tuple[float, list[float]]` — sentence-level with max-aggregation; returns (aggregate_score, per_chunk_scores)
- `score_claim_coverage(source, summary, support_threshold=0.6) -> tuple[float, list[float], list[str]]` — per-claim coverage; returns (coverage, per_claim_divergences, claims)

## nli_available()

```python
from director_ai.core.nli import nli_available

if nli_available():
    print("torch + transformers installed — NLI model available")
```

Returns `True` if `torch` and `transformers` are importable.

## Backend Comparison

| Backend | Model | Latency (GPU batch) | Accuracy | VRAM |
|---------|-------|---------------------|----------|------|
| `deberta` | FactCG-DeBERTa-v3-Large | 19 ms/pair | 75.6% BA | ~1.5 GB |
| `onnx` | Same (exported) | 14.6 ms/pair | 75.6% BA | ~1.2 GB |
| `minicheck` | MiniCheck-DeBERTa-L | ~60 ms/pair | 72.6% BA | ~400 MB |
| `lite` | word-overlap heuristic | <0.5 ms/pair | ~65% BA | 0 |

## Full API

::: director_ai.core.scoring.nli.NLIScorer

::: director_ai.core.scoring.nli.nli_available
