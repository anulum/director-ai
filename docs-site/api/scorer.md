# CoherenceScorer

The central scoring engine. Computes a composite coherence score from two independent signals — NLI contradiction probability (H_logical) and RAG fact deviation (H_factual) — then accepts or rejects the response.

```
coherence = 1.0 - (W_LOGIC × H_logical + W_FACT × H_factual)
```

Default weights: `W_LOGIC = 0.6`, `W_FACT = 0.4`.

## Usage

```python
from director_ai import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("capital", "Paris is the capital of France.")

scorer = CoherenceScorer(
    threshold=0.6,
    ground_truth_store=store,
    use_nli=True,
)

approved, score = scorer.review(
    "What is the capital of France?",
    "The capital of France is Berlin.",
)

print(f"Approved: {approved}")        # False
print(f"Score: {score.score:.3f}")    # ~0.35
print(f"Evidence: {score.evidence}")  # Retrieved context + NLI details
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.5` | Minimum coherence to approve (0.0–1.0) |
| `soft_limit` | `float \| None` | `threshold + 0.1` | Warning zone upper bound |
| `w_logic` | `float` | `0.6` | Weight for NLI divergence |
| `w_fact` | `float` | `0.4` | Weight for factual divergence |
| `strict_mode` | `bool` | `False` | Reject if NLI unavailable (no heuristic fallback) |
| `use_nli` | `bool \| None` | `None` | `True` = force NLI, `False` = disable, `None` = auto-detect |
| `nli_model` | `str \| None` | `None` | HuggingFace model ID (default: FactCG-DeBERTa-v3-Large) |
| `ground_truth_store` | `GroundTruthStore \| None` | `None` | Fact store for RAG retrieval |
| `cache_size` | `int` | `0` | LRU cache max entries (0 = disabled) |
| `cache_ttl` | `float` | `300.0` | Cache entry TTL in seconds |
| `scorer_backend` | `str` | `"deberta"` | Backend: `deberta`, `onnx`, `minicheck`, `hybrid`, `lite`, `rust` |
| `nli_quantize_8bit` | `bool` | `False` | 8-bit quantization (reduces VRAM from ~1.5GB to ~400MB) |
| `nli_device` | `str \| None` | `None` | Torch device (`"cuda"`, `"cuda:0"`, `"cpu"`) |
| `nli_torch_dtype` | `str \| None` | `None` | Torch dtype (`"float16"`, `"bfloat16"`) |
| `history_window` | `int` | `5` | Rolling history size for trend detection |
| `llm_judge_enabled` | `bool` | `False` | Escalate to LLM when NLI confidence is low |
| `llm_judge_confidence_threshold` | `float` | `0.3` | Softmax margin below which to escalate |
| `llm_judge_provider` | `str` | `""` | `"openai"` or `"anthropic"` |
| `privacy_mode` | `bool` | `False` | Redact PII before sending to LLM judge |
| `onnx_path` | `str \| None` | `None` | Directory with exported ONNX model |
| `nli_devices` | `str \| None` | `None` | Multi-GPU sharding (comma-separated: `"cuda:0,cuda:1"`) |

## Methods

### review()

```python
approved, score = scorer.review(query: str, response: str) -> tuple[bool, CoherenceScore]
```

Score a single prompt/response pair. Returns `(approved, CoherenceScore)`.

### review_batch()

```python
results = scorer.review_batch(items: list[tuple[str, str]]) -> list[tuple[bool, CoherenceScore]]
```

Score multiple pairs in a single batch. Uses 2 GPU kernel calls total instead of 2×N — use for bulk evaluation or API batch endpoints.

```python
items = [
    ("What is 2+2?", "The answer is 4."),
    ("Capital of France?", "Paris is in Germany."),
]
results = scorer.review_batch(items)
for approved, score in results:
    print(f"approved={approved}  score={score.score:.3f}")
```

### score_chunked()

Sentence-level NLI scoring with max-aggregation. Catches localized hallucinations that full-text comparison would miss.

## Scorer Backends

| Backend | Install | Latency | Accuracy | GPU |
|---------|---------|---------|----------|-----|
| `deberta` | `pip install director-ai[nli]` | 19 ms/pair (GPU batch) | 75.8% BA | Yes |
| `onnx` | `pip install director-ai[onnx]` | 14.6 ms/pair (GPU batch) | 75.8% BA | Yes |
| `minicheck` | `pip install director-ai[minicheck]` | ~60 ms/pair | 72.6% BA | Yes |
| `lite` | included | <0.5 ms/pair | ~65% BA | No |
| `hybrid` | `[nli]` + LLM API key | 20-50 ms/pair | ~78% BA | Yes |
| `rust` | build `backfire-kernel` | ~1 ms/pair | ~65% BA | No |

## Validation Rules

- `threshold` must be in [0.0, 1.0]
- `soft_limit` must be >= `threshold`
- `w_logic + w_fact` must equal 1.0
- `hybrid` backend requires `llm_judge_provider`

## Full API

::: director_ai.core.scorer.CoherenceScorer
