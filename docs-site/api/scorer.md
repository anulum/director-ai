# CoherenceScorer

The central scoring engine. Computes a composite coherence score from two independent signals — NLI contradiction probability (H_logical) and RAG fact deviation (H_factual) — then accepts or rejects the response.

```
coherence = 1.0 - (W_LOGIC × H_logical + W_FACT × H_factual)
```

Effective default weights: `W_LOGIC = 0.6`, `W_FACT = 0.4`. The constructor
accepts `None` for `w_logic` and `w_fact` to inherit those defaults.

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
| `w_logic` | `float \| None` | `None` | Override weight for NLI divergence; `None` inherits `0.6` |
| `w_fact` | `float \| None` | `None` | Override weight for factual divergence; `None` inherits `0.4` |
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
| `llm_judge_provider` | `str` | `""` | `"openai"`, `"anthropic"`, or `"local"` |
| `privacy_mode` | `bool` | `False` | Redact PII before sending to an external LLM judge |
| `onnx_path` | `str \| None` | `None` | Directory with exported ONNX model |
| `nli_devices` | `list[str] \| None` | `None` | Multi-GPU sharding devices, for example `["cuda:0", "cuda:1"]` |

## Methods

### review()

```python
approved, score = scorer.review(prompt: str, action: str, session=None, tenant_id: str = "") -> tuple[bool, CoherenceScore]
```

Score a single prompt/response pair. Returns `(approved, CoherenceScore)`.

### review_batch()

```python
results = scorer.review_batch(items: list[tuple[str, str]]) -> list[tuple[bool, CoherenceScore]]
```

Score multiple pairs. Currently routes each item through `review()` sequentially. For parallel execution, wrap the scorer in `BatchProcessor`.

```python
items = [
    ("What is 2+2?", "The answer is 4."),
    ("Capital of France?", "Paris is in Germany."),
]
results = scorer.review_batch(items)
for approved, score in results:
    print(f"approved={approved}  score={score.score:.3f}")
```

### review_with_samples()

```python
scorer.enable_self_consistency(weight=0.25)  # opt-in, once
approved, score = scorer.review_with_samples(prompt, action, samples)
```

`review()` fused with a SelfCheckGPT-style semantic-entropy signal:
`samples` are alternative generations for the same prompt (a proxy
fanning out `n>1` completions, or an agent re-querying its model).
Samples are clustered by bidirectional entailment (the scorer's NLI
backend when model-backed, a lexical fallback otherwise — the backend
used is recorded in `score.self_consistency_backend`); the fused score
is `(1 − weight)·review + weight·consistency`. Fusion can revoke an
approval (fused score under the threshold) but never approves what
`review()` rejected. Attached fields: `self_consistency_score`,
`semantic_entropy`, `self_consistency_backend`.

### Chunked NLI

Sentence-level NLI scoring lives on `NLIScorer.score_chunked()`, not
`CoherenceScorer`. `CoherenceScorer` calls the NLI scorer internally when the
configured mode needs sentence-level aggregation.

## Scorer Backends

| Backend | Install | Latency | Accuracy | GPU |
|---------|---------|---------|----------|-----|
| `deberta` | `pip install director-ai[nli]` | 19 ms/pair (GPU batch) | 75.6% BA | Yes |
| `onnx` | `pip install director-ai[onnx]` | 14.6 ms/pair (GPU batch) | 75.6% BA | Yes |
| `minicheck` | `pip install director-ai[minicheck]` | ~60 ms/pair | 72.6% BA | Yes |
| `lite` | included | <0.5 ms/pair | ~65% BA | No |
| `hybrid` | `[nli]` + judge provider | 20-50 ms/pair | ~78% BA | Yes |
| `rust` | build `backfire-kernel` | ~1 ms/pair | ~65% BA | No |

### Backend registry and scaling helpers

Programmatic access to the same registry the `scorer_model` knob uses:

- `ScorerBackend` — abstract base class every backend implements.
- `register_backend(name, cls)`, `get_backend(name)`, `list_backends()`
  — register and resolve backend classes; `get_backend` raises
  `KeyError` for unknown names.
- `ShardedNLIScorer` — fans one logical scorer out over N `NLIScorer`
  instances pinned to different CUDA devices for multi-GPU hosts.
- `MetaClassifier` — logistic regression that predicts the dataset type
  so the pipeline can pick a matching scoring threshold.
- `clear_model_cache()` — evicts every cached NLI model to free GPU
  memory (useful between fine-tune activations).
- `export_tensorrt(...)` — pre-builds the TensorRT engine cache from an
  exported ONNX model so first-request latency stays flat.

## Validation Rules

- `threshold` must be in [0.0, 1.0]
- `soft_limit` must be >= `threshold`
- `w_logic + w_fact` must equal 1.0 when either override is provided
- `hybrid` backend requires `llm_judge_provider`
- External judge providers send structured JSON chat requests, request JSON
  responses, and use `privacy_mode=True` to redact PII before egress. The
  `local` judge provider keeps the escalation on host but requires a local
  judge model to be configured.

## OpenTelemetry

When `director-ai[otel]` is installed and OpenTelemetry is configured,
`review()` emits a parent `director_ai.review` span plus optional stage spans
for cache lookup, retrieval, NLI inference, calibration, and judge escalation.
The stage spans are no-ops when OTel is unavailable or no collector is
configured.

Stage span names and core attributes:

| Span | Key attributes |
|------|----------------|
| `director_ai.cache` | `cache.hit`, `cache.scope_present` |
| `director_ai.retrieval` | `retrieval.top_k`, `retrieval.tenant_scoped`, `retrieval.has_context`, `retrieval.result_count` |
| `director_ai.nli` | `nli.stage`, `nli.model_available`, `nli.score`, `nli.token_count` |
| `director_ai.calibration` | `calibration.stage`, `calibration.threshold`, `calibration.verdict_confidence`, `calibration.signal_agreement` |
| `director_ai.judge` | `judge.provider`, `judge.cache_hit`, `judge.nli_score`, `judge.adjusted_score` |

## Full API

::: director_ai.core.scoring.scorer.CoherenceScorer
