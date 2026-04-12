# Streaming Overhead

Director-AI's `StreamingKernel` scores every token by default. For
latency-critical paths you can reduce scoring frequency with cadence
parameters.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_every_n` | `int` | 1 | Score every N-th token. Skipped tokens reuse the last score. |
| `adaptive` | `bool` | False | Automatically adjust cadence based on coherence. |
| `max_cadence` | `int` | 8 | Upper bound for adaptive cadence ramp-up. |

These parameters are available on both `StreamingKernel` and
`AsyncStreamingKernel`.

## Overhead by Cadence

Measured with 200 tokens, heuristic scorer (no NLI), 5 iterations:

| Cadence | Tokens/s | Wall (ms) | Overhead | Callbacks |
|---------|----------|-----------|----------|-----------|
| none | ~150K | ~1 | — | 0 |
| 1 | ~4.5K | ~44 | +3200% | 200 |
| 4 | ~16K | ~13 | +870% | 50 |
| 8 | ~28K | ~7 | +440% | 25 |
| adaptive | ~18K | ~11 | +730% | ~38 |

Numbers are CPU-only (heuristic word-overlap scorer). NLI-backed
scoring re-runs the full DeBERTa pipeline on the accumulated text
per callback (~15-50 ms on GPU). Cadence reduction via
`score_every_n` or `adaptive=True` is critical for NLI production
deployments.

## Domain Recommendations

| Domain | Cadence | Rationale |
|--------|---------|-----------|
| Medical / Legal | `score_every_n=1` | Every token matters. False negatives are costly. |
| General chat | `score_every_n=4` | Good balance of safety and throughput. |
| Latency-critical | `score_every_n=8` | Minimal overhead. Rely on window/trend for catch-up. |
| Mixed workload | `adaptive=True` | Ramps up when coherent, snaps back on any dip. |

## Adaptive Cadence

When `adaptive=True`, the kernel adjusts scoring frequency at runtime:

- **Ramp up**: if the sliding window average is above `soft_limit` and
  the current cadence is below `max_cadence`, increment cadence by 1.
- **Snap back**: if any scored token falls below `soft_limit`, reset
  cadence to 1 (score every token).

This gives near-full coverage during degradation while reducing
overhead during stable generation.

## Backend Selection

The scorer backend determines per-callback cost. Choose based on your
latency budget and accuracy requirements.

| Backend | Per-pair (GPU) | Per-pair (CPU) | Accuracy | Best for |
|---------|---------------|----------------|----------|----------|
| `heuristic` | <0.1 ms | <0.1 ms | ~55% | Prototyping, edge devices |
| `deberta` (PyTorch) | 19 ms (batch) / 197 ms (seq) | N/A | 75.6% | Standard GPU production |
| `onnx` | **14.6 ms** (batch) / 65 ms (seq) | 383 ms | 75.6% | Fastest GPU path |
| `lite` | <1 ms | <1 ms | ~60% | High-throughput, low-accuracy OK |
| `hybrid` | 200-500 ms | 500-2000 ms | ~78% est. | Max accuracy, summarisation |

### Combining cadence with backends

The total per-token overhead is roughly `backend_latency / score_every_n`:

| Combination | Per-token overhead | Use case |
|-------------|-------------------|----------|
| ONNX GPU + cadence=1 | ~14.6 ms | Medical/legal — every token scored |
| ONNX GPU + cadence=4 | ~3.7 ms | General chat |
| ONNX GPU + cadence=8 | ~1.8 ms | Latency-critical streaming |
| Heuristic + cadence=1 | <0.1 ms | Offline/CPU-only environments |
| Hybrid + cadence=4 | ~50-125 ms | Summarisation with high catch rate |

### When to use hybrid mode

`scorer_backend="hybrid"` runs NLI first, then falls back to LLM-as-judge
when NLI confidence is in the grey zone (0.40-0.60). Trade-offs:

- **Latency**: 10-30x slower than pure NLI (adds an LLM API call)
- **Accuracy**: Strongest on summarisation (AggreFact-CNN, ExpertQA)
- **Privacy**: LLM judge sends truncated text (500 chars) to external API
- **Cost**: Each grey-zone call consumes LLM tokens

Recommended: use `hybrid` only for high-stakes summarisation pipelines
where the 200-500 ms per-pair overhead is acceptable.

## Reproducing

```bash
python -m benchmarks.streaming_overhead_bench
```

Outputs a JSON file at `benchmarks/results/streaming_overhead.json`
and a table to stdout.
