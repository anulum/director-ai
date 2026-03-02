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
scoring is ~10x slower per callback; cadence reduction is more
impactful.

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

## Reproducing

```bash
python -m benchmarks.streaming_overhead_bench
```

Outputs a JSON file at `benchmarks/results/streaming_overhead.json`
and a table to stdout.
