# Streaming Halt

## Overview

`StreamingKernel` monitors coherence token-by-token and halts generation the moment quality degrades. Three independent halt mechanisms:

1. **Hard limit** — any single token below threshold
2. **Sliding window** — rolling average drops below window threshold
3. **Downward trend** — coherence drops by more than a delta over N tokens

## Architecture

```
LLM token stream ──► StreamingKernel
                       │
                       ├─ coherence_callback(token) → float
                       │    ├─ accumulate tokens into partial response
                       │    ├─ scorer.review(prompt, partial_response)
                       │    └─ return score.score
                       │
                       ├─ 3 halt mechanisms:
                       │    ├─ hard_limit:   score < threshold → emergency stop
                       │    ├─ window_avg:   sliding window mean < threshold
                       │    └─ trend_drop:   coherence[0] - coherence[-1] > delta
                       │
                       └─ StreamSession with full trace
```

The caller supplies the `coherence_callback` — typically a closure that accumulates tokens, calls `scorer.review()`, and returns the score. The kernel only decides *when to halt*; the caller decides *how to score*.

## How NLI Premise/Hypothesis Are Constructed

When `use_nli=True` on the scorer:

1. **Premise** = the prompt text, or the top-k retrieval results from `GroundTruthStore` / `VectorGroundTruthStore` if a KB is configured.
2. **Hypothesis** = the accumulated response text so far (all tokens joined).
3. The NLI model (DeBERTa) scores the entailment probability P(premise entails hypothesis).
4. The divergence score is `1.0 - entailment_probability` — higher means more hallucination risk.

With chunked NLI (`scorer.score_chunked()`):

- Both premise and hypothesis are split into sentence-level chunks.
- All (premise_chunk, hypothesis_chunk) pairs are scored.
- The final score uses **max aggregation** across pairs — the worst contradiction in any pair drives the score.
- This catches localized hallucinations that would be diluted in a full-text comparison.

## Basic Usage

```python
from director_ai import StreamingKernel

kernel = StreamingKernel(
    hard_limit=0.4,
    window_size=10,
    window_threshold=0.55,
    trend_window=5,
    trend_threshold=0.15,
    soft_limit=0.6,
)

session = kernel.stream_tokens(token_generator, coherence_callback)

if session.halted:
    print(f"Halted: {session.halt_reason}")
    print(f"Safe output: {session.output}")
else:
    print(f"Approved: {session.output}")
```

## on_halt Callback

```python
def handle_halt(session):
    print(f"Halted at token {session.halt_index}")
    # Log, alert, switch to fallback

kernel = StreamingKernel(on_halt=handle_halt)
```

## Async Streaming

```python
from director_ai import AsyncStreamingKernel

kernel = AsyncStreamingKernel(
    hard_limit=0.4,
    on_halt=handle_halt,
    soft_limit=0.6,
)

session = await kernel.stream_to_session(async_token_gen, coherence_fn)
```

## Threshold Tuning by Domain

| Domain | hard_limit | window_threshold | trend_threshold | window_size | Rationale |
|--------|-----------|-----------------|----------------|-------------|-----------|
| General | 0.4 | 0.50 | 0.15 | 10 | Balanced — catches obvious hallucinations without over-halting |
| Medical | 0.5 | 0.60 | 0.10 | 8 | Strict — medical misinformation is high-risk |
| Finance | 0.5 | 0.55 | 0.12 | 8 | Strict — numerical claims must be grounded |
| Legal | 0.45 | 0.55 | 0.12 | 10 | Moderate-strict — citations and precedent matter |
| Creative | 0.3 | 0.40 | 0.20 | 15 | Loose — creative writing tolerates divergence |

See the [Threshold Tuning Guide](threshold-tuning.md) for detailed tuning methodology.

## Debug Mode

Enable `streaming_debug=True` to get per-token diagnostic snapshots:

```python
kernel = StreamingKernel(
    hard_limit=0.4,
    window_size=10,
    streaming_debug=True,
)

session = kernel.stream_tokens(token_gen, coherence_cb)

for snap in session.debug_log:
    print(
        f"token {snap['index']}: "
        f"coherence={snap['coherence']:.3f} "
        f"window_avg={snap['window_avg']:.3f} "
        f"trend_drop={snap['trend_drop']:.3f} "
        f"accumulated={snap['accumulated_tokens']}"
    )
```

Each `TokenEvent` also gets a `debug_info` dict with the same fields. Use this to diagnose why a halt triggered or to tune thresholds on your data.

**Fields in each debug snapshot:**

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Token position in stream |
| `coherence` | float | Raw coherence score for this token |
| `window_avg` | float | Current sliding window average |
| `trend_drop` | float | Coherence delta over trend window |
| `accumulated_tokens` | int | Total tokens processed so far |

## StreamSession Properties

| Property | Type | Description |
|----------|------|-------------|
| `output` | str | Safe partial output (up to halt point) |
| `halted` | bool | Whether generation was stopped |
| `halt_index` | int | Token index where halt occurred |
| `halt_reason` | str | Which mechanism triggered |
| `avg_coherence` | float | Mean coherence across all tokens |
| `min_coherence` | float | Lowest coherence observed |
| `warning_count` | int | Tokens in soft warning zone |
| `duration_ms` | float | Total processing time |
| `debug_log` | list[dict] | Debug snapshots (only when `streaming_debug=True`) |
