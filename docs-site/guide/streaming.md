# Streaming Halt

## Overview

`StreamingKernel` monitors coherence token-by-token and halts generation the moment quality degrades. Three independent halt mechanisms:

1. **Hard limit** — any single token below threshold
2. **Sliding window** — rolling average drops below window threshold
3. **Downward trend** — coherence drops by more than a delta over N tokens

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
