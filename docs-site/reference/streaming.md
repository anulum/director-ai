# StreamingKernel

Token-by-token safety kernel with sliding-window coherence monitoring.

## Constructor

```python
from director_ai.core.streaming import StreamingKernel

kernel = StreamingKernel(
    hard_limit=0.5,         # absolute coherence floor (immediate halt)
    window_size=10,         # sliding window token count
    window_threshold=0.55,  # halt when window average drops below this
    trend_window=5,         # tokens checked for downward trend
    trend_threshold=0.15,   # halt when coherence drops this much over trend_window
    on_halt=None,           # callable(StreamSession) on halt
    soft_limit=0.6,         # warning zone floor
    halt_mode="hard",       # "hard" or "soft"
)
```

## Halt Modes

**hard** -- truncates at the exact violating token. Always used for `hard_limit` breaches.
**soft** -- continues until sentence boundary (`.!?`) or 50-token cap. Sets `soft_halted=True`.

## `stream_tokens(token_generator, coherence_callback, ...) -> StreamSession`

```python
session = kernel.stream_tokens(
    token_generator,          # iterable[str]
    coherence_callback,       # callable(str) -> float
    evidence_callback=None,   # callable(str) -> str | None (halt only)
    scorer=None,              # CoherenceScorer for structured HaltEvidence
    top_k=3,                  # evidence chunks on halt
)
```

## TokenEvent

| Field | Type | Description |
|-------|------|-------------|
| `token` | `str` | Token text |
| `index` | `int` | Zero-based position |
| `coherence` | `float` | Score at this token |
| `timestamp` | `float` | `time.monotonic()` value |
| `halted` | `bool` | This token triggered halt |
| `warning` | `bool` | Coherence < soft_limit |

## StreamSession

| Property | Type | Description |
|----------|------|-------------|
| `output` | `str` | Joined tokens (truncated at halt point if halted) |
| `halted` | `bool` | Any halt occurred |
| `soft_halted` | `bool` | Halt used soft mode |
| `halt_index` | `int` | Token index of halt (-1 if none) |
| `halt_reason` | `str` | e.g. `"hard_limit (0.32 < 0.5)"` |
| `avg_coherence` | `float` | Mean coherence |
| `min_coherence` | `float` | Lowest coherence |
| `warning_count` | `int` | Tokens with warning |
| `duration_ms` | `float` | Wall-clock time |

## Example

```python
from director_ai import CoherenceScorer
from director_ai.core.streaming import StreamingKernel

scorer = CoherenceScorer(threshold=0.5, use_nli=True)
kernel = StreamingKernel(hard_limit=0.4, halt_mode="soft")

acc = []
def cb(tok):
    acc.append(tok)
    return scorer.review("capital of France", " ".join(acc))[1].score

session = kernel.stream_tokens(iter(["The", "capital", "is", "Paris", "."]), cb)
print(session.output, session.halted, f"{session.avg_coherence:.3f}")
```
