# Streaming API

Token-by-token streaming oversight. `StreamingKernel` monitors coherence on every token
(or every N-th token) and halts generation when it degrades below threshold.

See also: [Streaming Reference](../reference/streaming.md) for halt modes, cadence tuning, and examples.

::: director_ai.core.streaming.StreamingKernel

::: director_ai.core.streaming.StreamSession

::: director_ai.core.streaming.TokenEvent

::: director_ai.core.async_streaming.AsyncStreamingKernel
