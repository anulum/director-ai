# Streaming Halt

Post-hoc review catches the hallucination too late — by the time
the user reads paragraph three, the fabricated number is already
cached in their summary. Streaming halt severs the token stream
the moment coherence crosses a floor. This guide walks through
the three halt mechanisms the kernel ships with, how to observe
them, and where to put the ceiling for different deployment
profiles.

## Minimal reproduction

```python
from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.runtime.streaming import StreamingKernel

store = GroundTruthStore()
store.add("revenue_2025", "ANULUM reported CHF 4.2M in FY2025.")
scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

kernel = StreamingKernel(
    hard_limit=0.4,         # any token below this halts immediately
    window_size=4,           # sliding window over recent coherence
    window_threshold=0.5,    # halt when window average dips
    trend_window=4,           # consider the last N tokens for trend
    trend_threshold=0.25,    # halt when coherence drops more than this
)

def coherence(accumulated: str) -> float:
    _, score = scorer.review(
        prompt="Summarise ANULUM's 2025 performance.",
        action=accumulated,
    )
    return score.score

tokens = [
    "ANULUM", " reported", " CHF", " 4.2M",
    " in", " 2025.", " It", " is", " expected",
    " to", " double", " next", " year",  # drift starts
    " and", " reach", " CHF", " 20M",    # fabricated projection
    ".",
]
session = kernel.stream_tokens(iter(tokens), coherence_callback=coherence)

if session.halted:
    print("halted:", session.halt_reason)
    print("prefix:", "".join(session.tokens))
```

One of three things happens:

* `hard_limit` — one token's coherence drops below 0.4 → halt.
* `window_avg` — the four-token average dips below 0.5 → halt.
* `downward_trend` — coherence fell by more than 0.25 over the
  last four tokens → halt.

## Which halt triggers when

| Halt mechanism | Catches | Misses |
| --- | --- | --- |
| `hard_limit` | Abrupt fabrications ("…instead, the answer is 42") | Gradual drift — each token is individually plausible |
| `window_avg` | Short-burst hallucinations — two or three consecutive off-source tokens | One borderline token in otherwise clean output |
| `downward_trend` | Slow drift into speculation, where no single token crosses the floor but the whole passage walks off-source | Flat but wrong responses — coherence stays stable at a wrong value |

The three are complementary; keeping all of them on gives the
broadest catch. Tune the thresholds per deployment:

* **Medical / legal / finance** — lower the hard limit (0.55+) and
  shrink the window (3 tokens) so even one off-source token halts.
* **Creative writing / general chat** — raise the hard limit (0.3)
  and widen the trend window (8 tokens) so the kernel tolerates
  normal stylistic variation.
* **Code generation** — defer to `director_ai.core.verification`
  (AST + type check) rather than coherence scores; the kernel
  still catches policy violations but the scoring channel is not
  the primary signal.

## Live token observation

`StreamingKernel.stream_tokens` accepts a list of
`TokenTraceCallback` objects (see
`director_ai.core.observability`). Every emitted token is fanned
out to every callback:

```python
from director_ai.core.observability import (
    LangfuseTokenCallback,
    TokenTraceCallback,
    TokenTraceEvent,
)


class PrintCallback(TokenTraceCallback):
    def on_token(self, event: TokenTraceEvent) -> None:
        print(f"{event.index:02d} {event.token!r:>12} score={event.coherence:.3f}")


session = kernel.stream_tokens(
    iter(tokens),
    coherence_callback=coherence,
    trace_callbacks=[PrintCallback()],
    tenant_id="acme-corp",
    request_id="req-2026-04-17-001",
)
```

The same protocol wires into Langfuse via
`LangfuseTokenCallback(client)` — one trace per `request_id`, one
span and one `coherence` score entry per token, one trace update
at `on_stream_end` with the halt reason. Token text defaults to a
truncated SHA-256 hash; pass `record_token_text=True` when the
deployment has a legal basis for storing raw assistant output.

## Seeing it in the browser

`demo/streaming_halt_live.py` is a Gradio app that streams one
token at a time with a live coherence gauge. Three scenarios
ship: a truthful stream, an abrupt hard-halt case, and a gradual
drift trend-halt. Run locally:

```bash
pip install -e ".[demo]"
python demo/streaming_halt_live.py
```

The app uses the same `TokenTraceCallback` hook — the demo's
queue-based sink is a concrete example of pushing tokens to a UI
without stalling the scoring loop.

## Edge runtime

`backfire-kernel/crates/backfire-wasm` builds the halt check as a
`~110 KB` WebAssembly module that any browser, Cloudflare Worker,
or Deno runtime can load. The WASM path covers the halt decision
only; the host must supply a coherence score per token (typically
a quantised Transformers.js model running in a Web Worker). Build
and run the browser example:

```bash
make wasm-build
cd backfire-kernel/crates/backfire-wasm/example
python3 -m http.server 8000
```

## Audit and forensics

`StreamSession` carries the full token trace after the stream
ends. The `.events` list has one `TokenEvent` per emitted token
including its coherence, timestamp, and any halt evidence; the
`halt_evidence_structured` attribute (when a `CoherenceScorer` is
passed in) lists the top-K contradicting source chunks with their
NLI scores. This is the artefact to hand to an incident reviewer —
it shows *which* token failed and *which* source document
contradicted it.

## Operating notes

* **Scoring every token is the slow path.** `score_every_n=4` with
  `adaptive=True` keeps the cadence low during high-coherence
  sections and accelerates to per-token when coherence dips.
* **Always keep the hard limit.** Lower to 0.2 if you tolerate
  borderline outputs, but do not disable it — a single
  low-coherence token is the strongest signal you have.
* **Benchmark on your text.** The AggreFact per-dataset mean
  (75.6% BA for FactCG at threshold 0.5) is a baseline; your
  retrieval quality and your dialogue style dominate the numbers
  more than any single parameter.
* **Pair with the gateway risk router.** The Go middleware in
  `gateway/go/internal/risk` refuses obvious attacks before any
  streaming happens, so the streaming kernel only sees prompts
  that cleared the gateway's budget check.

## See also

- [`docs-site/cookbook/long-context-rag-drift.md`](./long-context-rag-drift.md)
  — chunked NLI strategies when a response is long enough that the
  three halt mechanisms cannot localise the drift on their own.
- [`docs-site/cookbook/multi-agent-handoff-failures.md`](./multi-agent-handoff-failures.md)
  — catching the same drift across agent boundaries with
  `HandoffScorer` + `SwarmGuardian`.
- `docs/BENCHMARKS.md` — tiered scorer latency and AggreFact
  per-dataset breakdown.
