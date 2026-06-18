# Director-Lite — streaming halt in 3 lines

Director-AI's one differentiator is the **token-level streaming halt**: it stops
a model's output *before* a hallucination finishes generating — a seatbelt that
deploys before the crash, not an incident report after it.

`director_ai.lite` is the 30-second front door to exactly that. No model
download, no knowledge-base wiring, no config.

Install the standalone first-run package when the public surface should be this
facade only:

```bash
pip install director-ai-lite
```

```python
from director_ai_lite import guard

result = guard(
    token_stream,
    facts={"capital": "Paris is the capital of France."},
    prompt="What is the capital of France?",
)
```

The standalone package delegates to the same `director_ai.lite` implementation,
so the full and Lite installs share halt semantics, type behavior, and upgrade
paths.

!!! info "Priority"
    Director-AI publicly shipped token-level streaming halt in early 2026 and
    [deposited it on Zenodo](https://doi.org/10.5281/zenodo.18822166) (March
    2026) — ahead of the streaming-guardrail research literature.

```python
from director_ai import StreamGuard

guard = StreamGuard(facts={"capital": "Paris is the capital of France."})

result = guard.guard(token_stream, prompt="What is the capital of France?")
print(result.output)       # surviving text (halted tokens removed)
print(result.halted)       # True if the stream was stopped
print(result.halt_reason)  # e.g. "hard_limit (0.30 < 0.50)"
```

`token_stream` is any iterable of string tokens — wire it straight to your LLM's
streaming response.

## One call

```python
from director_ai import streaming_guard

result = streaming_guard(
    token_stream,
    facts={"capital": "Paris is the capital of France."},
    prompt="What is the capital of France?",
)
```

## How it relates to the full product

`StreamGuard` wraps the same Apache-2.0 `StreamingKernel` and `CoherenceScorer`
the full Director-AI uses. By default it runs in **heuristic mode** (no NLI
model) so it works anywhere with zero install — great for a first look, but
approximate.

To upgrade to model-backed accuracy, install the NLI extra and pass a configured
scorer; the call site does not change:

```python
pip install "director-ai[nli]"
```

```python
from director_ai import CoherenceScorer, GroundTruthStore, StreamGuard

store = GroundTruthStore()
store.add("capital", "Paris is the capital of France.")
scorer = CoherenceScorer(ground_truth_store=store)  # loads the NLI model

guard = StreamGuard(scorer=scorer, threshold=0.6)
```

From there, the rest of Director-AI — RAG grounding, the sealed evidence packet,
the tamper-evident audit chain, multi-tenant isolation, the REST/gRPC server, and
the [CI quality gate](guide/ci-gate.md) — is one import away when you need it.

## Parameters

| | |
|---|---|
| `facts` | mapping of key → grounded statement for factual scoring |
| `threshold` | coherence floor; the stream halts below it (default `0.5`) |
| `window_size` | sliding-window length for trend-based halting (default `10`) |
| `scorer` | optional pre-built scorer (e.g. model-backed NLI) — overrides the heuristic default |
