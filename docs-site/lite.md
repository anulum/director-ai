# Director-Lite — streaming halt in 3 lines

`director-ai-lite` is a **standalone, dependency-free** package: it stops an LLM
token stream *before* a hallucination finishes generating, with zero heavy
dependencies and no `director-ai` requirement.

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
print(result.output)       # surviving text (halted tokens removed)
print(result.halted)       # True if the stream was stopped
print(result.halt_reason)  # why it was stopped
```

`token_stream` is any iterable of string tokens — wire it straight to your LLM's
streaming response.

!!! info "Priority"
    Director-AI publicly shipped streaming contradiction-halt surfaces in early
    2026 and [deposited the related artefact on Zenodo](https://doi.org/10.5281/zenodo.18822166)
    (March 2026). Treat that as provenance for the mechanism, not as a standalone
    production accuracy claim.

## How it works

The default path is **model-free**. Each accumulated prefix is scored by a
grounding heuristic (content-word overlap against the supplied `facts`) and the
same calibrated coherence combination the full package uses in its no-model path.
The stream hard-halts on the first token whose coherence drops below `threshold`
(default `0.5`). With no `facts`, scoring stays neutral and nothing is halted.

This is great for a first look and runs anywhere, but it is approximate — the
heuristic has no model behind it.

## One call, or a reusable guard

```python
from director_ai_lite import StreamGuard, streaming_guard

# one-shot
result = streaming_guard(token_stream, facts={...}, prompt="...")

# reusable
g = StreamGuard(facts={...}, threshold=0.6)
result = g.guard(token_stream, prompt="...")
text = g.safe_text(token_stream, prompt="...")  # surviving text only
```

## Parameters

| | |
|---|---|
| `facts` | mapping of key → grounded statement for factual scoring |
| `threshold` | coherence floor in `[0, 1]`; the stream halts below it (default `0.5`) |
| `scorer` | optional `review(prompt, text)` scorer (e.g. model-backed NLI) that overrides the heuristic |

## Upgrade to model-backed scoring

The grounding heuristic and coherence calibration match the full package's
no-model path, so upgrading does not change the call site — install the full
package and pass its scorer:

```bash
pip install "director-ai-lite[full]"
```

```python
from director_ai_lite import StreamGuard

g = StreamGuard(facts={...}, scorer=my_nli_scorer)  # any review(prompt, text) scorer
```

## Tiers

Director-Lite is the free, standalone entry point. The wider product ladder:

| Tier | What it is |
|---|---|
| **Director-Lite** | **Free.** This package — standalone, model-free streaming halt, zero dependencies. |
| **Director-AI** | The full runtime — model-backed NLI/RAG scoring, REST/gRPC server, framework integrations, sealed evidence packets, tamper-evident audit. |
| **Director-AI Pro** | Production-tier licence and support on top of the full runtime. |
| **Director-AI Full** | The complete advanced + labs capability set. |
| **Director-Class AI** | Enterprise: managed/on-prem deployment, domain tuning, evidence reviews, SLA, procurement support. |

License: **Apache-2.0**.
