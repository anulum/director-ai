<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-AI — CoherenceAgent API
-->

# CoherenceAgent

Integrated orchestrator that combines a generator, scorer, ground truth store, and safety kernel into a single `process()` call. Handles candidate generation, scoring, fallback, and output interlock.

## Usage

```python
from director_ai import CoherenceAgent

agent = CoherenceAgent(
    use_nli=True,
    fallback="retrieval",
)

result = agent.process("What is the refund policy?")
print(result.output)
print(result.coherence.score if result.coherence else None)
```

Without `llm_api_url` or `provider`, the agent uses its built-in demonstration
generator. Set `production_mode=True` with an explicit real generator for
deployment; the constructor rejects production mode without one.

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_api_url` | `str \| None` | `None` | Direct URL to OpenAI-compatible endpoint |
| `use_nli` | `bool \| None` | `None` | `True` configures the model scoring path; otherwise prefer native heuristics when available |
| `provider` | `str \| None` | `None` | `"openai"` or `"anthropic"` (reads API key from env) |
| `fallback` | `str \| None` | `None` | Fallback mode: `"retrieval"`, `"disclaimer"`, `None` |
| `disclaimer_prefix` | `str` | `"[Unverified] "` | Prefix for disclaimer fallback text |

!!! note "Mutual exclusivity"
    `llm_api_url` and `provider` are mutually exclusive. Use one or the other.

## Scorer selection and tenants

The agent uses the high-level `CoherenceScorer` API even when Rust is
available. Its native path therefore preserves per-call tenant grounding and
the Python score result contract. An explicit `use_nli=True` request selects
the model scoring path rather than replacing it with native heuristics.
This configures NLI; it does not prove that model weights are available or
establish prediction quality. See the [scorer model policies](scorer.md).

Pass the authenticated `tenant_id` to `process()`, `aprocess()` or `stream()`.
Both streaming halt modes use the request's tenant for grounding. An empty
identity selects unscoped behaviour. Custom `ContradictionHalt` instances must
provide a `retrieve_scoped(claim, tenant_id)` callback for tenant-scoped calls.
Callers authenticate tenant identity and receive scoped
scoring errors directly.

## Methods

### process()

```python
result = agent.process(prompt, tenant_id="example")
```

Generate candidates, score them, return the best approved response (or fallback).

### aprocess()

```python
result = await agent.aprocess(prompt, tenant_id="example")
```

Async variant of `process()`.

### stream()

```python
async for token, score in agent.stream(prompt, tenant_id="example"):
    print(token, end="")
```

Async streaming with real-time coherence monitoring. Yields `(token, coherence_score)` tuples.
Each call on the coherence-based path maintains its own sliding window and trend
history, including when multiple calls share an agent.

The tuple iterator includes the token that triggered a halt. To make a
pre-delivery decision, iterate over `agent.stream_events(prompt, tenant_id=...)`
and check each event's `halt_reason` before delivering its token. A
whole-response rejection reports a halt with no token.

For an external token generator, the agent calls the returned async iterator's
`aclose()` when that method is available. This applies on completion, on a
halt, and when the caller closes the agent iterator early.

## Fallback Modes

| Mode | Behavior |
|------|----------|
| `None` | Reject if all candidates fail — returns a halt message with `halted=True` |
| `"retrieval"` | Return KB context when all candidates fail |
| `"disclaimer"` | Prepend warning to the best-rejected candidate |

A candidate rejection keeps the selected rejection score. If that score came
from the scalar cache, the agent reviews the selected candidate again for the
same tenant to attach current grounding evidence.

The final output monitor can also halt an approved candidate. Both
`process()` and `aprocess()` set `halted=True` and put the monitor reason in
`halt_evidence.reason`. The response keeps its established halt message, and
`stream_events()` reports the same reason in a terminal event with no token.
After an emergency stop, subsequent calls report `kernel_inactive` until the
monitor is re-armed.

```python
# Retrieval fallback
agent = CoherenceAgent(fallback="retrieval")
result = agent.process("What is the refund policy?")
if result.halted:
    print("Fell back to KB retrieval")

# Disclaimer fallback
agent = CoherenceAgent(fallback="disclaimer")
result = agent.process("What is the refund policy?")
# Response prefixed with "[Unverified] ..."
```

## Full API

::: director_ai.core.agent.CoherenceAgent
