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

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_api_url` | `str \| None` | `None` | Direct URL to OpenAI-compatible endpoint |
| `use_nli` | `bool \| None` | `None` | Enable NLI model scoring |
| `provider` | `str \| None` | `None` | `"openai"` or `"anthropic"` (reads API key from env) |
| `fallback` | `str \| None` | `None` | Fallback mode: `"retrieval"`, `"disclaimer"`, `None` |
| `disclaimer_prefix` | `str` | `"[Unverified] "` | Prefix for disclaimer fallback text |

!!! note "Mutual exclusivity"
    `llm_api_url` and `provider` are mutually exclusive. Use one or the other.

## Methods

### process()

```python
result = agent.process(query: str) -> ReviewResult
```

Generate candidates, score them, return the best approved response (or fallback).

### aprocess()

```python
result = await agent.aprocess(query: str) -> ReviewResult
```

Async variant of `process()`.

### stream()

```python
async for token, score in agent.stream(prompt: str, tenant_id: str = ""):
    print(token, end="")
```

Async streaming with real-time coherence monitoring. Yields `(token, coherence_score)` tuples.

## Fallback Modes

| Mode | Behavior |
|------|----------|
| `None` | Reject if all candidates fail — returns empty response with `halted=True` |
| `"retrieval"` | Return KB context when all candidates fail |
| `"disclaimer"` | Prepend warning to the best-rejected candidate |

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
