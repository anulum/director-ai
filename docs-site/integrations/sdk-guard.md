# OpenAI / Anthropic SDK Guard

2-line integration that wraps your existing SDK client with coherence scoring.

## OpenAI

```python
from director_ai import guard
from openai import OpenAI

client = guard(
    OpenAI(),
    facts={"refund": "within 30 days", "hours": "9am-5pm EST"},
    threshold=0.6,
    on_fail="raise",  # "raise" | "log" | "metadata"
)

# Works exactly like normal — hallucinations are caught transparently
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the refund policy?"}],
)
```

## Anthropic

```python
from director_ai import guard
import anthropic

client = guard(
    anthropic.Anthropic(),
    facts={"refund": "within 30 days"},
)

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is the refund policy?"}],
)
```

## Failure Modes

| Mode | Behavior |
|------|----------|
| `on_fail="raise"` | Raises `HallucinationError` |
| `on_fail="log"` | Logs warning, returns response |
| `on_fail="metadata"` | Stores score in context var, returns response |

## Retrieving Scores

```python
from director_ai import guard, get_score

client = guard(OpenAI(), facts={...}, on_fail="metadata")
response = client.chat.completions.create(...)

score = get_score()
if score and not score.approved:
    print(f"Low coherence: {score.score:.3f}")
```

## Streaming Support

Streaming is automatically guarded with periodic coherence checks every 8 tokens:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    stream=True,
)
for chunk in stream:  # Raises if coherence drops
    print(chunk.choices[0].delta.content, end="")
```
