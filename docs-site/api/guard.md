# guard / score / get_score

The three top-level entry points for adding hallucination guardrails to any LLM application.

## guard()

Wrap an existing LLM SDK client with transparent coherence scoring. Every call to `client.chat.completions.create()` (or equivalent) is automatically scored against your knowledge base.

```python
from director_ai import guard
from openai import OpenAI

client = guard(
    OpenAI(),
    facts={"refund": "within 30 days", "hours": "9am-5pm EST"},
    threshold=0.3,
    on_fail="raise",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the refund policy?"}],
)
```

### Supported SDK Shapes

| SDK | Detection | Method Patched |
|-----|-----------|----------------|
| **OpenAI** (+ vLLM, Groq, LiteLLM, Ollama, Together) | `client.chat.completions.create` | `chat.completions.create()` |
| **Anthropic** | `client.messages.create` (no `client.chat`) | `messages.create()` |
| **AWS Bedrock** | `client.converse` | `converse()` / `converse_stream()` |
| **Google Gemini** | `client.generate_content` | `generate_content()` |
| **Cohere** | `client.chat` (no `client.completions`) | `chat()` |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | any | required | SDK client instance |
| `facts` | `dict[str, str]` | `None` | Key-value fact pairs for grounding |
| `store` | `GroundTruthStore` | `None` | Pre-built fact store (overrides `facts`) |
| `threshold` | `float` | `0.3` | Minimum coherence score to approve |
| `use_nli` | `bool \| None` | `None` | Force NLI on/off; `None` = auto-detect |
| `on_fail` | `str` | `"raise"` | Failure mode (see below) |

### Failure Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `"raise"` | Raises `HallucinationError` | Production — block hallucinations |
| `"log"` | Logs warning, returns response unchanged | Development — observe without blocking |
| `"metadata"` | Stores score in context var, returns response | Async pipelines — check score later |

### Streaming

Streaming calls are automatically guarded with periodic coherence checks every 8 tokens:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain our refund policy"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

If coherence drops during streaming, `HallucinationError` is raised (with `on_fail="raise"`).

---

## score()

Score a single prompt/response pair without wrapping an SDK client.

```python
from director_ai import score

result = score(
    "What is the capital of France?",
    "The capital of France is Berlin.",
    facts={"capital": "Paris is the capital of France."},
)

print(f"Score: {result.score:.3f}")      # ~0.35
print(f"Approved: {result.approved}")    # False
print(f"H_logical: {result.h_logical:.3f}")
print(f"H_factual: {result.h_factual:.3f}")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | required | User query |
| `response` | `str` | required | LLM response to verify |
| `facts` | `dict[str, str]` | `None` | Fact pairs for grounding |
| `store` | `GroundTruthStore` | `None` | Pre-built store |
| `threshold` | `float` | `0.3` | Approval threshold |
| `use_nli` | `bool \| None` | `None` | Force NLI on/off |
| `profile` | `str \| None` | `None` | Named config profile (e.g. `"medical"`, `"fast"`) |

### Returns

[`CoherenceScore`](types.md) — contains `score`, `approved`, `h_logical`, `h_factual`, `evidence`, `warning`.

---

## get_score()

Retrieve the last coherence score stored by `guard()` with `on_fail="metadata"`.

```python
from director_ai import guard, get_score
from openai import OpenAI

client = guard(OpenAI(), facts={...}, on_fail="metadata")
response = client.chat.completions.create(...)

last_score = get_score()
if last_score and not last_score.approved:
    print(f"Low coherence: {last_score.score:.3f}")
```

### Returns

`CoherenceScore | None` — the most recent score from the current thread/async context, or `None` if no scoring has occurred.

!!! note "Thread safety"
    `get_score()` uses a `ContextVar`, so scores are isolated per-thread and per-async-task.

---

## Full API

::: director_ai.integrations.sdk_guard.guard

::: director_ai.integrations.sdk_guard.score

::: director_ai.integrations.sdk_guard.get_score
