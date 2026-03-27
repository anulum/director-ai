# DSPy / Instructor Integration

*Added in v3.11.0*

Director-AI provides assertion and validation functions for [DSPy](https://dspy-docs.vercel.app/) modules and [Instructor](https://python.useinstructor.com/) structured output pipelines.

## DSPy Assertion

Use `director_assert()` inside a `dspy.Module.forward()` to enforce factual grounding:

```python
import dspy
from director_ai.integrations.dspy import director_assert

class FactCheckedQA(dspy.Module):
    def forward(self, question):
        answer = self.generate(question=question)
        director_assert(
            answer.response,
            facts={"pricing": "Team plan costs $19/user/month."},
        )
        return answer
```

Raises `HallucinationError` if coherence is below threshold.

## Standalone Validation

`coherence_check()` works with any pipeline — Instructor, plain Python, or any framework:

```python
from director_ai.integrations.dspy import coherence_check

result = coherence_check(
    response="The team plan costs $29/month.",
    facts={"pricing": "Team plan costs $19/user/month."},
)
if not result["approved"]:
    print(f"Hallucination: score={result['score']:.3f}")
```

## API Reference

### `coherence_check(response, prompt, facts, store, threshold, use_nli) -> dict`

Returns `{"approved": bool, "score": float, "evidence": ...}`.

### `director_assert(response, prompt, facts, store, threshold, use_nli, message) -> None`

Raises `HallucinationError` if coherence is below threshold. Silent on pass.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response` | `str` | required | LLM output to verify |
| `prompt` | `str` | `""` | Original prompt for context |
| `facts` | `dict[str, str]` | `None` | Key-value facts for grounding |
| `store` | `GroundTruthStore` | `None` | Pre-built store (overrides facts) |
| `threshold` | `float` | `0.5` | Minimum coherence to pass |
| `use_nli` | `bool | None` | `None` | NLI mode |
