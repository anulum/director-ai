# Guardrails AI

Director-AI provides a custom Guardrails AI validator that checks generated text
with the same coherence scorer used by the rest of the runtime.

```bash
pip install "director-ai[guardrails]" guardrails-ai
```

The `guardrails` extra is a stable documentation anchor. The adapter imports
Guardrails AI only when used, so install `guardrails-ai` from a package index
that provides a compatible build for your Python runtime.

```python
from guardrails import Guard
from director_ai.integrations.guardrails_ai import attach_guardrails_validator

guard = attach_guardrails_validator(
    Guard(),
    facts={"sla": "Support SLA is four hours."},
    threshold=0.4,
    use_nli=False,
)

outcome = guard.parse(
    "The support SLA is four hours.",
    metadata={
        "prompt": "What is the support SLA?",
        "tenant_id": "tenant-a",
    },
)
```

## Validator Class

Use `build_guardrails_validator_class()` when the validator must be registered
before constructing a Guardrails spec:

```python
from guardrails import Guard
from director_ai.integrations.guardrails_ai import (
    build_guardrails_validator_class,
)

DirectorAICoherenceValidator = build_guardrails_validator_class()

guard = Guard().use(
    DirectorAICoherenceValidator(
        facts={"refunds": "Refunds are available within 30 days."},
        threshold=0.5,
    )
)
```

## Metadata Contract

The validator accepts tenant-safe metadata:

| Field | Purpose |
|-------|---------|
| `prompt`, `query`, `question`, or `input` | Original user request used as scorer context |
| `messages` | Chat message list; the latest user message is used |
| `tenant_id` or `tenant` | Optional tenant id passed to the scorer |

Failure messages do not include the raw model output. Score details are returned
under `metadata["director_ai"]` on the Guardrails validation result.

::: director_ai.integrations.guardrails_ai.build_guardrails_validator_class

::: director_ai.integrations.guardrails_ai.build_guardrails_validator

::: director_ai.integrations.guardrails_ai.attach_guardrails_validator
