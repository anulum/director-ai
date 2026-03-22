# Exceptions

Director-AI exception hierarchy. All exceptions inherit from `DirectorAIError`.

```
DirectorAIError
├── HallucinationError     # guard() coherence failure
├── KernelHaltError        # SafetyKernel stream halt
├── CoherenceError         # Scoring computation failure
├── GeneratorError         # LLM generation failure
├── ValidationError        # Invalid configuration or input
├── DependencyError        # Missing optional package
└── PhysicsError           # Numerical instability
    └── NumericalError     # NaN/Inf in computation
```

## HallucinationError

Raised by `guard()` with `on_fail="raise"` when coherence drops below threshold.

```python
from director_ai import guard, HallucinationError
from openai import OpenAI

client = guard(OpenAI(), facts={"policy": "30-day refund"})

try:
    response = client.chat.completions.create(...)
except HallucinationError as e:
    print(f"Query: {e.query}")
    print(f"Response: {e.response[:100]}")
    print(f"Score: {e.score.score:.3f}")
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `query` | `str` | The user prompt |
| `response` | `str` | The rejected LLM response |
| `score` | `CoherenceScore` | Full score details |

---

## KernelHaltError {: #kernelhalterror }

Raised when `SafetyKernel` halts the output stream due to coherence dropping below `hard_limit`.

---

## CoherenceError

Raised when the scoring computation itself fails (model error, corrupt input).

---

## ValidationError {: #validationerror }

Raised for invalid configuration values.

```python
from director_ai import CoherenceScorer

try:
    scorer = CoherenceScorer(threshold=1.5)  # Invalid
except ValueError as e:
    print(e)  # "threshold must be in [0, 1], got 1.5"
```

---

## DependencyError {: #dependencyerror }

Available for optional-dependency checks. Note: `CoherenceScorer(use_nli=True)` falls back silently to heuristic scoring when NLI packages are missing. Use `strict_mode=True` if you need hard failures on missing dependencies.

```python
from director_ai.core.exceptions import DependencyError
```

---

## GeneratorError

Raised when the LLM provider returns an error during candidate generation.

---

## Catching All Director-AI Errors

```python
from director_ai import DirectorAIError

try:
    result = agent.process(query)
except DirectorAIError as e:
    logger.error("Director-AI error: %s", e)
    # Fallback to unguarded response
```

## Full API

::: director_ai.core.exceptions.DirectorAIError

::: director_ai.core.exceptions.HallucinationError

::: director_ai.core.exceptions.KernelHaltError

::: director_ai.core.exceptions.CoherenceError

::: director_ai.core.exceptions.ValidationError

::: director_ai.core.exceptions.DependencyError

::: director_ai.core.exceptions.GeneratorError
