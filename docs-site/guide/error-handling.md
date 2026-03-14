# Error Handling

How to handle failures from Director-AI in production applications.

## Exception Hierarchy

```
DirectorAIError
├── HallucinationError     # guard() detected low coherence
├── KernelHaltError        # SafetyKernel stopped the stream
├── CoherenceError         # Scoring computation failure
├── GeneratorError         # LLM provider error
├── ValidationError        # Bad configuration
├── DependencyError        # Missing optional package
├── PhysicsError           # Numerical instability
└── NumericalError         # NaN/Inf detected
```

## guard() Failure Modes

The `on_fail` parameter controls what happens when `guard()` detects a hallucination:

=== "Raise (default)"

    ```python
    from director_ai import guard, HallucinationError

    client = guard(OpenAI(), facts=facts, on_fail="raise")

    try:
        response = client.chat.completions.create(...)
    except HallucinationError as e:
        print(f"Blocked: {e.score.score:.3f}")
        # Return fallback response to user
    ```

=== "Log"

    ```python
    client = guard(OpenAI(), facts=facts, on_fail="log")

    # Never raises — logs warning and returns response unchanged
    response = client.chat.completions.create(...)
    ```

=== "Metadata"

    ```python
    from director_ai import guard, get_score

    client = guard(OpenAI(), facts=facts, on_fail="metadata")
    response = client.chat.completions.create(...)

    score = get_score()
    if score and not score.approved:
        # Handle low-coherence response asynchronously
        flag_for_review(response, score)
    ```

## Scoring Errors

### NLI Model Unavailable

When `use_nli=True` but the model can't load:

```python
from director_ai import CoherenceScorer

# strict_mode=False (default): falls back to heuristic scoring
scorer = CoherenceScorer(use_nli=True, strict_mode=False)

# strict_mode=True: raises if NLI unavailable
scorer = CoherenceScorer(use_nli=True, strict_mode=True)
```

### Missing Dependencies

```python
from director_ai import DependencyError

try:
    scorer = CoherenceScorer(use_nli=True)
except DependencyError:
    # pip install director-ai[nli]
    scorer = CoherenceScorer(use_nli=False)
```

## Streaming Halt Recovery

When `StreamingKernel` halts, the session contains the safe partial output:

```python
session = kernel.stream_tokens(token_gen, score_fn)

if session.halted:
    safe_output = session.output  # text up to halt point
    reason = session.halt_reason  # "hard_limit", "window_avg", "trend_drop"

    # Option 1: Return partial output
    return safe_output

    # Option 2: Retry with stricter KB
    return retry_with_kb(session.halt_evidence_structured)

    # Option 3: Return KB context directly
    return store.retrieve_context(query)
```

## Production Error Handling Pattern

```python
from director_ai import (
    guard, get_score,
    DirectorAIError, HallucinationError, DependencyError,
)

def guarded_llm_call(client, messages, facts):
    try:
        guarded = guard(client, facts=facts, on_fail="raise")
        return guarded.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
    except HallucinationError as e:
        logger.warning("Hallucination blocked (%.3f): %s", e.score.score, e.response[:100])
        return fallback_response(e.query, facts)
    except DependencyError:
        logger.error("NLI model unavailable — serving unguarded")
        return client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    except DirectorAIError as e:
        logger.error("Director-AI error: %s", e)
        return client.chat.completions.create(model="gpt-4o-mini", messages=messages)
```

## Validation Errors

Configuration validation happens at construction time:

```python
from director_ai import CoherenceScorer, ValidationError

try:
    scorer = CoherenceScorer(threshold=1.5)
except ValueError:
    # "threshold must be in [0, 1], got 1.5"
    pass

try:
    scorer = CoherenceScorer(w_logic=0.5, w_fact=0.3)
except ValueError:
    # "w_logic + w_fact must equal 1.0, got 0.8"
    pass
```

## Logging

Director-AI uses Python's `logging` module under the `DirectorAI` namespace:

```python
import logging

logging.getLogger("DirectorAI").setLevel(logging.DEBUG)

# Or use structured JSON logging
from director_ai.core.config import DirectorConfig
config = DirectorConfig(log_json=True)
```
