# Semantic Kernel Integration

*Added in v3.11.0*

Director-AI provides a function invocation filter for [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/).

## Setup

```python
from semantic_kernel import Kernel
from director_ai.integrations.semantic_kernel import DirectorAIFilter

kernel = Kernel()
kernel.add_filter("function_invocation", DirectorAIFilter(
    facts={"pricing": "Team plan costs $19/user/month."},
    threshold=0.5,
))
```

## How It Works

The filter runs after each function invocation. If the LLM output coherence is below the threshold, it either raises `HallucinationError` (default) or annotates the result with `approved=False`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `facts` | `dict[str, str]` | `None` | Key-value facts for the knowledge base |
| `store` | `GroundTruthStore` | `None` | Pre-built store (overrides facts) |
| `threshold` | `float` | `0.5` | Minimum coherence to pass |
| `use_nli` | `bool | None` | `None` | NLI mode (None=auto-detect) |
| `raise_on_fail` | `bool` | `True` | Raise on failure vs annotate result |

## Non-Raising Mode

```python
filter = DirectorAIFilter(
    facts={"pricing": "Team plan costs $19/user/month."},
    raise_on_fail=False,
)
# Result will be a dict with approved=False instead of raising
```
