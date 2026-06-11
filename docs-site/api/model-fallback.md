# Fallback Model Registry

`nli_model` is a configurable, revision-pinned default, but a single hardcoded id
has no recourse if that repository is delisted or unreachable on the Hugging Face
Hub — the NLI scorer then drops all the way to its word-overlap heuristic. The
fallback registry keeps an ordered chain of vetted, revision-pinned alternates
and resolves the primary to the first **available** model in the chain, so a
deployment degrades to a strong alternate model instead of the heuristic floor.

## Built-in NLI chain

The `nli` role's alternates are MNLI-style sequence classifiers the DeBERTa
backend loads directly, each already pinned in the
[revision registry](nli.md):

1. `yaxili96/FactCG-DeBERTa-v3-Large` (the configured primary)
2. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
3. `roberta-large-mnli`

Embedding and reranker fallbacks are intentionally omitted: a different-dimension
embedding model is not a drop-in replacement (it invalidates an existing index),
so that swap is a deliberate re-indexing decision, not an automatic failover.

## Enabling

Off by default — it probes the Hub at startup, so it is opt-in:

```python
from director_ai.core.config import DirectorConfig

scorer = DirectorConfig(model_fallback_enabled=True).build_scorer()
# If FactCG is delisted, the scorer is built on the first reachable alternate
# rather than falling through to the heuristic.
```

## Resolution

`FallbackModelRegistry.resolve(role, primary)` tries the primary first, then each
vetted fallback, and returns the first whose availability probe passes. If none
is reachable it returns the primary unchanged, letting the scorer's own heuristic
floor take over — it never raises on a missing model.

Availability is decided by an injected `AvailabilityProbe` (default: a cheap
Hugging Face `model_info` metadata call, no weights downloaded), so resolution is
deterministic and fully tested offline. Probe results are cached per model for
the registry's lifetime. Every chain entry must be revision-pinned, enforced at
construction.

```python
from director_ai.core.model_registry import FallbackModelRegistry

registry = FallbackModelRegistry()
resolved = registry.resolve("nli", "yaxili96/FactCG-DeBERTa-v3-Large")
print(resolved.model_id, resolved.revision, resolved.is_fallback)
```

## Full API

::: director_ai.core.model_registry.FallbackModelRegistry

::: director_ai.core.model_registry.ResolvedModel
