# Standalone Interlock Kernel

`director_ai.interlock` packages the halt/interlock kernel as a lightweight
bring-your-own-scorer library surface. It does not load retrieval, NLI, vector,
server, or training dependencies. The caller supplies a scorer callable, and the
kernel decides whether candidate tokens may be admitted.

Use it when another runtime already owns generation and scoring, but still needs
Director-compatible halt decisions and tenant-safe `SafetyEvent` records.

```python
from director_ai import InterlockKernel, InterlockPolicy

kernel = InterlockKernel(
    InterlockPolicy(
        hard_limit=0.5,
        window_size=4,
        window_threshold=0.55,
        hook_id="gateway.interlock",
        policy_id="policy.gateway.regulated",
    )
)

result = kernel.run(
    token_stream,
    scorer=lambda candidate_text: external_score(candidate_text),
)

if result.decision == "halt":
    audit(result.halt_event)
    return result.output
```

## Guarantees

- low-score tokens are scored before they are appended to `output`
- hard-limit, sliding-window, and optional downward-trend checks are available
- `warn_only=True` keeps the stream running while still emitting a warning event
- scorer results can be floats or objects with a `.score` attribute
- scores must be finite and in `[0, 1]`
- halt and warning events use evidence references such as
  `interlock://token/4`, not rejected token text
- the module is importable as `director_ai.interlock` and from the root package

## Full API

::: director_ai.interlock.InterlockPolicy

::: director_ai.interlock.InterlockDecision

::: director_ai.interlock.InterlockKernel
