# Execution Rings — graduated human authorisation

A prompt-injected agent will try to act. Execution rings bound the blast radius:
every action is classified into an ordered risk ring, and higher rings demand
more **out-of-band human authorisation factors** before the action runs. Even a
fully bypassed guardrail cannot delete data or exfiltrate it without confirmation
a model cannot forge — an operator approval, a cooling period, a second operator,
a CISO notification.

## The rings

| `ExecutionRing` | Example | Required factors (cumulative) |
|-----------------|---------|-------------------------------|
| `READ` (0) | get, list, search | *(none)* |
| `WRITE` (1) | create, update, append | operator approval |
| `DELETE` (2) | delete, drop, purge | + cooling period |
| `EXECUTE` (3) | run, shell, invoke | + second operator |
| `EXFILTRATE` (4) | export, send, upload | + CISO notification |

`classify_operation()` tokenises the operation verb and matches highest-ring
first, so a compound action ("update then **export**") is classified by its most
dangerous part. An unrecognised operation **fails closed to `EXECUTE`** — it is
treated as high-risk, not waved through as a read.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.execution_rings import AuthorizationEvidence

gate = ProductionGuard(DirectorConfig()).execution_rings()  # 24h cooling default

# A read needs nothing.
gate.authorize("list invoices").allowed                    # True

# A delete needs an approval AND a cooling period to elapse.
pending = gate.authorize(
    "delete the customer record",
    AuthorizationEvidence(operator_approval=True, cooling_elapsed_seconds=60),
)
print(pending.allowed)       # False
print(pending.to_dict()["missing"])   # ['cooling_period']
```

`evaluate(ring, evidence)` and `authorize(operation, evidence)` return a
`RingDecision`:

| Field | Meaning |
|-------|---------|
| `ring` | The classified `ExecutionRing`. |
| `allowed` | True only when no required factor is missing. |
| `required` | The factors this ring demands. |
| `satisfied` | The required factors that have been collected. |
| `missing` | The required factors still outstanding. |

`to_dict()` is tenant-safe — ring name and factor sets only, never the action
payload.

## Authorisation factors

`AuthorizationEvidence` is what a caller has actually collected. The reduction to
satisfied factors enforces the real-world rules:

- **cooling period** counts only once an operator approval has started the clock
  and `cooling_elapsed_seconds` has reached the gate's window (default 24 hours);
- **second operator** counts only alongside a first approval (two-person rule);
- **CISO notification** is independent.

A model has no way to set any of these — they are collected out of band — so the
gate remains a hard backstop even when every in-band guardrail has been bypassed.
