# Risk-Adaptive Thresholding

One global approval threshold is wrong for every request. A read-only summary for
an internal admin and an irreversible, externally-published action touching PII
in a regulated domain deserve different bars. Risk-adaptive thresholding computes
a **per-request** threshold from a documented risk profile, deterministically,
recording every factor's contribution so the decision is auditable.

A positive contribution raises the threshold (stricter — more grounding
required); a negative one lowers it. Only a demonstrated high false-halt rate
lowers the bar.

## Factors

| Factor | Direction |
|---|---|
| `user_role` | per-role delta (trusted roles relax, anonymous tightens) |
| `tenant_risk` | higher tenant risk → stricter |
| `domain` | per-domain delta (medical / finance / legal tighten) |
| `retrieval_confidence` | weak retrieval → stricter |
| `action_reversibility` | irreversible action → stricter |
| `external_exposure` | answer leaves the org → stricter |
| `pii_present` | personal data involved → stricter |
| `freshness` | stale evidence → stricter |
| `historical_fpr` | high false-halt rate → **relax** |

```python
from director_ai.core.risk_threshold import RiskAdaptiveThreshold, RiskFactors

adapter = RiskAdaptiveThreshold()   # default policy, base 0.6

decision = adapter.evaluate(
    RiskFactors(
        user_role="anonymous",
        domain="medical",
        retrieval_confidence=0.4,
        action_reversibility=0.2,
        external_exposure=True,
        pii_present=True,
        freshness=0.5,
    )
)

print(decision.threshold)        # clamped, e.g. 0.95
print(decision.contributions)    # {"user_role": 0.1, "domain": 0.1, ...}
print(decision.total_delta)      # sum before clamping
```

The result is clamped to the policy's `[min_threshold, max_threshold]`. Every
weight and per-category delta lives in `RiskThresholdPolicy`, so the mapping is
explicit and tunable.

## Through the guard

`ProductionGuard.risk_threshold(factors)` adapts the guard's configured coherence
threshold and returns the decision; the host applies it (the guard does not
mutate its own threshold):

```python
from director_ai.guard import ProductionGuard
from director_ai.core.risk_threshold import RiskFactors

guard = ProductionGuard.from_profile("finance")
decision = guard.risk_threshold(RiskFactors(domain="finance", pii_present=True))
approved = result.score >= decision.threshold
```

## Relation to the other threshold mechanisms

This is a **deterministic, per-request** adjustment from known risk inputs. It
composes with — and does not replace — the
[adaptive threshold learner](adaptive-threshold.md) (which learns a base
threshold from feedback over time) and the
[uncertainty router](uncertainty-routing.md) (which routes a single result by its
conformal interval).
