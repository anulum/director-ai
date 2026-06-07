# Hallucination Root-Cause Analysis

Move from detective to **prescriptive**: where the
[mechanistic interpretability attributor](mechanistic-interpretability.md)
(ReDeEP) says *which* layers and attention heads produced a hallucination signal,
the root-cause analyzer says *why* and *what to fix* — classifying the dominant
failure mode and emitting targeted fine-tuning recommendations that can feed the
Customer Model Factory.

## Failure modes

| Cause | Mechanistic signature | Prescription |
|-------|----------------------|--------------|
| `parametric_knowledge_override` | A Knowledge-FFN layer with high `ffn_knowledge` **and** low `external_attention` | Fine-tune to down-weight parametric recall, up-weight retrieved context |
| `attention_ignores_evidence` | Low `external_attention` without strong parametric recall | Add retrieval-grounding training examples |
| `underactive_copying_heads` | Copying-Heads with low `copying_score` | Train on copy-from-context exemplars |

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig

guard = ProductionGuard(DirectorConfig())

# `report` is a MechanisticAttributionReport from the ReDeEP attributor.
diagnosis = guard.root_cause_analyzer.diagnose(report)

print(diagnosis.dominant_cause)        # e.g. "parametric_knowledge_override"
print(diagnosis.implicated_layers)     # e.g. (18, 19, 20)
for rec in diagnosis.recommendations:
    print(rec.cause, "→", rec.action, rec.targets)
```

`diagnose()` returns a `RootCauseDiagnosis` with the `dominant_cause`, all
detected `causes`, per-cause `Recommendation`s (`cause`, `action`, `targets` like
`ffn_layer:18` / `head:5.2`), and the `implicated_layers` / `implicated_heads`.
When the report is not a hallucination the dominant cause is `no_hallucination`
with no recommendations.

## Tenant safety

The diagnosis is computed from the attribution report alone — no model access and
no prompt/response text. `to_dict()` carries only component indices, scores, and
recommendations, so it is safe to log and to attach to audit records.

## Tuning

```python
from director_ai.core.interpretability import HallucinationRootCauseAnalyzer

analyzer = HallucinationRootCauseAnalyzer(
    parametric_knowledge_threshold=0.5,   # FFN knowledge at/above → strong recall
    low_attention_threshold=0.3,          # external attention at/below → ignoring evidence
    low_copying_threshold=0.3,            # copying score at/below → weak copying head
)
```

## Notes

- Pure deterministic analysis over the attribution report; no ML stack required
  (the report's signals come from the ReDeEP `ActivationProvider`, which a real
  deployment backs with transformer activations/attention).
- The recommendations are designed to be consumed by a targeted fine-tuning
  pipeline (Customer Model Factory), closing the loop from detection to repair.
