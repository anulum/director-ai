# Answer Bill of Materials

The Answer Bill of Materials (`AnswerBOM`) is a versioned, machine-readable
manifest of one guarded answer. Where the guard's decision says *whether* an
answer was approved, the BOM says *why, claim by claim*: which model produced
the answer, which scorer judged it against which threshold, and for every atomic
claim — what evidence supports it, how strongly, and the verdict the guard
reached.

It is the artefact a reviewer or auditor reads to answer *"which parts of this
answer are grounded, and in what?"* without re-running the guard, and it
round-trips through an audit log via `to_dict` / `from_dict`.

## Shape

```json
{
  "schema_version": "director.answer_bom.v1",
  "answer_id": "abom_…",
  "timestamp": "2026-06-06T16:40:00Z",
  "model": "gpt-4o",
  "scorer": "rust",
  "threshold": 0.6,
  "tenant": "acme",
  "claims": [
    {
      "claim": "Refunds are available within 30 days.",
      "verdict": "supported",
      "support": 0.91,
      "evidence_ids": ["vector:refund-policy"],
      "freshness": "2026-01-01",
      "tenant": "acme",
      "policy_refs": ["refund-policy-v3"]
    }
  ],
  "unsupported_claims": [],
  "support_coverage": 1.0
}
```

`verdict` is one of `supported`, `unsupported`, or `contradicted`. `support` is
a strength in `[0, 1]`. `unsupported_claims` and `support_coverage` are derived
from `claims` and recomputed on load, so editing them in a stored manifest has
no effect.

## Building from a guard result

`ProductionGuard.answer_bom()` builds the manifest from a `GuardResult`. The
threshold is the calibrated threshold when calibration is enabled, otherwise the
configured coherence threshold; the scorer id is the configured backend.

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard.from_profile("finance")
guard.load_facts({"refund": "Refunds are available within 30 days."})

result = guard.check("What is the refund window?", "Refunds close after 30 days.")
bom = guard.answer_bom(
    result,
    model="gpt-4o",
    tenant="acme",
    freshness="2026-01-01",
    policy_refs=["refund-policy-v3"],
)

for claim in bom.unsupported_claims:
    print(claim.claim, claim.verdict, claim.support)

audit_log.write(bom.to_json())          # round-trips via AnswerBOM.from_dict
```

`build_answer_bom(score, ...)` is the underlying free function for callers that
already hold a `CoherenceScore`.

## How claims are mapped

The BOM consumes the scorer's existing claim-level provenance — the atomic
claims and, for each, the source sentence it was attributed to with a
support/contradiction divergence:

- **supported** → `verdict="supported"`, `support = 1 − divergence`, and
  `evidence_ids` resolved from the attributed source chunk's id (or a positional
  `source:N` reference when the evidence carries no id).
- **not supported, high divergence** (≥ `contradiction_threshold`, default 0.5)
  → `verdict="contradicted"`, no evidence ids.
- **not supported, low divergence** → `verdict="unsupported"`.

The manifest's claim-level richness is exactly that of the scorer's provenance:
when the scorer produced no attributions, the BOM records the
model/scorer/threshold header with no per-claim rows rather than inventing them.
