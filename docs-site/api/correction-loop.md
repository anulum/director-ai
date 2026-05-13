# Correction Loop

The correction loop creates approval-gated remediation proposals after a halt
or warning. It does not release replacement text automatically. A proposal must
first pass cross-verifier consensus with an `allow` decision, then receive an
explicit operator approval ID before `release()` returns candidate text.

Physical-domain and irreversible actions are rejected at proposal time. Those
cases must stay in the no-go or human-review path because automatic remediation
can hide actuator, security, or irreversible operational risk.

## Usage

```python
from director_ai.core import CorrectionLoop
from director_ai.core.guard_control import RiskEnvelope, VerifierSignal
from director_ai.core.scoring.consensus import CrossVerifierConsensus

loop = CorrectionLoop(
    consensus=CrossVerifierConsensus(),
    risk_envelope=RiskEnvelope(
        action_category="text",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.65,
        no_go_threshold=0.9,
    ),
    policy_id="policy.correction.regulated",
)

proposal = loop.propose(
    candidate_text="Corrected response text.",
    signals=[
        VerifierSignal(
            verifier="nli",
            modality="text",
            score=0.08,
            verdict="supported",
            confidence_low=0.03,
            confidence_high=0.14,
            evidence_refs=("kb://fact-1",),
        )
    ],
    evidence_refs=("trace://halt-7",),
)

approved = loop.approve(proposal, approval_id="review-20260513-001")
released_text = loop.release(approved)
```

## Audit Boundary

`CorrectionProposal.to_dict()` excludes `candidate_text` by default so shared
audit records do not leak generated payloads. Use `include_candidate=True` only
inside a tenant-controlled forensic store. Structured recovery audit records
include metadata, errors, validity, and halt offset; they do not include raw
partial output or recovered payload text.

## Full API

::: director_ai.core.runtime.correction.CorrectionLoop

::: director_ai.core.runtime.correction.CorrectionProposal
