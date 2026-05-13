# Formal and Code Verifiers

Formal and code verifier routing converts symbolic formula checks and generated
code checks into the shared `GuardDecision` and `SafetyEvent` contracts.

## Failure Semantics

`FormalCodeVerifierAdapter` keeps verifier uncertainty explicit:

- contradictory formulae map to `halt`
- invalid generated code maps to `halt`
- missing formal backends map to `warn`
- verifier exceptions map to `warn`
- verifier timeouts map to `warn`
- source code and formula text are excluded from audit serialisation
- generated code is not executed by this adapter

```python
from director_ai.core.formal_verification import (
    And,
    FormalCodeVerifierAdapter,
    Not,
    Variable,
)
from director_ai.core.guard_control import RiskEnvelope

adapter = FormalCodeVerifierAdapter(timeout_ms=1000.0)
result = adapter.verify_formula(
    formula=And(Variable("approved"), Not(Variable("approved"))),
    risk_envelope=RiskEnvelope(
        action_category="code",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    ),
    policy_id="policy.formal.regulated",
    evidence_ref="formal://claim-1",
)
```

For code, the adapter delegates to `verify_code()` by default and keeps
execution disabled:

```python
result = adapter.verify_code(
    code=generated_code,
    risk_envelope=risk_envelope,
    policy_id="policy.code.regulated",
    api_manifest={"pd": {"read_csv", "DataFrame"}},
    evidence_ref="code://snippet-1",
)
```

## Full API

::: director_ai.core.formal_verification.adapter.FormalCodeVerifierAdapter

::: director_ai.core.formal_verification.adapter.FormalCodeVerificationResult
