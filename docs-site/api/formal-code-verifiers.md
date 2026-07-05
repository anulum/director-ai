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

## Theorem Backend Selection

Use `with_theorem_backend()` when a deployment needs an explicit theorem-prover
profile. The built-in `dpll` profile has no optional dependency. The `z3` profile
uses the `director-ai[formal]` extra, and the `lean` profile accepts a caller
owned runner so the operator controls the Lean invocation, sandbox, and file
system boundary.

```python
adapter = FormalCodeVerifierAdapter.with_theorem_backend("dpll")

lean_adapter = FormalCodeVerifierAdapter.with_theorem_backend(
    "lean",
    lean_runner=run_lean_in_sandbox,
)
```

The selected backend is recorded in the result sandbox and verifier id, for
example `formal.dpll` or `formal.lean`.

The default DPLL backend returns a total model for satisfiable formulae: every
variable that appears in the checked formula is present in the returned model.
Variables that are not needed by the SAT search are assigned deterministic
defaults so audit consumers can compare verifier output without depending on
search-path artefacts.

## Code Contracts

`verify_code_contract()` first runs the structural code verifier. If the code is
invalid, the theorem backend is not called. If the code passes structural
checks, the formal contract is verified through the selected backend.

```python
result = adapter.verify_code_contract(
    code=generated_code,
    contract=contract_formula,
    risk_envelope=risk_envelope,
    policy_id="policy.code.contract.regulated",
    evidence_ref="code-contract://snippet-1",
)
```

Audit payloads include evidence references, backend names, and sandbox metadata;
they do not include raw source code or raw formula text.

## Evidence Packet

Generate the local formal-symbolic evidence packet before promoting deployments
that rely on theorem-backed code, math, or numeric guards:

```bash
PYTHONPATH=src python -m benchmarks.formal_symbolic_evidence
```

The packet checks DPLL formula halts/allows, Lean runner invocation, Z3 profile
handling, code-contract ordering, and tenant-safe serialisation. When the
`director-ai[formal]` extra is not installed, the Z3 probe records the optional
dependency gate instead of claiming an actual Z3 proof run.
Customer Model Factory release promotion requires a separate
`FormalSymbolicEvidence` record with external Lean proof and actual Z3 release
packet URIs before the release gate can pass.

## Full API

::: director_ai.core.formal_verification.adapter.FormalCodeVerifierAdapter

::: director_ai.core.formal_verification.adapter.FormalCodeVerificationResult
