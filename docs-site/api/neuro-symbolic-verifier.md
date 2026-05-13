# Neuro-Symbolic Verifier

::: director_ai.core.verification.neuro_symbolic.NeuroSymbolicVerifier

::: director_ai.core.verification.neuro_symbolic.NeuroSymbolicVerifierInput

::: director_ai.core.verification.neuro_symbolic.NeuroSymbolicVerificationResult

## Verification Model

`NeuroSymbolicVerifier` fuses a neural plausibility score with deterministic
checks where the claim is mechanically verifiable:

- numeric/date/probability consistency through `verify_numeric()`
- caller-supplied logical constraints through `ReasoningVerifier`
- optional DPLL, Z3, or Lean-compatible formal backends through the existing
  formal verification interfaces

Numeric and symbolic contradictions are decisive rejects. A low neural score is
a warning when symbolic checks pass, because neural uncertainty and formal
inconsistency are different failure modes.

```python
from director_ai.core import NeuroSymbolicVerifier, NeuroSymbolicVerifierInput
from director_ai.core.formal_verification import ReasoningStep, Variable

verifier = NeuroSymbolicVerifier(neural_accept_threshold=0.7)
result = verifier.verify(
    NeuroSymbolicVerifierInput(
        text=response,
        neural_score=score.score,
        symbolic_steps=(ReasoningStep("claim-a", Variable("A")),),
        evidence_ref="claim://a",
    )
)
```

Serialization redacts raw text by default. Use `include_text=True` only inside a
trusted tenant or forensic boundary.
