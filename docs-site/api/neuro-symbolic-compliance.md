# Neuro-Symbolic Compliance (SMT)

Check an LLM's structured output against a **formally specified policy** with an
SMT solver (Z3), getting back a per-rule verdict and a concrete counterexample
when a rule is broken — logical correctness, not a similarity score. Suited to
regulated workflows (finance, legal, healthcare) where a policy is a set of hard
constraints over typed values.

This complements the propositional [neuro-symbolic verifier](neuro-symbolic-verifier.md):
that checks reasoning-step entailment; this checks an output against arithmetic
and relational *policy* constraints.

## Two-stage design

1. **Formalise** — a natural-language policy becomes typed SMT constraints. This
   step is delegated to an injectable `PolicyFormaliser` (an LLM in production; a
   structured stub in tests). Two independent formalisations can be cross-checked
   for logical equivalence to catch a faulty formalisation.
2. **Check** — a candidate output's structured facts are checked against the
   constraints with Z3, producing a per-rule verdict with counterexamples.

The policy AST is solver-agnostic, so policies can be authored, serialised, and
cross-checked without a solver; only `check`/`is_consistent`/`equivalent_to`/
`to_smt_lib` need Z3 (the `[formal]` extra).

## Quick start

```python
from director_ai.core.neuro_symbolic import (
    CompliancePolicy, Constraint, NeuroSymbolicComplianceEngine,
    var, lit, le, gt, implies, REAL, BOOL,
)

amount = var("amount", REAL)
approved = var("manager_approved", BOOL)

policy = CompliancePolicy([
    Constraint("amount_limit", le(amount, lit(10000))),
    Constraint("approval_required", implies(gt(amount, lit(5000)), approved)),
])

engine = NeuroSymbolicComplianceEngine(policy)

verdict = engine.check({"amount": 15000, "manager_approved": False})
print(verdict.compliant)                       # False
print([v.name for v in verdict.violations])    # ['amount_limit', 'approval_required']
print(verdict.violations[0].counterexample)    # {'amount': 15000.0, 'manager_approved': False}
```

`check()` returns a `ComplianceVerdict` with `compliant`, the list of
`ConstraintViolation`s (each with a counterexample assignment), and the number of
constraints checked. `to_dict()` gives a tenant-safe record (rule names and
counterexamples only).

## Policy language

Typed variables (`bool`, `int`, `real`) with comparisons (`eq`, `ne`, `lt`, `le`,
`gt`, `ge`), arithmetic (`add`, `sub`, `mul`), and boolean connectives (`and_`,
`or_`, `not_`, `implies`):

```python
from director_ai.core.neuro_symbolic import var, lit, le, mul, INT, Constraint

quantity = var("quantity", INT)
Constraint("cap", le(mul(lit(2), quantity), lit(20)))   # 2 * quantity <= 20
```

## Consistency and equivalence

```python
engine.is_consistent()              # False if the policy is self-contradictory
engine.equivalent_to(other_policy)  # True if two formalisations are logically equal
print(engine.to_smt_lib())          # SMT-LIB text — an auditable artefact
```

`equivalent_to` is the cross-check for the two-stage flow: formalise the same
policy two ways and confirm they agree on every assignment before trusting
either.

## Via the guard

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig

engine = ProductionGuard(DirectorConfig()).compliance_engine(policy)
```

## Notes

- Requires the `[formal]` extra (`pip install director-ai[formal]`); the engine
  raises a clear error if Z3 is unavailable.
- The solver-agnostic policy AST is always importable; only solving needs Z3.
- Counterexamples are full variable assignments, so a violation is auditable and
  reproducible.
