# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Neuro-Symbolic Compliance Engine (SMT/Z3)
"""Check LLM outputs against a formalised policy with an SMT solver.

Two-stage neuro-symbolic compliance: a policy is formalised into typed SMT
constraints (the natural-language → SMT-LIB step is delegated to an injectable
:class:`PolicyFormaliser` — an LLM in production, a structured stub in tests), and
a candidate output's structured facts are checked against those constraints with
Z3. The engine reports a per-constraint verdict with a concrete counterexample
when a rule is violated, can cross-check that two independent formalisations are
logically equivalent (catching a faulty formalisation), and emits the SMT-LIB
text as an auditable artefact.

Z3 is imported lazily; constructing a policy needs no solver, only
:meth:`NeuroSymbolicComplianceEngine.check` and friends require ``z3-solver``
(the ``[formal]`` extra).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .expression import (
    BOOL,
    INT,
    Arith,
    BoolOp,
    Compare,
    Const,
    Expr,
    Not,
    Var,
    variables,
)


@dataclass(frozen=True)
class Constraint:
    """A named policy rule (a boolean expression that must hold)."""

    name: str
    expr: Expr

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Constraint.name must be non-empty")


@dataclass(frozen=True)
class ConstraintViolation:
    """A violated rule plus a counterexample assignment that breaks it."""

    name: str
    counterexample: dict[str, bool | int | float]

    def to_dict(self) -> dict[str, Any]:
        """Serialisable violation record."""
        return {"name": self.name, "counterexample": dict(self.counterexample)}


@dataclass(frozen=True)
class ComplianceVerdict:
    """The outcome of checking facts against a policy."""

    compliant: bool
    violations: tuple[ConstraintViolation, ...]
    constraints_checked: int

    def to_dict(self) -> dict[str, Any]:
        """Tenant-safe serialisable verdict (rule names + counterexamples only)."""
        return {
            "compliant": self.compliant,
            "constraints_checked": self.constraints_checked,
            "violations": [v.to_dict() for v in self.violations],
        }


class CompliancePolicy:
    """An ordered set of named constraints over typed variables."""

    def __init__(self, constraints: tuple[Constraint, ...] | list[Constraint]) -> None:
        constraints = tuple(constraints)
        if not constraints:
            raise ValueError("a policy needs at least one constraint")
        names = [c.name for c in constraints]
        if len(set(names)) != len(names):
            raise ValueError("constraint names must be unique")
        self._constraints = constraints
        # Validating variables here surfaces sort conflicts before any solving.
        self._variables = self._collect_variables()

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """The policy's constraints in declaration order."""
        return self._constraints

    def variables(self) -> dict[str, str]:
        """Return ``{name: sort}`` across all constraints."""
        return dict(self._variables)

    def _collect_variables(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for constraint in self._constraints:
            for name, sort in variables(constraint.expr).items():
                existing = merged.get(name)
                if existing is not None and existing != sort:
                    raise ValueError(
                        f"variable {name!r} used as both {existing} and {sort}"
                    )
                merged[name] = sort
        return merged


@runtime_checkable
class PolicyFormaliser(Protocol):
    """Turns a natural-language policy into a :class:`CompliancePolicy`.

    The production implementation is LLM-backed (natural language → SMT-LIB);
    tests provide a deterministic structured stub. Cross-checking two independent
    formalisations with :meth:`NeuroSymbolicComplianceEngine.equivalent_to`
    guards against a single faulty formalisation.
    """

    def formalise(self, natural_language_policy: str) -> CompliancePolicy: ...


def _require_z3() -> Any:
    try:
        import z3
    except ImportError as exc:  # pragma: no cover — exercised only without z3
        raise RuntimeError(
            "the neuro-symbolic compliance engine requires z3; "
            "install the [formal] extra (pip install director-ai[formal])"
        ) from exc
    return z3


class NeuroSymbolicComplianceEngine:
    """Verify outputs against a :class:`CompliancePolicy` with Z3."""

    def __init__(self, policy: CompliancePolicy) -> None:
        self._policy = policy
        self._z3 = _require_z3()

    @property
    def policy(self) -> CompliancePolicy:
        """The policy being enforced."""
        return self._policy

    def _z3_vars(self, sorts: dict[str, str]) -> dict[str, Any]:
        z3 = self._z3
        out: dict[str, Any] = {}
        for name, sort in sorts.items():
            if sort == BOOL:
                out[name] = z3.Bool(name)
            elif sort == INT:
                out[name] = z3.Int(name)
            else:
                out[name] = z3.Real(name)
        return out

    def _compile(self, expr: Expr, zvars: dict[str, Any]) -> Any:
        z3 = self._z3
        if isinstance(expr, Var):
            return zvars[expr.name]
        if isinstance(expr, Const):
            if isinstance(expr.value, bool):
                return z3.BoolVal(expr.value)
            return (
                z3.RealVal(expr.value)
                if isinstance(expr.value, float)
                else z3.IntVal(expr.value)
            )
        if isinstance(expr, Not):
            return z3.Not(self._compile(expr.operand, zvars))
        if isinstance(expr, BoolOp):
            left = self._compile(expr.left, zvars)
            right = self._compile(expr.right, zvars)
            if expr.op == "and":
                return z3.And(left, right)
            if expr.op == "or":
                return z3.Or(left, right)
            return z3.Implies(left, right)
        if isinstance(expr, Compare):
            left = self._compile(expr.left, zvars)
            right = self._compile(expr.right, zvars)
            # if/elif (not a dict) so only the selected comparison is built —
            # z3 booleans support == / != but not the ordering operators.
            if expr.op == "eq":
                return left == right
            if expr.op == "ne":
                return left != right
            if expr.op == "lt":
                return left < right
            if expr.op == "le":
                return left <= right
            if expr.op == "gt":
                return left > right
            return left >= right
        if isinstance(expr, Arith):
            left = self._compile(expr.left, zvars)
            right = self._compile(expr.right, zvars)
            if expr.op == "add":
                return left + right
            if expr.op == "sub":
                return left - right
            return left * right
        raise TypeError(f"unknown expression node: {expr!r}")  # pragma: no cover

    def _fact_assignments(
        self, facts: dict[str, bool | int | float], zvars: dict[str, Any]
    ) -> list[Any]:
        sorts = self._policy.variables()
        assignments = []
        for name, value in facts.items():
            if name not in zvars:
                raise ValueError(f"unknown fact variable {name!r}")
            sort = sorts[name]
            if sort == BOOL and not isinstance(value, bool):
                raise ValueError(f"fact {name!r} must be boolean")
            if sort != BOOL and isinstance(value, bool):
                raise ValueError(f"fact {name!r} must be numeric")
            assignments.append(zvars[name] == value)
        return assignments

    def _model_to_py(
        self, model: Any, zvars: dict[str, Any]
    ) -> dict[str, bool | int | float]:
        z3 = self._z3
        out: dict[str, bool | int | float] = {}
        for name, zvar in zvars.items():
            raw = model.eval(zvar, model_completion=True)
            if z3.is_bool(zvar):
                out[name] = bool(z3.is_true(raw))
            elif z3.is_int(zvar):
                out[name] = raw.as_long()
            else:
                out[name] = float(raw.as_fraction())
        return out

    def check(self, facts: dict[str, bool | int | float]) -> ComplianceVerdict:
        """Check ``facts`` against every constraint, reporting per-rule violations.

        A constraint is violated when the facts admit an assignment under which
        the rule is false; the returned counterexample is that assignment. When
        the facts pin every variable the counterexample is the facts themselves.
        """
        z3 = self._z3
        zvars = self._z3_vars(self._policy.variables())
        assignments = self._fact_assignments(facts, zvars)
        violations: list[ConstraintViolation] = []
        for constraint in self._policy.constraints:
            solver = z3.Solver()
            solver.add(*assignments)
            solver.add(z3.Not(self._compile(constraint.expr, zvars)))
            if solver.check() == z3.sat:
                violations.append(
                    ConstraintViolation(
                        name=constraint.name,
                        counterexample=self._model_to_py(solver.model(), zvars),
                    )
                )
        return ComplianceVerdict(
            compliant=not violations,
            violations=tuple(violations),
            constraints_checked=len(self._policy.constraints),
        )

    def is_consistent(self) -> bool:
        """True when the policy itself is satisfiable (not self-contradictory)."""
        z3 = self._z3
        zvars = self._z3_vars(self._policy.variables())
        solver = z3.Solver()
        for constraint in self._policy.constraints:
            solver.add(self._compile(constraint.expr, zvars))
        return bool(solver.check() == z3.sat)

    def equivalent_to(self, other: CompliancePolicy) -> bool:
        """True when this policy and ``other`` are logically equivalent.

        Cross-check for redundant formalisations: equivalence holds iff their
        conjunctions agree on every assignment (``¬(A ↔ B)`` is unsatisfiable).
        """
        z3 = self._z3
        sorts = dict(self._policy.variables())
        for name, sort in other.variables().items():
            if sorts.get(name, sort) != sort:
                raise ValueError(f"variable {name!r} has conflicting sorts")
            sorts[name] = sort
        zvars = self._z3_vars(sorts)
        a = z3.And(*(self._compile(c.expr, zvars) for c in self._policy.constraints))
        b = z3.And(*(self._compile(c.expr, zvars) for c in other.constraints))
        solver = z3.Solver()
        solver.add(z3.Not(a == b))
        return bool(solver.check() == z3.unsat)

    def to_smt_lib(self) -> str:
        """Return the policy conjunction as SMT-LIB text (an audit artefact)."""
        z3 = self._z3
        zvars = self._z3_vars(self._policy.variables())
        solver = z3.Solver()
        for constraint in self._policy.constraints:
            solver.add(self._compile(constraint.expr, zvars))
        return str(solver.to_smt2())
