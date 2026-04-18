# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ReasoningVerifier + external backends

"""Compose the CNF converter + DPLL solver with optional
external backends.

:class:`ReasoningVerifier` takes a list of :class:`ReasoningStep`
(each a :class:`Formula` plus an opaque label) and asks the
selected backend whether their conjunction is satisfiable. The
shipped default is :class:`DpllSolver`. :class:`Z3Backend` and
:class:`LeanBackend` are lazy adapters for the canonical SMT /
proof-assistant backends.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .cnf import CnfConverter
from .dpll import DpllSolver, Solution
from .formula import And, Formula


@dataclass(frozen=True)
class ReasoningStep:
    """One step in a reasoning chain."""

    label: str
    formula: Formula

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("ReasoningStep.label must be non-empty")


@dataclass(frozen=True)
class ReasoningVerdict:
    """Outcome of one :meth:`ReasoningVerifier.verify` call."""

    consistent: bool
    model: dict[str, bool]
    step_count: int
    backend: str

    @property
    def contradictory(self) -> bool:
        return not self.consistent


@runtime_checkable
class VerifierBackend(Protocol):
    """Protocol implemented by any concrete solver adapter."""

    name: str

    def solve(self, formula: Formula) -> Solution: ...


class ReasoningVerifier:
    """Bind a :class:`VerifierBackend` to a CNF converter.

    Parameters
    ----------
    backend :
        The solver to use. Default is a fresh :class:`DpllSolver`
        wrapped in :class:`_DpllBackend`.
    """

    def __init__(self, *, backend: VerifierBackend | None = None) -> None:
        self._backend: VerifierBackend = backend or _DpllBackend()

    def verify(self, steps: Sequence[ReasoningStep]) -> ReasoningVerdict:
        if not steps:
            raise ValueError("steps must be non-empty")
        conjunction = steps[0].formula
        for step in steps[1:]:
            conjunction = And(conjunction, step.formula)
        solution = self._backend.solve(conjunction)
        return ReasoningVerdict(
            consistent=solution.satisfiable,
            model=dict(solution.model) if solution.satisfiable else {},
            step_count=len(steps),
            backend=self._backend.name,
        )


class _DpllBackend:
    """Default backend wrapping :class:`DpllSolver`."""

    name = "dpll"

    def __init__(self, *, solver: DpllSolver | None = None) -> None:
        self._solver = solver or DpllSolver()
        self._converter = CnfConverter()

    def solve(self, formula: Formula) -> Solution:
        clauses = self._converter.convert(formula)
        return self._solver.solve(clauses)


class Z3Backend:
    """``z3`` adapter loaded lazily.

    Use :meth:`from_z3` when you want the backend to import and
    initialise the ``z3`` solver itself. Passing an already-built
    solver via the constructor is useful when the caller wants
    to tune Z3 parameters.
    """

    name = "z3"

    def __init__(self, *, z3_solver: Any) -> None:
        if z3_solver is None:
            raise ValueError("z3_solver is required")
        self._solver = z3_solver

    @classmethod
    def from_z3(cls) -> Z3Backend:
        try:
            import z3
        except ImportError as exc:
            raise ImportError(
                "Z3Backend.from_z3 requires the z3-solver package. "
                "Install with: pip install director-ai[formal]"
            ) from exc
        return cls(z3_solver=z3.Solver())

    def solve(self, formula: Formula) -> Solution:
        try:
            import z3
        except ImportError as exc:  # pragma: no cover — covered by from_z3
            raise ImportError("Z3Backend.solve requires the z3-solver package") from exc
        self._solver.reset()
        self._solver.add(_formula_to_z3(formula, z3))
        result = self._solver.check()
        if result == z3.sat:
            z3_model = self._solver.model()
            model: dict[str, bool] = {}
            for decl in z3_model.decls():
                model[decl.name()] = bool(z3_model[decl])
            return Solution(satisfiable=True, model=model)
        return Solution(satisfiable=False)


def _formula_to_z3(formula: Formula, z3: Any) -> Any:
    """Map the local AST into z3 Bool expressions. Local import
    of z3 keeps the module loadable without the optional
    dependency."""
    from .formula import Iff, Implies, Not, Or, Variable

    if isinstance(formula, Variable):
        return z3.Bool(formula.name)
    if isinstance(formula, Not):
        return z3.Not(_formula_to_z3(formula.operand, z3))
    if isinstance(formula, And):
        return z3.And(
            _formula_to_z3(formula.left, z3),
            _formula_to_z3(formula.right, z3),
        )
    if isinstance(formula, Or):
        return z3.Or(
            _formula_to_z3(formula.left, z3),
            _formula_to_z3(formula.right, z3),
        )
    if isinstance(formula, Implies):
        return z3.Implies(
            _formula_to_z3(formula.antecedent, z3),
            _formula_to_z3(formula.consequent, z3),
        )
    if isinstance(formula, Iff):
        return _formula_to_z3(formula.left, z3) == _formula_to_z3(formula.right, z3)
    raise TypeError(
        f"unknown formula node {type(formula).__name__}"
    )  # pragma: no cover


class LeanBackend:
    """Lean 4 adapter.

    Constructs a ``lean`` subprocess command template that the
    caller runs offline. The adapter only handles the
    formula-to-Lean string serialisation + the model-parsing
    contract; the actual Lean invocation happens through the
    caller's configured :class:`subprocess.Popen` command.
    """

    name = "lean"

    def __init__(self, *, runner: Any) -> None:
        if runner is None:
            raise ValueError("runner is required")
        if not callable(runner):
            raise ValueError("runner must be callable")
        self._runner = runner

    def solve(self, formula: Formula) -> Solution:
        lean_source = _formula_to_lean(formula)
        response = self._runner(lean_source)
        if not isinstance(response, dict):
            raise ValueError(
                "LeanBackend runner must return a dict with 'sat' + 'model' keys"
            )
        sat = bool(response.get("sat", False))
        model_raw = response.get("model") or {}
        if not isinstance(model_raw, dict):
            raise ValueError("LeanBackend runner returned a non-dict model")
        model = {str(k): bool(v) for k, v in model_raw.items()}
        return Solution(satisfiable=sat, model=model)


def _formula_to_lean(formula: Formula) -> str:
    """Render ``formula`` as a Lean 4 ``def`` over Bool."""
    from .formula import Iff, Implies, Not, Or, Variable

    def render(f: Formula) -> str:
        if isinstance(f, Variable):
            return f.name
        if isinstance(f, Not):
            return f"(¬ {render(f.operand)})"
        if isinstance(f, And):
            return f"({render(f.left)} ∧ {render(f.right)})"
        if isinstance(f, Or):
            return f"({render(f.left)} ∨ {render(f.right)})"
        if isinstance(f, Implies):
            return f"({render(f.antecedent)} → {render(f.consequent)})"
        if isinstance(f, Iff):
            return f"({render(f.left)} ↔ {render(f.right)})"
        raise TypeError(f"unknown formula node {type(f).__name__}")  # pragma: no cover

    body = render(formula)
    return f"def target : Prop := {body}"
