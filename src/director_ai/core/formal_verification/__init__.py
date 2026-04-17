# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — formal verification hooks

"""Propositional verification backend for structured reasoning
chains.

Three layers:

* :class:`Formula` AST — ``Variable``, ``Not``, ``And``, ``Or``,
  ``Implies``, ``Iff``. Every node is immutable and hashable so
  CNF conversion can memoise; ``str`` gives a readable
  serialisation for audit output.
* :class:`CnfConverter` — converts any formula into conjunctive
  normal form via the classical De Morgan + implication-removal
  + distribution rules. Output is a list-of-clauses-of-literals
  shape consumable by the solver.
* :class:`DpllSolver` — Davis-Putnam-Logemann-Loveland with unit
  propagation, pure-literal elimination, and chronological
  backtracking. Returns a :class:`Solution` with either a model
  (variable → truth) or ``UNSAT``.

:class:`VerifierBackend` Protocol plus two drop-in adapters for
external solvers:

* :class:`Z3Backend` — ``z3`` lazy-import adapter for SMT-level
  reasoning.
* :class:`LeanBackend` — exposes a ``subprocess`` interface to
  Lean 4's ``leanc`` so callers who want the full proof
  assistant can route through the same Protocol.

:class:`ReasoningVerifier` wraps a backend with an extraction
step that turns a list of caller-supplied reasoning steps into a
single conjunction and asks the backend whether the conjunction
is satisfiable. The shipped default is :class:`DpllSolver`.
"""

from .cnf import Clause, CnfConverter, Literal
from .dpll import DpllSolver, Solution
from .formula import And, Formula, Iff, Implies, Not, Or, Variable
from .verifier import (
    LeanBackend,
    ReasoningStep,
    ReasoningVerdict,
    ReasoningVerifier,
    VerifierBackend,
    Z3Backend,
)

__all__ = [
    "And",
    "Clause",
    "CnfConverter",
    "DpllSolver",
    "Formula",
    "Iff",
    "Implies",
    "LeanBackend",
    "Literal",
    "Not",
    "Or",
    "ReasoningStep",
    "ReasoningVerdict",
    "ReasoningVerifier",
    "Solution",
    "Variable",
    "VerifierBackend",
    "Z3Backend",
]
