# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — causal counterfactual verifier package

"""Causal counterfactual trajectory verifier (roadmap Tier 1 Batch 3 #5).

A fast, dependency-free structural causal model (SCM) evaluator
that answers "would the safety invariant still hold if the agent
had made decision X instead of Y?" by:

* :class:`CausalGraph` — a typed DAG of variables with structural
  equations attached. Topological sort is pre-computed once so
  intervention + propagation is O(V + E) per branch.
* :class:`Intervention` — Pearl's ``do(X = x)`` operator: fix a
  variable and propagate the downstream values.
* :class:`CounterfactualVerifier` — generates several what-if
  branches around a decision point and reports which branches
  keep the safety invariant satisfied.

Foundation scope: deterministic structural equations (caller-supplied
callables). Stochastic structural equations, DoWhy-style effect
estimation, and the Rust kernel are drop-ins on top of the stable
:class:`CausalGraph` and :class:`Intervention` boundaries.
"""

from .counterfactual import CounterfactualBranch, CounterfactualVerifier, Verdict
from .graph import CausalGraph, GraphCycleError
from .intervention import Intervention

__all__ = [
    "CausalGraph",
    "CounterfactualBranch",
    "CounterfactualVerifier",
    "GraphCycleError",
    "Intervention",
    "Verdict",
]
