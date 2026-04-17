# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — neural-symbolic reasoning chain package

"""Verifiable neural-symbolic reasoning chain (roadmap Tier 1 Batch 3 #6).

Extract atomic claims and their relations from a response, then
run the set through a theorem prover to detect contradictions
that pure NLI can miss (transitive implications, disjunctions,
modal operators).

* :class:`Claim` — one atomic proposition with a stable id.
* :class:`ClaimRelation` — ``implies``, ``contradicts``,
  ``equivalent`` between two claim IDs.
* :class:`ProverBackend` — Protocol that accepts a set of
  :class:`Claim` + :class:`ClaimRelation` and returns a
  :class:`ConsistencyReport`.
* :class:`GraphProver` — ships with the foundation. Pure-Python
  graph closure: detects direct contradictions and the simplest
  transitive ``A implies B`` / ``B contradicts C`` → ``A
  contradicts C`` pattern. Z3/Lean drop in on the same Protocol.
* :class:`NeuralSymbolicVerifier` — the glue that turns an NLI
  output and a claim set into a :class:`ConsistencyReport`.

Foundation scope: deterministic graph prover, no Z3, no Lean.
The Protocol boundary is stable so those can land without API
churn.
"""

from .claims import Claim, ClaimRelation, ClaimRelationKind
from .prover import ConsistencyReport, GraphProver, ProverBackend
from .verifier import NeuralSymbolicVerifier

__all__ = [
    "Claim",
    "ClaimRelation",
    "ClaimRelationKind",
    "ConsistencyReport",
    "GraphProver",
    "NeuralSymbolicVerifier",
    "ProverBackend",
]
