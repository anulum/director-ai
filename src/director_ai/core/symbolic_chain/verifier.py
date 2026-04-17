# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NeuralSymbolicVerifier

"""Glue between the NLI scoring layer and the symbolic prover.

The verifier takes a set of claims + relations (typically
extracted by an upstream component) and a prover backend, runs
the prover, and returns a :class:`ConsistencyReport`. This lets
the scoring pipeline escalate from fuzzy NLI to a structured
check when a policy demands it.

The verifier is a thin adapter by design. The interesting work
lives in the prover (swappable via :class:`ProverBackend`) and
the claim extractor, which runs upstream and hands pre-built
claims to :meth:`verify`.
"""

from __future__ import annotations

from collections.abc import Iterable

from .claims import Claim, ClaimRelation
from .prover import ConsistencyReport, GraphProver, ProverBackend


class NeuralSymbolicVerifier:
    """Run ``prover.check`` on the supplied claim set.

    Parameters
    ----------
    prover :
        Any :class:`ProverBackend`. Defaults to :class:`GraphProver`.
    """

    def __init__(self, *, prover: ProverBackend | None = None) -> None:
        self._prover: ProverBackend = prover or GraphProver()

    def verify(
        self,
        claims: Iterable[Claim],
        relations: Iterable[ClaimRelation],
    ) -> ConsistencyReport:
        return self._prover.check(claims, relations)
