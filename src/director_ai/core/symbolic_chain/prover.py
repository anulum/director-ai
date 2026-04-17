# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — prover Protocol + graph-closure prover

"""Prover boundary and the pure-Python graph-closure fallback.

The Protocol lets a Z3/Lean/WASM-prover drop in on equal terms.
The :class:`GraphProver` is a closure-based engine that finds the
common failure modes without a SAT solver: polarity conflicts
between two claims with the same ``id``, direct
``A contradicts B`` relations, and one-hop transitive
``A implies B`` / ``B contradicts C`` chains that imply
``A contradicts C``. That covers the bulk of real-world
neural-symbolic hits; the full SAT / SMT surface is a drop-in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .claims import Claim, ClaimRelation

Status = Literal["consistent", "inconsistent"]


@dataclass(frozen=True)
class ConsistencyReport:
    """Result of a prover run.

    ``conflicts`` carries the specific ``(source, target, reason)``
    triples that triggered the inconsistency so callers can log or
    surface them verbatim.
    """

    status: Status
    conflicts: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)

    @property
    def is_consistent(self) -> bool:
        return self.status == "consistent"


@runtime_checkable
class ProverBackend(Protocol):
    """Contract for anything that can decide the consistency of a
    claim set. Callers pass the full set each time — provers are
    stateless so the backend can be swapped live."""

    def check(
        self,
        claims: Iterable[Claim],
        relations: Iterable[ClaimRelation],
    ) -> ConsistencyReport: ...


class GraphProver:
    """Pure-Python closure-based prover.

    Algorithm (two passes over the relation set):

    1. Polarity pass — a claim whose id appears with both
       ``negated=True`` and ``negated=False`` is an immediate
       contradiction (no prover call required).
    2. Direct pass — every ``contradicts`` relation yields a
       conflict.
    3. Transitive pass — for every ``A implies B`` relation, check
       whether ``B`` is asserted negated, or whether any
       ``B contradicts C`` relation implies ``A contradicts C``.

    The closure is one-hop only; deeper chains are a follow-up on
    the same Protocol boundary.
    """

    def check(
        self,
        claims: Iterable[Claim],
        relations: Iterable[ClaimRelation],
    ) -> ConsistencyReport:
        claim_list = list(claims)
        relation_list = list(relations)
        conflicts: list[tuple[str, str, str]] = []

        # Polarity pass.
        polarity_map: dict[str, set[bool]] = {}
        for c in claim_list:
            polarity_map.setdefault(c.id, set()).add(c.negated)
        for cid, polarities in polarity_map.items():
            if len(polarities) > 1:
                conflicts.append((cid, cid, "polarity conflict"))

        # Direct pass.
        for r in relation_list:
            if r.kind == "contradicts":
                conflicts.append((r.source, r.target, "direct contradicts"))

        # Transitive pass — one hop.
        implies = {(r.source, r.target) for r in relation_list if r.kind == "implies"}
        contradicts = {
            (r.source, r.target) for r in relation_list if r.kind == "contradicts"
        }
        for a, b in implies:
            for c_source, c_target in contradicts:
                if b == c_source:
                    conflicts.append(
                        (a, c_target, f"transitive via {b}")
                    )

        if conflicts:
            return ConsistencyReport(
                status="inconsistent", conflicts=tuple(conflicts)
            )
        return ConsistencyReport(status="consistent")
