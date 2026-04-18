# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — claim data model

"""Atomic propositions and the relations that can hold between them.

Claims are string-keyed so any extractor (LLM, grammar, human) can
produce them without agreeing on an identifier scheme. Polarity
is explicit — a negated claim is its own ID with ``negated=True``
rather than a magic-string convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ClaimRelationKind = Literal["implies", "contradicts", "equivalent"]

_VALID_RELATIONS: frozenset[str] = frozenset(("implies", "contradicts", "equivalent"))


@dataclass(frozen=True)
class Claim:
    """One atomic proposition.

    ``id`` is the stable handle used by :class:`ClaimRelation`.
    ``text`` is the human-readable gloss — used for audit output
    and prover error messages; not part of reasoning.
    ``negated`` flips the polarity; the prover treats ``(A, True)``
    and ``(A, False)`` as mutually contradictory regardless of
    whether the extractor declared that relation.
    """

    id: str
    text: str
    negated: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Claim.id must be non-empty")
        if not self.text:
            raise ValueError("Claim.text must be non-empty")


@dataclass(frozen=True)
class ClaimRelation:
    """A binary relation between two claim IDs.

    The source / target are claim IDs, not :class:`Claim`
    instances, so the same relation can be reused across different
    extractor runs that produce the same IDs.
    """

    source: str
    target: str
    kind: ClaimRelationKind

    def __post_init__(self) -> None:
        if self.kind not in _VALID_RELATIONS:
            raise ValueError(
                f"ClaimRelation.kind must be one of {sorted(_VALID_RELATIONS)}; "
                f"got {self.kind!r}"
            )
        if self.source == self.target:
            raise ValueError(
                f"ClaimRelation must be between distinct claims; got {self.source!r}"
            )
        if not self.source or not self.target:
            raise ValueError("ClaimRelation source/target must be non-empty")
