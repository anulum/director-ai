# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MerkleTree

"""SHA-256 Merkle tree for :class:`CitationFact` collections.

Leaves are the per-fact SHA-256 hashes. Internal nodes are
``SHA-256(left || right)``. When a level has an odd node count,
the last node is duplicated so the tree stays balanced without
adding any extra fact — matching the Bitcoin / RFC 6962 shape.

The tree builds the full path up front (no deferred construction)
so :meth:`MerkleTree.proof_for` is O(log n) lookups.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .facts import CitationFact


@dataclass(frozen=True)
class MerkleProof:
    """Path from a leaf up to the root.

    ``leaf_hash`` is the per-fact SHA-256 digest. ``path`` is a
    tuple of ``(sibling_hash, is_right)`` pairs: ``is_right ==
    True`` means the sibling sits to the right of the current
    node, so the parent is ``SHA-256(current || sibling)``;
    otherwise the sibling is on the left and parent is
    ``SHA-256(sibling || current)``. ``root_hash`` is the
    resulting Merkle root — verifying a proof means folding the
    path and comparing to this value.
    """

    leaf_hash: str
    path: tuple[tuple[str, bool], ...]
    root_hash: str

    def verify(self) -> bool:
        current = self.leaf_hash
        for sibling, sibling_on_right in self.path:
            if sibling_on_right:
                current = _hash_pair(current, sibling)
            else:
                current = _hash_pair(sibling, current)
        return current == self.root_hash


class MerkleTree:
    """Tree built from a fact list.

    Parameters
    ----------
    facts :
        Non-empty sequence of :class:`CitationFact`. The insertion
        order is preserved at the leaf level so proofs reference
        a stable leaf index.
    """

    def __init__(self, facts: tuple[CitationFact, ...] | list[CitationFact]) -> None:
        if not facts:
            raise ValueError("facts must be non-empty")
        leaves = [fact.content_hash for fact in facts]
        self._facts = tuple(facts)
        self._levels: list[list[str]] = [leaves]
        current_level = leaves
        while len(current_level) > 1:
            current_level = _combine_level(current_level)
            self._levels.append(current_level)

    @property
    def root(self) -> str:
        return self._levels[-1][0]

    @property
    def leaf_count(self) -> int:
        return len(self._facts)

    def facts(self) -> tuple[CitationFact, ...]:
        return self._facts

    def proof_for(self, fact: CitationFact) -> MerkleProof:
        """Return the inclusion proof for ``fact``. Raises
        :class:`ValueError` when the fact is not in the tree."""
        try:
            index = self._facts.index(fact)
        except ValueError as exc:
            raise ValueError(f"fact not in tree: {fact.content_hash!r}") from exc
        return self._proof_by_index(index)

    def _proof_by_index(self, index: int) -> MerkleProof:
        path: list[tuple[str, bool]] = []
        current_index = index
        for level in self._levels[:-1]:
            level_size = len(level)
            if current_index % 2 == 0:
                sibling_index = current_index + 1
                sibling_on_right = True
                if sibling_index >= level_size:
                    # Odd level — node is its own sibling.
                    sibling_index = current_index
            else:
                sibling_index = current_index - 1
                sibling_on_right = False
            path.append((level[sibling_index], sibling_on_right))
            current_index //= 2
        return MerkleProof(
            leaf_hash=self._facts[index].content_hash,
            path=tuple(path),
            root_hash=self.root,
        )


def _combine_level(level: list[str]) -> list[str]:
    """Fold a level up one step, duplicating the last node when
    the level has an odd count."""
    padded = list(level)
    if len(padded) % 2 == 1:
        padded.append(padded[-1])
    out: list[str] = []
    for i in range(0, len(padded), 2):
        out.append(_hash_pair(padded[i], padded[i + 1]))
    return out


def _hash_pair(left: str, right: str) -> str:
    return hashlib.sha256((left + right).encode("utf-8")).hexdigest()
