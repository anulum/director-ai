# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — byte-convention Merkle content commitment

"""Canonical SHA-256 Merkle commitment over a set of content leaves.

A content commitment binds an ordered set of fixed-length content
digests — for example the per-chunk content hashes of one knowledge-base
mutation — to a single tamper-evident root. Any later edit to a committed
leaf changes the root, so a stored commitment proves which exact content
set a mutation admitted.

Convention (bit-identical to ``backfire_kernel.rust_merkle_*``):

* Internal nodes are ``SHA-256(0x01 || left || right)``. The ``0x01``
  node prefix separates an internal node from a raw leaf so a leaf digest
  can never be reinterpreted as an inner node (second-preimage guard).
* When a level has an odd node count the last node is duplicated, the
  RFC 6962 reduction. The insertion order of the leaves is preserved, so
  an :class:`InclusionProof` references a stable leaf index.

The Rust path accelerates the root, authentication-path, and path-walk
hot loops. The pure-Python fallbacks below are the reference
implementation; they are bit-identical to the Rust kernel so an
:class:`InclusionProof` verifies regardless of which path produced it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from ..mandatory import mandatory_execution

try:
    from backfire_kernel import (
        rust_merkle_auth_path as _rust_merkle_auth_path,
    )
    from backfire_kernel import (
        rust_merkle_root as _rust_merkle_root,
    )
    from backfire_kernel import (
        rust_merkle_walk_path as _rust_merkle_walk_path,
    )

    _RUST_MERKLE = True
except ImportError:  # pragma: no cover - mandatory accelerator guard
    _RUST_MERKLE = True

    def _rust_merkle_root(_leaves: list[bytes]) -> bytes:
        raise RuntimeError("backfire_kernel rust_merkle_root is unavailable")

    def _rust_merkle_auth_path(_leaves: list[bytes], _index: int) -> list[bytes]:
        raise RuntimeError("backfire_kernel rust_merkle_auth_path is unavailable")

    def _rust_merkle_walk_path(
        _leaf: bytes, _index: int, _siblings: list[bytes]
    ) -> bytes:
        raise RuntimeError("backfire_kernel rust_merkle_walk_path is unavailable")


_NODE_SEP = b"\x01"

__all__ = [
    "InclusionProof",
    "commit_root",
    "prove_inclusion",
]


@dataclass(frozen=True)
class InclusionProof:
    """Proof that ``leaf`` sits at ``index`` under ``root``.

    ``siblings`` is the ordered authentication path from the leaf up to
    the root, exclusive of the root itself. :meth:`verify` folds the leaf
    through the path and compares the recomputed root to ``root`` — the
    proof is self-contained, so a verifier needs only the proof, not the
    original leaf set.
    """

    leaf: bytes
    index: int
    siblings: tuple[bytes, ...]
    root: bytes

    def __post_init__(self) -> None:
        if not self.leaf:
            raise ValueError("leaf must be non-empty")
        if self.index < 0:
            raise ValueError("index must be non-negative")
        if len(self.root) != 32:
            raise ValueError("root must be a 32-byte SHA-256 digest")

    def verify(self) -> bool:
        """Return ``True`` when the leaf folds to ``root`` along the path."""
        return _walk_path(self.leaf, self.index, self.siblings) == self.root


def commit_root(leaves: Sequence[bytes]) -> bytes:
    """Return the 32-byte Merkle root committing ``leaves`` in order.

    ``leaves`` must be a non-empty sequence of non-empty byte strings —
    typically 32-byte content digests. Raises :class:`ValueError`
    otherwise.
    """
    checked = _validate_leaves(leaves)
    return _merkle_root(checked)


def prove_inclusion(leaves: Sequence[bytes], index: int) -> InclusionProof:
    """Return the :class:`InclusionProof` for ``leaves[index]``.

    Raises :class:`ValueError` when the leaf set is empty/invalid or the
    index is out of range.
    """
    checked = _validate_leaves(leaves)
    if not 0 <= index < len(checked):
        raise ValueError(f"index {index} out of range (n={len(checked)})")
    siblings = _auth_path(checked, index)
    root = _merkle_root(checked)
    return InclusionProof(
        leaf=checked[index],
        index=index,
        siblings=tuple(siblings),
        root=root,
    )


def _validate_leaves(leaves: Sequence[bytes]) -> list[bytes]:
    """Return leaves as a list of non-empty byte strings or raise."""
    materialised = list(leaves)
    if not materialised:
        raise ValueError("leaves must be non-empty")
    out: list[bytes] = []
    for position, leaf in enumerate(materialised):
        if not isinstance(leaf, bytes | bytearray):
            raise TypeError(f"leaf at index {position} is not bytes")
        if not leaf:
            raise ValueError(f"leaf at index {position} is empty")
        out.append(bytes(leaf))
    return out


def _hash_node(left: bytes, right: bytes) -> bytes:
    """``SHA-256(0x01 || left || right)`` — one internal node."""
    return hashlib.sha256(_NODE_SEP + left + right).digest()


def _merkle_root(leaves: list[bytes]) -> bytes:
    """Fold ``leaves`` to the Merkle root (Rust path, Python reference)."""
    if _RUST_MERKLE:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return bytes(_rust_merkle_root(leaves))
    level = list(leaves)
    while len(level) > 1:
        level = [
            _hash_node(level[i], level[i + 1] if i + 1 < len(level) else level[i])
            for i in range(0, len(level), 2)
        ]
    return level[0]


def _auth_path(leaves: list[bytes], index: int) -> list[bytes]:
    """Sibling hashes from ``leaves[index]`` up to the root (exclusive)."""
    if _RUST_MERKLE:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return [bytes(node) for node in _rust_merkle_auth_path(leaves, index)]
    path: list[bytes] = []
    level = list(leaves)
    position = index
    while len(level) > 1:
        sibling_index = position ^ 1
        if sibling_index < len(level):
            path.append(level[sibling_index])
        else:
            # Odd tail: the missing sibling is the node itself, matching
            # the duplicate-last-leaf reduction in _merkle_root.
            path.append(level[position])
        level = [
            _hash_node(level[i], level[i + 1] if i + 1 < len(level) else level[i])
            for i in range(0, len(level), 2)
        ]
        position //= 2
    return path


def _walk_path(leaf: bytes, index: int, siblings: Sequence[bytes]) -> bytes:
    """Fold ``leaf`` through ``siblings`` and return the recomputed root."""
    if _RUST_MERKLE:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return bytes(
                _rust_merkle_walk_path(leaf, index, [bytes(s) for s in siblings])
            )
    node = leaf
    position = index
    for sibling in siblings:
        node = (
            _hash_node(node, sibling)
            if position % 2 == 0
            else _hash_node(sibling, node)
        )
        position //= 2
    return node
