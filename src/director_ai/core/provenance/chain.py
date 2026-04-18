# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ProvenanceChain

"""HMAC-signed provenance chain.

Each :class:`ProvenanceEntry` wraps a :class:`MerkleTree` root
with a monotonic index, a parent-entry hash, and an HMAC-SHA-256
tag over ``(parent_hash || merkle_root)``. The chain thus
records both the order of responses and the Merkle root of each
response's citation set. Tampering with any entry breaks the
tag; reordering entries breaks the parent chain.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
from dataclasses import dataclass

_ZERO_HASH = "0" * 64


class HmacChainError(ValueError):
    """Raised when the chain's HMAC or parent hash fails
    verification."""


@dataclass(frozen=True)
class ProvenanceEntry:
    """One entry in the chain."""

    index: int
    merkle_root: str
    parent_hash: str
    tag: str


class ProvenanceChain:
    """Thread-safe HMAC-signed chain.

    Parameters
    ----------
    secret :
        HMAC key. Minimum 32 bytes.
    """

    def __init__(self, *, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("secret must be at least 32 bytes")
        self._secret = secret
        self._lock = threading.Lock()
        self._entries: list[ProvenanceEntry] = []

    def append(self, *, merkle_root: str) -> ProvenanceEntry:
        """Append a new entry for the response's Merkle root."""
        if not merkle_root:
            raise ValueError("merkle_root must be non-empty")
        with self._lock:
            parent_hash = (
                _digest_entry(self._entries[-1]) if self._entries else _ZERO_HASH
            )
            tag = _hmac(self._secret, parent_hash, merkle_root)
            entry = ProvenanceEntry(
                index=len(self._entries),
                merkle_root=merkle_root,
                parent_hash=parent_hash,
                tag=tag,
            )
            self._entries.append(entry)
            return entry

    def snapshot(self) -> tuple[ProvenanceEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def verify(self) -> tuple[bool, int | None]:
        """Return ``(ok, first_bad_index)``. Walks the entire
        chain re-deriving each parent hash and HMAC tag."""
        with self._lock:
            entries = tuple(self._entries)
        previous_hash = _ZERO_HASH
        for entry in entries:
            if entry.parent_hash != previous_hash:
                return False, entry.index
            expected_tag = _hmac(self._secret, entry.parent_hash, entry.merkle_root)
            if not hmac.compare_digest(expected_tag, entry.tag):
                return False, entry.index
            previous_hash = _digest_entry(entry)
        return True, None


def _digest_entry(entry: ProvenanceEntry) -> str:
    payload = (
        f"{entry.index}|{entry.merkle_root}|{entry.parent_hash}|{entry.tag}".encode()
    )
    return hashlib.sha256(payload).hexdigest()


def _hmac(secret: bytes, parent_hash: str, merkle_root: str) -> str:
    message = (parent_hash + merkle_root).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()
