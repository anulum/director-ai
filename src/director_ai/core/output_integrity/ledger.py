# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Tamper-evident output ledger

"""An append-only, hash-chained ledger of output interactions.

Each entry commits to the SHA-256 digest of its (canonicalised) payload and to
the previous entry's hash, so the whole chain is tamper-evident: altering,
removing, or reordering any past entry breaks every hash from that point on, and
:meth:`TamperEvidentLedger.verify` recomputes the chain to detect it. Only
digests are stored — never the raw payload — so the ledger is tenant-safe and can
be exported for audit without leaking interaction content.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["GENESIS_HASH", "LedgerEntry", "TamperEvidentLedger"]

GENESIS_HASH = "00" * 32
"""The previous-hash of the first entry — a chain anchored to a fixed root."""


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _link(prev_hash: str, payload_digest: str) -> str:
    return hashlib.sha256(
        bytes.fromhex(prev_hash) + bytes.fromhex(payload_digest)
    ).hexdigest()


@dataclass(frozen=True)
class LedgerEntry:
    """One link in the tamper-evident chain (digests only, no raw payload)."""

    index: int
    payload_digest: str
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a tenant-safe JSON dict."""
        return {
            "index": self.index,
            "payload_digest": self.payload_digest,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


class TamperEvidentLedger:
    """An append-only hash chain over output-interaction digests."""

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """The chain so far, oldest first."""
        return tuple(self._entries)

    @property
    def head(self) -> str:
        """The most recent entry hash, or :data:`GENESIS_HASH` when empty."""
        return self._entries[-1].entry_hash if self._entries else GENESIS_HASH

    def append(self, payload: Mapping[str, Any]) -> LedgerEntry:
        """Commit ``payload``'s digest to the chain and return the new entry."""
        digest = _canonical_digest(payload)
        prev = self.head
        entry = LedgerEntry(
            index=len(self._entries),
            payload_digest=digest,
            prev_hash=prev,
            entry_hash=_link(prev, digest),
        )
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        """Recompute the chain and report whether it is intact."""
        return self.verify_entries(self._entries)

    @staticmethod
    def verify_entries(entries: list[LedgerEntry] | tuple[LedgerEntry, ...]) -> bool:
        """Verify an exported list of entries forms an unbroken hash chain."""
        prev = GENESIS_HASH
        for position, entry in enumerate(entries):
            if entry.index != position:
                return False
            if entry.prev_hash != prev:
                return False
            if entry.entry_hash != _link(prev, entry.payload_digest):
                return False
            prev = entry.entry_hash
        return True
