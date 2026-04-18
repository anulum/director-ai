# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HMAC audit chain

"""Hash-chained audit log.

Each :class:`AuditEntry` records a parent hash, an event hash,
and an HMAC-SHA-256 tag keyed on the caller's secret. Tampering
with any entry breaks the chain — :meth:`AuditChain.verify`
re-derives every hash and tag and reports the first tampered
entry.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

_ZERO_HASH = "0" * 64


@dataclass(frozen=True)
class AuditEntry:
    """One link in the chain.

    ``event`` is free-form caller-supplied metadata. ``event_hash``
    is SHA-256 over the canonical event JSON. ``parent_hash`` ties
    this entry to the previous one (``"0" * 64`` for the first
    entry). ``tag`` is HMAC-SHA-256 over ``parent_hash +
    event_hash`` — tampering with either breaks the tag.
    """

    index: int
    timestamp: float
    event: Mapping[str, object]
    event_hash: str
    parent_hash: str
    tag: str

    def canonical_event(self) -> bytes:
        return json.dumps(self.event, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )


class AuditChain:
    """Thread-safe hash-chained audit log.

    Parameters
    ----------
    secret :
        HMAC key. Must be at least 32 bytes.
    clock :
        Timestamp source. Injection point for tests.
    """

    def __init__(
        self,
        *,
        secret: bytes,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if len(secret) < 32:
            raise ValueError(f"secret must be at least 32 bytes; got {len(secret)}")
        self._secret = secret
        self._clock = clock or time.time
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()

    def append(self, event: Mapping[str, object]) -> AuditEntry:
        """Append ``event`` and return the resulting entry."""
        if not isinstance(event, Mapping):  # pragma: no cover — defensive
            raise TypeError("event must be a Mapping")
        with self._lock:
            parent_hash = self._entries[-1].event_hash if self._entries else _ZERO_HASH
            entry = self._build_entry(
                index=len(self._entries),
                event=dict(event),
                parent_hash=parent_hash,
            )
            self._entries.append(entry)
            return entry

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def snapshot(self) -> tuple[AuditEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def verify(self) -> tuple[bool, int | None]:
        """Return ``(ok, first_bad_index)``. ``ok`` is ``False`` when
        any entry has been tampered with; ``first_bad_index`` is
        the 0-based index of the tampered entry (``None`` on success).
        """
        with self._lock:
            entries = tuple(self._entries)
        previous_hash = _ZERO_HASH
        for entry in entries:
            event_hash = self._hash_event(entry.event)
            if event_hash != entry.event_hash:
                return False, entry.index
            if entry.parent_hash != previous_hash:
                return False, entry.index
            expected_tag = self._hmac(entry.parent_hash, entry.event_hash)
            if not hmac.compare_digest(expected_tag, entry.tag):
                return False, entry.index
            previous_hash = entry.event_hash
        return True, None

    def _build_entry(
        self,
        *,
        index: int,
        event: dict[str, object],
        parent_hash: str,
    ) -> AuditEntry:
        event_hash = self._hash_event(event)
        tag = self._hmac(parent_hash, event_hash)
        return AuditEntry(
            index=index,
            timestamp=float(self._clock()),
            event=event,
            event_hash=event_hash,
            parent_hash=parent_hash,
            tag=tag,
        )

    @staticmethod
    def _hash_event(event: Mapping[str, object]) -> str:
        payload = json.dumps(event, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    def _hmac(self, parent_hash: str, event_hash: str) -> str:
        message = (parent_hash + event_hash).encode("utf-8")
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()
