# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — operational knowledge-provenance ledger

"""Persistent, tamper-evident ledger of knowledge-base mutations.

Where :mod:`director_ai.core.provenance.verifier` proves the citation set
of a single response, this ledger records the *lifecycle of the knowledge
base itself*: every ingest, update, and delete is appended as a signed
:class:`LedgerEvent`. Two integrity layers compose:

* **Content commitment.** Each event carries a Merkle root
  (:mod:`content_commitment`) over the per-chunk content digests it
  admitted or removed. Editing a stored chunk after the fact breaks the
  inclusion proof returned by :meth:`KnowledgeProvenanceLedger.provenance_of`.
* **HMAC chain.** Each event is folded into a
  :class:`~director_ai.core.provenance.chain.ProvenanceChain` keyed on an
  HMAC of the event's full semantic payload, so reordering, deleting, or
  editing any event breaks :meth:`KnowledgeProvenanceLedger.verify`.

Events persist as one JSON object per line. The ledger reloads and
verifies the file on construction, so a process restart resumes the exact
chain — and a tampered file is rejected before any new event is appended.
The query surface answers the operational questions an auditor asks:
*where did this chunk come from* (:meth:`provenance_of`) and *what
happened to this document* (:meth:`history_for`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .chain import ProvenanceChain
from .content_commitment import InclusionProof, commit_root, prove_inclusion

__all__ = [
    "ChunkProvenance",
    "KnowledgeProvenanceLedger",
    "LedgerEvent",
    "LedgerTamperError",
]

_EVENT_TYPES = frozenset({"ingest", "update", "delete", "supersede"})
# Events that retire (rather than admit) their referenced chunks.
_RETIRING_EVENTS = frozenset({"delete", "supersede"})


class LedgerTamperError(ValueError):
    """Raised when a persisted ledger fails its integrity check on load."""


@dataclass(frozen=True)
class LedgerEvent:
    """One signed knowledge-base mutation.

    ``chunk_ids`` are the chunks this event *admitted*; ``leaf_hashes`` is
    the parallel list of their content digests (hex), the leaves of the
    event's content commitment. ``removed_chunk_ids`` are chunks this
    event retired (an update retires the previous revision's chunks; a
    delete retires all of a document's chunks). ``supersedes`` records
    document-level lineage for higher layers. ``index``, ``parent_hash``,
    and ``tag`` are the HMAC-chain fields and are derived, not
    caller-supplied.
    """

    index: int
    event_type: str
    doc_id: str
    tenant_id: str
    source: str
    content_hash: str
    content_root: str
    chunk_ids: tuple[str, ...]
    leaf_hashes: tuple[str, ...]
    removed_chunk_ids: tuple[str, ...]
    supersedes: tuple[str, ...]
    timestamp: float
    parent_hash: str
    tag: str

    def __post_init__(self) -> None:
        """Validate event type, chunk leaves, and chain index."""
        if self.event_type not in _EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {sorted(_EVENT_TYPES)}; "
                f"got {self.event_type!r}"
            )
        if len(self.chunk_ids) != len(self.leaf_hashes):
            raise ValueError("chunk_ids and leaf_hashes must be the same length")
        if self.index < 0:
            raise ValueError("index must be non-negative")

    def to_json(self) -> str:
        """Serialise the full event (including chain fields) as one line."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> LedgerEvent:
        """Parse a persisted event line back into a :class:`LedgerEvent`."""
        data = json.loads(raw)
        return cls(
            index=int(data["index"]),
            event_type=str(data["event_type"]),
            doc_id=str(data["doc_id"]),
            tenant_id=str(data["tenant_id"]),
            source=str(data["source"]),
            content_hash=str(data["content_hash"]),
            content_root=str(data["content_root"]),
            chunk_ids=tuple(data["chunk_ids"]),
            leaf_hashes=tuple(data["leaf_hashes"]),
            removed_chunk_ids=tuple(data.get("removed_chunk_ids", ())),
            supersedes=tuple(data.get("supersedes", ())),
            timestamp=float(data["timestamp"]),
            parent_hash=str(data["parent_hash"]),
            tag=str(data["tag"]),
        )


@dataclass(frozen=True)
class ChunkProvenance:
    """The active origin of one chunk plus its inclusion proof."""

    chunk_id: str
    doc_id: str
    tenant_id: str
    source: str
    event_index: int
    event_type: str
    timestamp: float
    proof: InclusionProof

    @property
    def verified(self) -> bool:
        """Return ``True`` when the inclusion proof folds to the root."""
        return self.proof.verify()


class KnowledgeProvenanceLedger:
    """Append-only, HMAC-chained, persistent knowledge-mutation ledger.

    Parameters
    ----------
    secret :
        HMAC key for the underlying chain. Minimum 32 bytes.
    path :
        JSONL file the events persist to. When the file already exists it
        is loaded and verified on construction; a tampered file raises
        :class:`LedgerTamperError`. Pass ``None`` for an in-memory ledger
        (CI, ephemeral workers).
    clock :
        Timestamp source; injection point for deterministic tests.
    """

    def __init__(
        self,
        *,
        secret: bytes,
        path: str | os.PathLike[str] | None = None,
        clock: object = None,
    ) -> None:
        self._secret = secret
        self._path = Path(path) if path is not None else None
        self._clock = clock if callable(clock) else _wall_clock
        self._lock = threading.Lock()
        self._chain = ProvenanceChain(secret=secret)
        self._events: list[LedgerEvent] = []
        # Active chunk -> admitting event index, maintained as events are
        # appended so provenance_of() is an O(1) lookup.
        self._active: dict[str, int] = {}
        if self._path is not None and self._path.exists():
            self._load_and_verify()

    # -- recording -------------------------------------------------------

    def record_ingest(
        self,
        *,
        doc_id: str,
        tenant_id: str,
        source: str,
        content_hash: str,
        chunk_leaves: Sequence[tuple[str, bytes]],
        supersedes: Sequence[str] = (),
    ) -> LedgerEvent:
        """Append an ``ingest`` event admitting ``chunk_leaves``.

        ``chunk_leaves`` pairs each new chunk id with its content digest
        (32 raw bytes). Raises :class:`ValueError` when the chunk set is
        empty.
        """
        return self._append(
            event_type="ingest",
            doc_id=doc_id,
            tenant_id=tenant_id,
            source=source,
            content_hash=content_hash,
            chunk_leaves=chunk_leaves,
            removed_chunk_ids=(),
            supersedes=supersedes,
        )

    def record_update(
        self,
        *,
        doc_id: str,
        tenant_id: str,
        source: str,
        content_hash: str,
        chunk_leaves: Sequence[tuple[str, bytes]],
        removed_chunk_ids: Sequence[str] = (),
        supersedes: Sequence[str] = (),
    ) -> LedgerEvent:
        """Append an ``update`` event.

        The event admits new chunks and retires the previous revision's
        ``removed_chunk_ids``.
        """
        return self._append(
            event_type="update",
            doc_id=doc_id,
            tenant_id=tenant_id,
            source=source,
            content_hash=content_hash,
            chunk_leaves=chunk_leaves,
            removed_chunk_ids=removed_chunk_ids,
            supersedes=supersedes,
        )

    def record_delete(
        self,
        *,
        doc_id: str,
        tenant_id: str,
        removed_chunk_ids: Sequence[str],
        source: str = "",
    ) -> LedgerEvent:
        """Append a ``delete`` event retiring ``removed_chunk_ids``.

        The content commitment is taken over the retired chunk ids so the
        delete is itself bound to the exact set it removed.
        """
        removed = tuple(removed_chunk_ids)
        if not removed:
            raise ValueError("delete must remove at least one chunk")
        leaves = [(chunk_id, _digest_text(chunk_id)) for chunk_id in removed]
        return self._append(
            event_type="delete",
            doc_id=doc_id,
            tenant_id=tenant_id,
            source=source,
            content_hash="",
            chunk_leaves=leaves,
            removed_chunk_ids=removed,
            supersedes=(),
            admit=False,
        )

    def record_supersede(
        self,
        *,
        doc_id: str,
        tenant_id: str,
        source: str,
        supersedes: Sequence[str],
        removed_chunk_ids: Sequence[str],
    ) -> LedgerEvent:
        """Append a ``supersede`` event: ``doc_id`` replaces ``supersedes``.

        ``supersedes`` records the document-level lineage; the retired
        ``removed_chunk_ids`` are committed and dropped from the active set,
        so a chunk of a superseded document no longer resolves through
        :meth:`provenance_of`.
        """
        superseded = tuple(supersedes)
        if not superseded:
            raise ValueError("supersede must name at least one superseded document")
        removed = tuple(removed_chunk_ids)
        if not removed:
            raise ValueError("supersede must remove at least one chunk")
        leaves = [(chunk_id, _digest_text(chunk_id)) for chunk_id in removed]
        return self._append(
            event_type="supersede",
            doc_id=doc_id,
            tenant_id=tenant_id,
            source=source,
            content_hash="",
            chunk_leaves=leaves,
            removed_chunk_ids=removed,
            supersedes=superseded,
            admit=False,
        )

    # -- queries ---------------------------------------------------------

    def provenance_of(self, chunk_id: str) -> ChunkProvenance | None:
        """Return the active origin of ``chunk_id`` with an inclusion proof.

        Returns ``None`` when the chunk was never admitted or has since
        been retired by an update or delete.
        """
        with self._lock:
            event_index = self._active.get(chunk_id)
            if event_index is None:
                return None
            event = self._events[event_index]
        leaves = [bytes.fromhex(leaf) for leaf in event.leaf_hashes]
        position = event.chunk_ids.index(chunk_id)
        proof = prove_inclusion(leaves, position)
        return ChunkProvenance(
            chunk_id=chunk_id,
            doc_id=event.doc_id,
            tenant_id=event.tenant_id,
            source=event.source,
            event_index=event.index,
            event_type=event.event_type,
            timestamp=event.timestamp,
            proof=proof,
        )

    def history_for(
        self, doc_id: str, *, tenant_id: str | None = None
    ) -> tuple[LedgerEvent, ...]:
        """Return every event for ``doc_id`` in chain order."""
        with self._lock:
            return tuple(
                event
                for event in self._events
                if event.doc_id == doc_id
                and (tenant_id is None or event.tenant_id == tenant_id)
            )

    def verify(self) -> tuple[bool, int | None]:
        """Re-derive the chain over the persisted events.

        Returns ``(ok, first_bad_index)``. ``ok`` is ``True`` only when
        every event's HMAC tag and parent hash match a fresh replay.
        """
        with self._lock:
            events = tuple(self._events)
        return _replay_verify(self._secret, events)

    def snapshot(self) -> tuple[LedgerEvent, ...]:
        """Return a point-in-time copy of every event."""
        with self._lock:
            return tuple(self._events)

    def __len__(self) -> int:
        """Return the number of ledger events."""
        with self._lock:
            return len(self._events)

    # -- internals -------------------------------------------------------

    def _append(
        self,
        *,
        event_type: str,
        doc_id: str,
        tenant_id: str,
        source: str,
        content_hash: str,
        chunk_leaves: Sequence[tuple[str, bytes]],
        removed_chunk_ids: Sequence[str],
        supersedes: Sequence[str],
        admit: bool = True,
    ) -> LedgerEvent:
        chunk_ids, leaves = _split_chunk_leaves(chunk_leaves)
        content_root = commit_root(leaves).hex()
        removed = tuple(removed_chunk_ids)
        with self._lock:
            timestamp = float(self._clock())
            digest = _event_digest(
                event_type=event_type,
                doc_id=doc_id,
                tenant_id=tenant_id,
                source=source,
                content_hash=content_hash,
                content_root=content_root,
                chunk_ids=chunk_ids,
                leaf_hashes=tuple(leaf.hex() for leaf in leaves),
                removed_chunk_ids=removed,
                supersedes=tuple(supersedes),
                timestamp=timestamp,
            )
            entry = self._chain.append(merkle_root=digest)
            event = LedgerEvent(
                index=entry.index,
                event_type=event_type,
                doc_id=doc_id,
                tenant_id=tenant_id,
                source=source,
                content_hash=content_hash,
                content_root=content_root,
                chunk_ids=chunk_ids,
                leaf_hashes=tuple(leaf.hex() for leaf in leaves),
                removed_chunk_ids=removed,
                supersedes=tuple(supersedes),
                timestamp=timestamp,
                parent_hash=entry.parent_hash,
                tag=entry.tag,
            )
            self._events.append(event)
            self._apply_active(event, admit=admit)
            self._persist(event)
        return event

    def _apply_active(self, event: LedgerEvent, *, admit: bool) -> None:
        """Update the active chunk map for one freshly-appended event."""
        for chunk_id in event.removed_chunk_ids:
            self._active.pop(chunk_id, None)
        if admit:
            for chunk_id in event.chunk_ids:
                self._active[chunk_id] = event.index

    def _persist(self, event: LedgerEvent) -> None:
        """Append one event line to the JSONL file via O_APPEND."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(event.to_json() + "\n")

    def _load_and_verify(self) -> None:
        """Load persisted events, verify the chain, and rebuild state."""
        assert self._path is not None
        events: list[LedgerEvent] = []
        with open(self._path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if line:
                    events.append(LedgerEvent.from_json(line))
        ok, bad_index = _replay_verify(self._secret, tuple(events))
        if not ok:
            raise LedgerTamperError(
                f"ledger {self._path} failed integrity check at event {bad_index}"
            )
        for event in events:
            self._chain.append(merkle_root=_digest_event(event))
            self._events.append(event)
            self._apply_active(event, admit=event.event_type not in _RETIRING_EVENTS)


def _replay_verify(
    secret: bytes, events: Sequence[LedgerEvent]
) -> tuple[bool, int | None]:
    """Replay ``events`` into a fresh chain and compare stored tags."""
    chain = ProvenanceChain(secret=secret)
    for event in events:
        expected = chain.append(merkle_root=_digest_event(event))
        if (
            expected.index != event.index
            or expected.parent_hash != event.parent_hash
            or not hmac.compare_digest(expected.tag, event.tag)
        ):
            return False, event.index
    return True, None


def _digest_event(event: LedgerEvent) -> str:
    """Return the HMAC-chain digest over an event's semantic fields.

    Excludes the derived chain fields (``index``, ``parent_hash``, ``tag``),
    which the chain binds itself.
    """
    return _event_digest(
        event_type=event.event_type,
        doc_id=event.doc_id,
        tenant_id=event.tenant_id,
        source=event.source,
        content_hash=event.content_hash,
        content_root=event.content_root,
        chunk_ids=event.chunk_ids,
        leaf_hashes=event.leaf_hashes,
        removed_chunk_ids=event.removed_chunk_ids,
        supersedes=event.supersedes,
        timestamp=event.timestamp,
    )


def _event_digest(
    *,
    event_type: str,
    doc_id: str,
    tenant_id: str,
    source: str,
    content_hash: str,
    content_root: str,
    chunk_ids: Sequence[str],
    leaf_hashes: Sequence[str],
    removed_chunk_ids: Sequence[str],
    supersedes: Sequence[str],
    timestamp: float,
) -> str:
    """Return a stable hex SHA-256 over an event's semantic payload."""
    payload = {
        "event_type": event_type,
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "source": source,
        "content_hash": content_hash,
        "content_root": content_root,
        "chunk_ids": list(chunk_ids),
        "leaf_hashes": list(leaf_hashes),
        "removed_chunk_ids": list(removed_chunk_ids),
        "supersedes": list(supersedes),
        "timestamp": timestamp,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _split_chunk_leaves(
    chunk_leaves: Sequence[tuple[str, bytes]],
) -> tuple[tuple[str, ...], list[bytes]]:
    """Validate and split ``(chunk_id, digest)`` pairs into parallel lists."""
    materialised = list(chunk_leaves)
    if not materialised:
        raise ValueError("event must reference at least one chunk")
    chunk_ids: list[str] = []
    leaves: list[bytes] = []
    for chunk_id, digest in materialised:
        if not chunk_id:
            raise ValueError("chunk_id must be non-empty")
        if not isinstance(digest, bytes | bytearray) or not digest:
            raise ValueError(f"chunk {chunk_id!r} digest must be non-empty bytes")
        chunk_ids.append(str(chunk_id))
        leaves.append(bytes(digest))
    return tuple(chunk_ids), leaves


def _digest_text(text: str) -> bytes:
    """Return the SHA-256 digest of ``text`` as raw bytes."""
    return hashlib.sha256(text.encode("utf-8")).digest()


def _wall_clock() -> float:
    """Return the current wall-clock time. Indirection eases test seams."""
    import time

    return time.time()
