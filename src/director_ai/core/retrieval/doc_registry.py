# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Document Registry

"""In-memory document metadata registry with tenant isolation."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger("DirectorAI.DocRegistry")


@dataclass
class DocRecord:
    """Metadata for one tenant-scoped ingested document."""

    doc_id: str
    source: str
    tenant_id: str
    created_at: float
    updated_at: float
    chunk_count: int
    chunk_ids: list[str] = field(default_factory=list)
    content_hash: str = ""


def _require_non_empty_string(value: str, field_name: str) -> str:
    """Return stripped text after enforcing a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_chunk_ids(chunk_ids: object) -> list[str]:
    """Validate an untrusted collection and return stripped chunk identifiers.

    Parameters
    ----------
    chunk_ids : object
        Candidate sequence; whole strings, bytes and nonsequences are rejected.

    Returns
    -------
    list[str]
        A nonempty list whose members are validated, stripped strings.

    Raises
    ------
    ValueError
        The collection is empty, not a supported sequence, or contains a
        nonstring or blank identifier.
    """
    if isinstance(chunk_ids, str | bytes) or not isinstance(chunk_ids, Sequence):
        raise ValueError("chunk_ids must be a non-empty sequence of strings")
    normalized = list(chunk_ids)
    if not normalized:
        raise ValueError("chunk_ids must be a non-empty sequence of strings")
    validated: list[str] = []
    for chunk_id in normalized:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_ids must be a non-empty sequence of strings")
        validated.append(chunk_id)
    return [chunk_id.strip() for chunk_id in validated]


def _snapshot(record: DocRecord) -> DocRecord:
    """Return a detached copy of a registry record."""
    return DocRecord(
        doc_id=record.doc_id,
        source=record.source,
        tenant_id=record.tenant_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        chunk_count=record.chunk_count,
        chunk_ids=list(record.chunk_ids),
        content_hash=record.content_hash,
    )


class DocRegistry:
    """Thread-safe document metadata store."""

    def __init__(self) -> None:
        self._docs: dict[str, DocRecord] = {}
        self._lock = threading.Lock()

    def register(
        self,
        doc_id: str,
        source: str,
        tenant_id: str,
        chunk_ids: object,
        *,
        content_hash: str = "",
    ) -> DocRecord:
        """Register a document under an unused identifier and return a snapshot.

        Parameters
        ----------
        doc_id : str
            Nonblank identifier, stripped and unique across this registry.
        source : str
            Nonblank source label stored after stripping surrounding whitespace.
        tenant_id : str
            Nonblank tenant label retained for matching reads.
        chunk_ids : object
            Untrusted nonempty sequence validated as nonblank string identifiers.
        content_hash : str, optional
            Digest label, stripped without validating its algorithm.

        Returns
        -------
        DocRecord
            Detached snapshot with creation/update timestamps and copied chunks.

        Raises
        ------
        ValueError
            Required fields or chunk identifiers are invalid, or the identifier
            already exists.

        Notes
        -----
        Registration does not authenticate callers or check tenant access policy.
        """
        doc_id = _require_non_empty_string(doc_id, "doc_id")
        source = _require_non_empty_string(source, "source")
        tenant_id = _require_non_empty_string(tenant_id, "tenant_id")
        normalized_chunk_ids = _require_chunk_ids(chunk_ids)
        now = time.time()
        record = DocRecord(
            doc_id=doc_id,
            source=source,
            tenant_id=tenant_id,
            created_at=now,
            updated_at=now,
            chunk_count=len(normalized_chunk_ids),
            chunk_ids=normalized_chunk_ids,
            content_hash=content_hash.strip(),
        )
        with self._lock:
            if doc_id in self._docs:
                raise ValueError(f"Document {doc_id!r} already registered")
            self._docs[doc_id] = record
        logger.info(
            "Registered doc %s (%d chunks, tenant=%s)",
            doc_id,
            len(normalized_chunk_ids),
            tenant_id,
        )
        return _snapshot(record)

    def update(
        self,
        doc_id: str,
        chunk_ids: object,
        source: str | None = None,
        *,
        content_hash: str | None = None,
    ) -> DocRecord:
        """Replace a document's chunks and supplied optional metadata.

        Parameters
        ----------
        doc_id : str
            Existing nonblank document identifier.
        chunk_ids : object
            Untrusted nonempty sequence validated as nonblank string identifiers.
        source : str or None, optional
            Replacement source; None retains the current source.
        content_hash : str or None, optional
            Replacement stripped hash label; None retains the current label.

        Returns
        -------
        DocRecord
            Detached snapshot with updated chunks and update timestamp.

        Raises
        ------
        ValueError
            The identifier, supplied source or chunk collection is invalid.
        KeyError
            The document identifier is not registered.

        Notes
        -----
        This method does not check a tenant or authenticate the caller.
        """
        doc_id = _require_non_empty_string(doc_id, "doc_id")
        normalized_chunk_ids = _require_chunk_ids(chunk_ids)
        if source is not None:
            source = _require_non_empty_string(source, "source")
        with self._lock:
            record = self._docs.get(doc_id)
            if record is None:
                raise KeyError(f"Document {doc_id!r} not found")
            if source is not None:
                record.source = source
            if content_hash is not None:
                record.content_hash = content_hash.strip()
            record.chunk_ids = normalized_chunk_ids
            record.chunk_count = len(normalized_chunk_ids)
            record.updated_at = time.time()
            return _snapshot(record)

    def delete(self, doc_id: str) -> DocRecord | None:
        """Delete a document record by id and return its last snapshot."""
        doc_id = _require_non_empty_string(doc_id, "doc_id")
        with self._lock:
            record = self._docs.pop(doc_id, None)
        if record:
            logger.info("Deleted doc %s (%d chunks)", doc_id, record.chunk_count)
            return _snapshot(record)
        return None

    def get(self, doc_id: str, tenant_id: str) -> DocRecord | None:
        """Return a document snapshot only when it belongs to the tenant."""
        doc_id = _require_non_empty_string(doc_id, "doc_id")
        tenant_id = _require_non_empty_string(tenant_id, "tenant_id")
        with self._lock:
            record = self._docs.get(doc_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return _snapshot(record)

    def list_for_tenant(self, tenant_id: str) -> list[DocRecord]:
        """Return detached document snapshots for a tenant."""
        tenant_id = _require_non_empty_string(tenant_id, "tenant_id")
        with self._lock:
            return [
                _snapshot(r) for r in self._docs.values() if r.tenant_id == tenant_id
            ]

    def exists(self, doc_id: str) -> bool:
        """Return whether a document id is registered."""
        doc_id = _require_non_empty_string(doc_id, "doc_id")
        with self._lock:
            return doc_id in self._docs

    @property
    def count(self) -> int:
        """Return the number of registered documents."""
        with self._lock:
            return len(self._docs)
