# SPDX-License-Identifier: AGPL-3.0-or-later
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
    doc_id: str
    source: str
    tenant_id: str
    created_at: float
    updated_at: float
    chunk_count: int
    chunk_ids: list[str] = field(default_factory=list)


def _require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_chunk_ids(chunk_ids: Sequence[str]) -> list[str]:
    if isinstance(chunk_ids, str | bytes) or not isinstance(chunk_ids, Sequence):
        raise ValueError("chunk_ids must be a non-empty sequence of strings")
    normalized = list(chunk_ids)
    if not normalized:
        raise ValueError("chunk_ids must be a non-empty sequence of strings")
    if any(
        not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in normalized
    ):
        raise ValueError("chunk_ids must be a non-empty sequence of strings")
    return [chunk_id.strip() for chunk_id in normalized]


def _snapshot(record: DocRecord) -> DocRecord:
    return DocRecord(
        doc_id=record.doc_id,
        source=record.source,
        tenant_id=record.tenant_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        chunk_count=record.chunk_count,
        chunk_ids=list(record.chunk_ids),
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
        chunk_ids: list[str],
    ) -> DocRecord:
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
        chunk_ids: list[str],
        source: str | None = None,
    ) -> DocRecord:
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
            record.chunk_ids = normalized_chunk_ids
            record.chunk_count = len(normalized_chunk_ids)
            record.updated_at = time.time()
            return _snapshot(record)

    def delete(self, doc_id: str) -> DocRecord | None:
        doc_id = _require_non_empty_string(doc_id, "doc_id")
        with self._lock:
            record = self._docs.pop(doc_id, None)
        if record:
            logger.info("Deleted doc %s (%d chunks)", doc_id, record.chunk_count)
            return _snapshot(record)
        return None

    def get(self, doc_id: str, tenant_id: str) -> DocRecord | None:
        doc_id = _require_non_empty_string(doc_id, "doc_id")
        tenant_id = _require_non_empty_string(tenant_id, "tenant_id")
        with self._lock:
            record = self._docs.get(doc_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return _snapshot(record)

    def list_for_tenant(self, tenant_id: str) -> list[DocRecord]:
        tenant_id = _require_non_empty_string(tenant_id, "tenant_id")
        with self._lock:
            return [
                _snapshot(r) for r in self._docs.values() if r.tenant_id == tenant_id
            ]

    def exists(self, doc_id: str) -> bool:
        doc_id = _require_non_empty_string(doc_id, "doc_id")
        with self._lock:
            return doc_id in self._docs

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._docs)
