# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
        now = time.time()
        record = DocRecord(
            doc_id=doc_id,
            source=source,
            tenant_id=tenant_id,
            created_at=now,
            updated_at=now,
            chunk_count=len(chunk_ids),
            chunk_ids=list(chunk_ids),
        )
        with self._lock:
            if doc_id in self._docs:
                raise ValueError(f"Document {doc_id!r} already registered")
            self._docs[doc_id] = record
        logger.info(
            "Registered doc %s (%d chunks, tenant=%s)",
            doc_id,
            len(chunk_ids),
            tenant_id,
        )
        return record

    def update(self, doc_id: str, chunk_ids: list[str]) -> DocRecord:
        with self._lock:
            record = self._docs.get(doc_id)
            if record is None:
                raise KeyError(f"Document {doc_id!r} not found")
            record.chunk_ids = list(chunk_ids)
            record.chunk_count = len(chunk_ids)
            record.updated_at = time.time()
        return record

    def delete(self, doc_id: str) -> DocRecord | None:
        with self._lock:
            record = self._docs.pop(doc_id, None)
        if record:
            logger.info("Deleted doc %s (%d chunks)", doc_id, record.chunk_count)
        return record

    def get(self, doc_id: str, tenant_id: str) -> DocRecord | None:
        with self._lock:
            record = self._docs.get(doc_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return record

    def list_for_tenant(self, tenant_id: str) -> list[DocRecord]:
        with self._lock:
            return [r for r in self._docs.values() if r.tenant_id == tenant_id]

    def exists(self, doc_id: str) -> bool:
        with self._lock:
            return doc_id in self._docs

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._docs)
