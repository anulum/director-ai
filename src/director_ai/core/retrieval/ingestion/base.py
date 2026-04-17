# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ingestion plugin protocol

"""Shared types for ingestion plugins.

An :class:`IngestedDocument` is the smallest record a plugin yields.
A concrete plugin implements :class:`IngestionPlugin.iter_documents`;
the base class's :meth:`IngestionPlugin.ingest` then walks every
document and calls ``store.add``. Operators who need custom
post-processing (deduplication, chunking, language detection)
subclass and override ``_prepare_document``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..knowledge import GroundTruthStore

logger = logging.getLogger("DirectorAI.Ingestion")


@dataclass(frozen=True)
class IngestedDocument:
    """One document pulled from an external source.

    ``source`` identifies the upstream system (``"s3"``, ``"notion"``,
    ``"gdrive"``). ``source_id`` is the stable identifier within that
    system (S3 key, Notion page ID, Drive file ID). ``metadata``
    carries free-form fields the plugin decides to surface —
    last-modified time, author, tenant. Plugins should not embed
    credentials in metadata.
    """

    key: str
    text: str
    source: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class IngestionPlugin(ABC):
    """Protocol + base class for auto-ingestion adapters.

    Subclasses implement :meth:`iter_documents`. The base class's
    :meth:`ingest` iterates every document, applies optional
    :meth:`_prepare_document` post-processing, and forwards to the
    caller-supplied ``GroundTruthStore``. Returns the count of
    documents actually written.
    """

    name: str = "ingestion"

    @abstractmethod
    def iter_documents(self) -> Iterator[IngestedDocument]:
        """Yield one :class:`IngestedDocument` at a time.

        Plugins should stream — large buckets must not buffer every
        object in memory. Raise :class:`RuntimeError` for transport
        failures; :class:`PermissionError` for auth issues.
        """
        ...  # pragma: no cover

    def _prepare_document(self, doc: IngestedDocument) -> IngestedDocument | None:
        """Post-process one document before it lands in the store.

        Return ``None`` to skip (e.g. empty body, unsupported format).
        Default is a pass-through that drops documents whose text is
        empty or only whitespace — retrieval on empty bodies is a
        waste of cache space.
        """
        if not doc.text or not doc.text.strip():
            return None
        return doc

    def ingest(
        self,
        store: GroundTruthStore,
        tenant_id: str = "",
        *,
        max_documents: int | None = None,
    ) -> int:
        """Walk :meth:`iter_documents`, post-process, and store each
        surviving record. Returns the number of documents written.

        ``max_documents`` caps the run — useful for smoke tests and
        for bounded budget ingestion against very large sources.
        """
        written = 0
        for doc in self.iter_documents():
            prepared = self._prepare_document(doc)
            if prepared is None:
                continue
            metadata = {
                "source": prepared.source,
                "source_id": prepared.source_id,
                **prepared.metadata,
            }
            if tenant_id:
                metadata["tenant_id"] = tenant_id
            _store_add(store, prepared.key, prepared.text, metadata, tenant_id)
            written += 1
            if max_documents is not None and written >= max_documents:
                break
        logger.info(
            "%s: wrote %d document(s) (tenant=%s)", self.name, written, tenant_id or "-"
        )
        return written


def _store_add(
    store: GroundTruthStore,
    key: str,
    value: str,
    metadata: dict[str, Any],
    tenant_id: str,
) -> None:
    """Call ``store.add`` with whichever signature the store supports.

    ``GroundTruthStore`` accepts ``(key, value, tenant_id)`` and
    ``VectorGroundTruthStore`` accepts ``(key, value, metadata,
    tenant_id)``. The adapter falls back gracefully so either works.
    """
    try:
        store.add(key=key, value=value, metadata=metadata, tenant_id=tenant_id)
    except TypeError:
        store.add(key, value, tenant_id=tenant_id)


def chunks(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Simple chunker used by several plugins — batches page-paginated
    APIs into ``size``-sized lists for bulk processing. Public so
    caller-supplied subclasses can reuse it without re-rolling."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
