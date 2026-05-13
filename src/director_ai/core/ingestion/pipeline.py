# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Document Ingestion Pipeline
"""Reusable document ingestion facade.

This package-level surface composes the mature retrieval parser, chunker,
document registry, and vector store into a single Python API for applications
that need the same behaviour as the REST ingestion endpoints without running
the server.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from director_ai.core.retrieval.doc_chunker import ChunkConfig, split
from director_ai.core.retrieval.doc_parser import parse
from director_ai.core.retrieval.doc_registry import DocRecord, DocRegistry
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

__all__ = [
    "DeletedDocument",
    "DocumentIngestionPipeline",
    "IngestionConfig",
    "IngestionResult",
]


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for parser-to-vector-store ingestion."""

    chunk_size: int = 512
    overlap: int = 64
    semantic: bool = False
    similarity_threshold: float = 0.3

    def __post_init__(self) -> None:
        self.to_chunk_config()

    def to_chunk_config(self) -> ChunkConfig:
        return ChunkConfig(
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            semantic=self.semantic,
            similarity_threshold=self.similarity_threshold,
        )


@dataclass(frozen=True)
class IngestionResult:
    """Metadata returned after ingesting or updating one document."""

    doc_id: str
    source: str
    tenant_id: str
    chunk_ids: list[str]
    content_hash: str
    unchanged: bool = False

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ids)


@dataclass(frozen=True)
class DeletedDocument:
    """Metadata returned after deleting one ingested document."""

    doc_id: str
    tenant_id: str
    chunks_removed: int


class DocumentIngestionPipeline:
    """Parse, chunk, store, update, and delete documents for a vector store."""

    def __init__(
        self,
        *,
        store: VectorGroundTruthStore,
        registry: DocRegistry | None = None,
        config: IngestionConfig | None = None,
    ) -> None:
        self.store = store
        self.registry = registry or DocRegistry()
        self.config = config or IngestionConfig()

    def ingest_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        doc_id: str | None = None,
        source: str | None = None,
        tenant_id: str = "",
        config: IngestionConfig | None = None,
    ) -> IngestionResult:
        """Parse bytes by filename, then ingest the resulting text."""
        text = parse(content, filename)
        if not text.strip():
            raise ValueError("parsed document contains no text")
        return self.ingest_text(
            text,
            doc_id=doc_id,
            source=source or filename,
            tenant_id=tenant_id,
            config=config,
        )

    def ingest_text(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        source: str = "text",
        tenant_id: str = "",
        config: IngestionConfig | None = None,
    ) -> IngestionResult:
        """Chunk and store a new text document."""
        clean_doc_id = _normalise_doc_id(doc_id or uuid.uuid4().hex)
        clean_source = _require_non_empty(source, "source")
        clean_tenant = _normalise_tenant(tenant_id)
        if self.registry.exists(clean_doc_id):
            raise ValueError(f"Document {clean_doc_id!r} already exists")

        content_hash = _content_hash(text)
        chunk_ids = self._stage_chunks(
            text,
            doc_id=clean_doc_id,
            tenant_id=clean_tenant,
            source=clean_source,
            content_hash=content_hash,
            prefix=clean_doc_id,
            config=config,
        )
        record = self.registry.register(
            clean_doc_id,
            clean_source,
            clean_tenant,
            chunk_ids,
            content_hash=content_hash,
        )
        return _result_from_record(record)

    def update_text(
        self,
        text: str,
        *,
        doc_id: str,
        source: str = "text",
        tenant_id: str = "",
        config: IngestionConfig | None = None,
    ) -> IngestionResult:
        """Replace a document's chunks while preserving registry identity."""
        clean_doc_id = _normalise_doc_id(doc_id)
        clean_source = _require_non_empty(source, "source")
        clean_tenant = _normalise_tenant(tenant_id)
        record = self.registry.get(clean_doc_id, clean_tenant)
        if record is None:
            raise KeyError(f"Document {clean_doc_id!r} not found")

        incoming_hash = _content_hash(text)
        if record.content_hash == incoming_hash:
            return _result_from_record(record, unchanged=True)

        prefix = f"{clean_doc_id}:rev:{uuid.uuid4().hex[:12]}"
        new_chunk_ids = self._stage_chunks(
            text,
            doc_id=clean_doc_id,
            tenant_id=clean_tenant,
            source=clean_source,
            content_hash=incoming_hash,
            prefix=prefix,
            config=config,
        )
        try:
            self._delete_chunks(record)
        except Exception:
            self._cleanup_chunks(new_chunk_ids)
            raise

        updated = self.registry.update(
            clean_doc_id,
            new_chunk_ids,
            source=clean_source,
            content_hash=incoming_hash,
        )
        return _result_from_record(updated)

    def delete(self, doc_id: str, *, tenant_id: str = "") -> DeletedDocument:
        """Delete a registered document and all stored chunks."""
        clean_doc_id = _normalise_doc_id(doc_id)
        clean_tenant = _normalise_tenant(tenant_id)
        record = self.registry.get(clean_doc_id, clean_tenant)
        if record is None:
            raise KeyError(f"Document {clean_doc_id!r} not found")
        removed = self._delete_chunks(record)
        self.registry.delete(clean_doc_id)
        return DeletedDocument(
            doc_id=clean_doc_id,
            tenant_id=clean_tenant,
            chunks_removed=removed,
        )

    def _stage_chunks(
        self,
        text: str,
        *,
        doc_id: str,
        tenant_id: str,
        source: str,
        content_hash: str,
        prefix: str,
        config: IngestionConfig | None,
    ) -> list[str]:
        chunks = _chunk_text(text, config or self.config)
        chunk_ids = [f"{prefix}:chunk:{index}" for index in range(len(chunks))]
        added: list[str] = []
        for index, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks, strict=True)):
            try:
                metadata = {
                    "doc_id": doc_id,
                    "source": source,
                    "tenant_id": tenant_id,
                    "content_hash": content_hash,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                }
                self.store.backend.add(
                    doc_id=chunk_id,
                    text=chunk,
                    metadata=metadata,
                )
                self.store.facts[chunk_id] = chunk
                added.append(chunk_id)
            except Exception:
                self._cleanup_chunks(added)
                raise
        return chunk_ids

    def _delete_chunks(self, record: DocRecord) -> int:
        removed = 0
        for chunk_id in record.chunk_ids:
            delete_count = self.store.backend.delete([chunk_id])
            if isinstance(delete_count, int):
                if delete_count != 1:
                    raise RuntimeError(
                        f"Backend reported {delete_count} deletions for {chunk_id!r}"
                    )
                removed += delete_count
            else:
                removed += 1
            self.store.facts.pop(chunk_id, None)
        return removed

    def _cleanup_chunks(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            try:
                self.store.backend.delete([chunk_id])
            finally:
                self.store.facts.pop(chunk_id, None)


def _chunk_text(text: str, config: IngestionConfig) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    chunks = split(text, config.to_chunk_config())
    if not chunks:
        raise ValueError("document produced no chunks")
    return chunks


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _normalise_doc_id(doc_id: str) -> str:
    clean = _require_non_empty(doc_id, "doc_id")
    if any(part in clean for part in ("/", "\\", "..")):
        raise ValueError("doc_id must not contain path separators")
    return clean


def _normalise_tenant(tenant_id: str) -> str:
    if not isinstance(tenant_id, str):
        raise ValueError("tenant_id must be a string")
    return tenant_id.strip() or "default"


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field_name} contains control characters")
    return value.strip()


def _result_from_record(
    record: DocRecord, *, unchanged: bool = False
) -> IngestionResult:
    return IngestionResult(
        doc_id=record.doc_id,
        source=record.source,
        tenant_id=record.tenant_id,
        chunk_ids=list(record.chunk_ids),
        content_hash=record.content_hash,
        unchanged=unchanged,
    )
