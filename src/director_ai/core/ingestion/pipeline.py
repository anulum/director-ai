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
from typing import TYPE_CHECKING

from director_ai.core.retrieval.doc_chunker import ChunkConfig, split
from director_ai.core.retrieval.doc_parser import parse
from director_ai.core.retrieval.doc_registry import DocRecord, DocRegistry
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

if TYPE_CHECKING:
    from director_ai.core.provenance import KnowledgeProvenanceLedger
    from director_ai.core.provenance.supersession import SupersessionDecision

__all__ = [
    "DeletedDocument",
    "DocumentIngestionPipeline",
    "IngestionConfig",
    "IngestionResult",
    "SupersessionResult",
]


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for parser-to-vector-store ingestion."""

    chunk_size: int = 512
    overlap: int = 64
    semantic: bool = False
    similarity_threshold: float = 0.3

    def __post_init__(self) -> None:
        """Validate chunking parameters through ChunkConfig construction."""
        self.to_chunk_config()

    def to_chunk_config(self) -> ChunkConfig:
        """Return the retrieval chunker configuration."""
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
        """Return the number of chunks stored for this document."""
        return len(self.chunk_ids)


@dataclass(frozen=True)
class DeletedDocument:
    """Metadata returned after deleting one ingested document."""

    doc_id: str
    tenant_id: str
    chunks_removed: int


@dataclass(frozen=True)
class SupersessionResult:
    """Metadata returned after applying a supersession decision."""

    incoming_doc_id: str
    superseded_doc_ids: tuple[str, ...]
    chunks_removed: int

    @property
    def superseded_count(self) -> int:
        """Return the number of documents retired by the supersession."""
        return len(self.superseded_doc_ids)


class DocumentIngestionPipeline:
    """Parse, chunk, store, update, and delete documents for a vector store."""

    def __init__(
        self,
        *,
        store: VectorGroundTruthStore,
        registry: DocRegistry | None = None,
        config: IngestionConfig | None = None,
        ledger: KnowledgeProvenanceLedger | None = None,
    ) -> None:
        self.store = store
        self.registry = registry or DocRegistry()
        self.config = config or IngestionConfig()
        # Optional operational provenance. When supplied, every mutation is
        # appended to the tamper-evident ledger; when omitted, ingestion
        # behaves exactly as before.
        self.ledger = ledger

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
        chunk_ids, leaves = self._stage_chunks(
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
        self._record_mutation(
            "ingest",
            record=record,
            leaves=leaves,
            removed_chunk_ids=(),
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
        removed_chunk_ids = tuple(record.chunk_ids)
        new_chunk_ids, leaves = self._stage_chunks(
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
        self._record_mutation(
            "update",
            record=updated,
            leaves=leaves,
            removed_chunk_ids=removed_chunk_ids,
        )
        return _result_from_record(updated)

    def delete(self, doc_id: str, *, tenant_id: str = "") -> DeletedDocument:
        """Delete a registered document and all stored chunks."""
        clean_doc_id = _normalise_doc_id(doc_id)
        clean_tenant = _normalise_tenant(tenant_id)
        record = self.registry.get(clean_doc_id, clean_tenant)
        if record is None:
            raise KeyError(f"Document {clean_doc_id!r} not found")
        removed_chunk_ids = tuple(record.chunk_ids)
        removed = self._delete_chunks(record)
        self.registry.delete(clean_doc_id)
        self._record_deletion(record, removed_chunk_ids=removed_chunk_ids)
        return DeletedDocument(
            doc_id=clean_doc_id,
            tenant_id=clean_tenant,
            chunks_removed=removed,
        )

    def apply_supersession(
        self,
        decision: SupersessionDecision,
        *,
        approved: bool = False,
    ) -> SupersessionResult:
        """Retire the documents named by an approved supersession decision.

        Each superseded document's chunks are removed from the store and
        registry, then a single ledger ``supersede`` event links them to the
        incoming document. A decision that still requires human approval is
        refused unless ``approved=True``; an auto-promoted decision applies
        without it. Documents already gone are skipped silently.
        """
        if decision.action == "none":
            return SupersessionResult(
                incoming_doc_id=decision.incoming_doc_id,
                superseded_doc_ids=(),
                chunks_removed=0,
            )
        if decision.requires_human_approval and not approved:
            raise PermissionError(
                "supersession decision requires human approval; "
                "pass approved=True to apply it"
            )
        removed_chunk_ids: list[str] = []
        superseded: list[str] = []
        for old_doc_id in decision.superseded_doc_ids:
            record = self.registry.get(old_doc_id, decision.tenant_id)
            if record is None:
                continue
            self._delete_chunks(record)
            self.registry.delete(old_doc_id)
            removed_chunk_ids.extend(record.chunk_ids)
            superseded.append(old_doc_id)
        if superseded and removed_chunk_ids and self.ledger is not None:
            self.ledger.record_supersede(
                doc_id=decision.incoming_doc_id,
                tenant_id=decision.tenant_id,
                source=decision.incoming_source,
                supersedes=superseded,
                removed_chunk_ids=removed_chunk_ids,
            )
        return SupersessionResult(
            incoming_doc_id=decision.incoming_doc_id,
            superseded_doc_ids=tuple(superseded),
            chunks_removed=len(removed_chunk_ids),
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
    ) -> tuple[list[str], list[bytes]]:
        """Chunk text, store every chunk, and return ids + content digests.

        The per-chunk SHA-256 digests are the leaves of the provenance
        ledger's content commitment, so they bind each stored chunk's
        exact text to the recorded mutation.
        """
        chunks = _chunk_text(text, config or self.config)
        chunk_ids = [f"{prefix}:chunk:{index}" for index in range(len(chunks))]
        leaves = [_chunk_leaf(chunk) for chunk in chunks]
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
        return chunk_ids, leaves

    def _delete_chunks(self, record: DocRecord) -> int:
        """Delete all vector-store chunks for a registered document."""
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
        """Best-effort cleanup for chunks staged before a failure."""
        for chunk_id in chunk_ids:
            try:
                self.store.backend.delete([chunk_id])
            finally:
                self.store.facts.pop(chunk_id, None)

    def _record_mutation(
        self,
        event_type: str,
        *,
        record: DocRecord,
        leaves: list[bytes],
        removed_chunk_ids: tuple[str, ...],
    ) -> None:
        """Append an ingest/update event to the ledger when one is attached."""
        if self.ledger is None:
            return
        chunk_leaves = list(zip(record.chunk_ids, leaves, strict=True))
        if event_type == "ingest":
            self.ledger.record_ingest(
                doc_id=record.doc_id,
                tenant_id=record.tenant_id,
                source=record.source,
                content_hash=record.content_hash,
                chunk_leaves=chunk_leaves,
            )
        else:
            self.ledger.record_update(
                doc_id=record.doc_id,
                tenant_id=record.tenant_id,
                source=record.source,
                content_hash=record.content_hash,
                chunk_leaves=chunk_leaves,
                removed_chunk_ids=removed_chunk_ids,
            )

    def _record_deletion(
        self, record: DocRecord, *, removed_chunk_ids: tuple[str, ...]
    ) -> None:
        """Append a delete event to the ledger when one is attached."""
        if self.ledger is None:
            return
        self.ledger.record_delete(
            doc_id=record.doc_id,
            tenant_id=record.tenant_id,
            removed_chunk_ids=removed_chunk_ids,
            source=record.source,
        )


def _chunk_text(text: str, config: IngestionConfig) -> list[str]:
    """Split non-empty text into ingestion chunks."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    chunks = split(text, config.to_chunk_config())
    if not chunks:
        raise ValueError("document produced no chunks")
    return chunks


def _content_hash(text: str) -> str:
    """Return a stable SHA-256 hash for document content."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _chunk_leaf(chunk: str) -> bytes:
    """Return the raw SHA-256 digest of one chunk's text (ledger leaf)."""
    return hashlib.sha256(chunk.encode("utf-8", errors="replace")).digest()


def _normalise_doc_id(doc_id: str) -> str:
    """Return a validated document id that cannot encode a path."""
    clean = _require_non_empty(doc_id, "doc_id")
    if any(part in clean for part in ("/", "\\", "..")):
        raise ValueError("doc_id must not contain path separators")
    return clean


def _normalise_tenant(tenant_id: str) -> str:
    """Return a stripped tenant id or the default tenant."""
    if not isinstance(tenant_id, str):
        raise ValueError("tenant_id must be a string")
    return tenant_id.strip() or "default"


def _require_non_empty(value: str, field_name: str) -> str:
    """Return stripped text after rejecting empty/control-character values."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field_name} contains control characters")
    return value.strip()


def _result_from_record(
    record: DocRecord, *, unchanged: bool = False
) -> IngestionResult:
    """Convert a registry record into an ingestion result."""
    return IngestionResult(
        doc_id=record.doc_id,
        source=record.source,
        tenant_id=record.tenant_id,
        chunk_ids=list(record.chunk_ids),
        content_hash=record.content_hash,
        unchanged=unchanged,
    )
