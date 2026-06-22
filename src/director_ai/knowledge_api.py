# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Knowledge Ingestion API

"""REST API for document upload, chunking, and vector store management.

Endpoints under ``/v1/knowledge/`` — all tenant-scoped.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from typing import Any

from .core.kb_write_security import (
    KBWriteAccessError,
    canonical_kb_payload,
    check_kb_write_access,
    parse_hmac_keys,
    verify_kb_payload_signature,
)

logger = logging.getLogger("DirectorAI.KnowledgeAPI")

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_MAX_TOP_K = 50
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ALLOWED_EXTENSIONS = frozenset(
    {
        "pdf",
        "docx",
        "html",
        "htm",
        "csv",
        "txt",
        "md",
        "markdown",
        "json",
        "xml",
    }
)
_GENERIC_CONTENT_TYPES = frozenset({"", "application/octet-stream"})
_CONTENT_TYPES_BY_EXTENSION = {
    "pdf": frozenset({"application/pdf"}),
    "docx": frozenset(
        {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        }
    ),
    "html": frozenset({"text/html", "application/xhtml+xml"}),
    "htm": frozenset({"text/html", "application/xhtml+xml"}),
    "csv": frozenset({"text/csv", "application/csv", "application/vnd.ms-excel"}),
    "txt": frozenset({"text/plain"}),
    "md": frozenset({"text/markdown", "text/plain"}),
    "markdown": frozenset({"text/markdown", "text/plain"}),
    "json": frozenset({"application/json", "text/plain"}),
    "xml": frozenset({"application/xml", "text/xml", "text/plain"}),
}

try:
    from fastapi import APIRouter, HTTPException, Request, UploadFile
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

if _FASTAPI_AVAILABLE:

    class IngestRequest(BaseModel):
        """Request body for tenant-scoped raw-text knowledge ingestion."""

        text: str = Field(..., min_length=1, max_length=10_000_000)
        source: str = Field("text", max_length=500)
        doc_id: str | None = None
        chunk_size: int = Field(512, ge=64, le=4096)
        overlap: int = Field(64, ge=0, le=512)
        signature: str = Field("", max_length=256)
        signature_key_id: str = Field("", max_length=128)


def _get_tenant(request: Request) -> str:
    """Return the validated tenant id from request state or headers."""
    tenant_id = getattr(request.state, "tenant_id", "") or request.headers.get(
        "X-Tenant-ID", ""
    )
    return _validate_optional_identifier("tenant_id", tenant_id)


def _validate_optional_identifier(name: str, value: str | None) -> str:
    """Validate an optional tenant, document, or signature identifier."""
    if value is None or value == "":
        return ""
    if not _SAFE_ID_RE.fullmatch(value):
        raise HTTPException(
            400,
            f"{name} must be 1-128 chars: letters, numbers, dot, underscore, colon, dash",
        )
    return value


def _validate_source(source: str) -> str:
    """Validate and normalise the user-visible document source label."""
    clean = source.strip()
    if not clean:
        raise HTTPException(400, "source must be non-empty")
    if any(ch in clean for ch in ("\x00", "\r", "\n")):
        raise HTTPException(400, "source contains invalid control characters")
    return clean


def _validate_chunking(chunk_size: int, overlap: int) -> None:
    """Reject chunking settings that would make progress impossible."""
    if overlap >= chunk_size:
        raise HTTPException(400, "overlap must be smaller than chunk_size")


def _validate_content_length(raw_value: str | None) -> None:
    """Validate the upload Content-Length header before reading the body."""
    if not raw_value:
        return
    try:
        content_length = int(raw_value)
    except ValueError as exc:
        raise HTTPException(400, "invalid content-length header") from exc
    if content_length < 0:
        raise HTTPException(400, "invalid content-length header")
    if content_length > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
        )


async def _read_within_limit(file: UploadFile) -> bytes:
    """Read an upload, rejecting once it exceeds the size limit.

    The Content-Length pre-check is only advisory — a client may omit the header.
    This reads the body in fixed chunks and aborts with 413 as soon as the
    running total passes ``_MAX_UPLOAD_BYTES``, so a missing or dishonest
    Content-Length cannot force an unbounded body into memory: at most one chunk
    beyond the limit is ever buffered.
    """
    chunk_size = 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _file_extension(filename: str) -> str:
    """Return the lower-case filename extension without the leading dot."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def _validate_upload_type(filename: str, content_type: str | None) -> str:
    """Validate upload extension and MIME type consistency."""
    ext = _file_extension(filename)
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            415,
            f"File type '.{ext}' not supported. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    clean_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if clean_content_type in _GENERIC_CONTENT_TYPES:
        return ext

    allowed = _CONTENT_TYPES_BY_EXTENSION.get(ext, frozenset())
    if allowed and clean_content_type not in allowed:
        raise HTTPException(
            415,
            f"Content-Type {clean_content_type!r} does not match '.{ext}' upload",
        )
    return ext


def _get_registry(request: Request) -> Any:
    """Return the configured document registry or raise service unavailable."""
    reg = request.app.state._state.get("doc_registry")
    if reg is None:
        raise HTTPException(503, "Document registry not initialised")
    return reg


def _get_store(request: Request) -> Any:
    """Return the configured vector ground-truth store."""
    from .core.retrieval.vector_store import VectorGroundTruthStore

    scorer = request.app.state._state.get("scorer")
    if scorer is None:
        raise HTTPException(503, "Scorer not initialised")
    store = getattr(scorer, "ground_truth_store", None)
    if not isinstance(store, VectorGroundTruthStore):
        raise HTTPException(
            503,
            "Vector store not configured. Use DirectorConfig with a vector backend.",
        )
    return store


def _get_config(request: Request) -> Any:
    """Return the active Director configuration from FastAPI state."""
    return getattr(request.app.state, "config", None) or request.app.state._state.get(
        "config"
    )


def _require_write_access(request: Request, tenant_id: str) -> None:
    """Enforce configured knowledge-base write authorisation rules."""
    cfg = _get_config(request)
    require_auth = bool(getattr(cfg, "knowledge_write_require_auth", False))
    require_binding = bool(getattr(cfg, "knowledge_write_require_tenant_binding", True))
    try:
        check_kb_write_access(
            require_auth=require_auth,
            require_tenant_binding=require_binding,
            authenticated=bool(getattr(request.state, "kb_write_key_ok", False)),
            tenant_binding_enforced=bool(
                getattr(request.state, "kb_tenant_binding_ok", False)
            ),
            bound_tenant=getattr(request.state, "tenant_id", ""),
            requested_tenant=tenant_id,
        )
    except KBWriteAccessError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


def _signature_metadata(
    request: Request,
    canonical_payload: str,
    signature: str = "",
    key_id: str = "",
) -> dict[str, object]:
    """Verify optional HMAC write signatures and return audit metadata."""
    cfg = _get_config(request)
    require_signature = bool(getattr(cfg, "knowledge_write_require_signature", False))
    keys = parse_hmac_keys(str(getattr(cfg, "knowledge_write_hmac_keys", "")))

    clean_signature = signature.strip()
    clean_key_id = _validate_optional_identifier("signature_key_id", key_id.strip())
    if not clean_signature:
        if require_signature:
            raise HTTPException(403, "Knowledge-base write signature required")
        return {}
    if not verify_kb_payload_signature(
        canonical_payload,
        clean_signature,
        keys,
        clean_key_id,
    ):
        raise HTTPException(403, "Invalid knowledge-base write signature")
    return {
        "kb_signature": clean_signature,
        "kb_signature_key_id": clean_key_id,
        "kb_signature_verified": True,
    }


def _chunk_and_store(
    text: str,
    doc_id: str,
    tenant_id: str,
    store: Any,
    chunk_size: int,
    overlap: int,
    signature_metadata: dict[str, object] | None = None,
    chunk_id_prefix: str | None = None,
) -> list[str]:
    """Split text into chunks, store them, and roll back partial writes."""
    from .core.retrieval.doc_chunker import ChunkConfig, split

    chunks = split(text, ChunkConfig(chunk_size=chunk_size, overlap=overlap))
    base_prefix = chunk_id_prefix or doc_id
    # Tenant-scope the chunk id so the keyword-fallback tenant filter
    # (GroundTruthStore.retrieve_context) can find ingested chunks, and so two
    # tenants ingesting the same doc id cannot collide on a shared cid. The cid
    # flows unchanged to the vector backend, the registry, and the delete/get
    # paths, so no tenant threading is needed at those sites.
    prefix = f"{tenant_id}:{base_prefix}" if tenant_id else base_prefix
    chunk_ids = [f"{prefix}:chunk:{i}" for i in range(len(chunks))]

    added_chunk_ids: list[str] = []
    for cid, chunk_text in zip(chunk_ids, chunks, strict=True):
        try:
            store.backend.add(
                doc_id=cid,
                text=chunk_text,
                metadata={
                    "doc_id": doc_id,
                    "tenant_id": tenant_id,
                    **(signature_metadata or {}),
                },
            )
            store.facts[cid] = chunk_text
            added_chunk_ids.append(cid)
        except Exception:
            _cleanup_chunk_ids(added_chunk_ids, store)
            raise

    return chunk_ids


def _content_hash(text: str) -> str:
    """Return a stable SHA-256 hash for ingested text content."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _cleanup_chunk_ids(chunk_ids: list[str], store: Any) -> None:
    """Best-effort rollback for chunks staged before an ingestion failure."""
    for cid in chunk_ids:
        try:
            store.backend.delete([cid])
        except Exception:
            logger.warning("Unable to roll back staged knowledge chunk %s", cid)
        finally:
            store.facts.pop(cid, None)


def _delete_chunks(record: Any, store: Any) -> int:
    """Delete all vector chunks for a registry record."""
    removed = 0
    for cid in record.chunk_ids:
        try:
            delete_count = store.backend.delete([cid])
        except Exception as exc:
            raise RuntimeError(f"Unable to delete chunk {cid!r}") from exc
        if isinstance(delete_count, int):
            if delete_count != 1:
                raise RuntimeError(
                    f"Backend reported {delete_count} deletions for chunk {cid!r}",
                )
            removed += delete_count
        else:
            removed += 1
        store.facts.pop(cid, None)
    return removed


def create_knowledge_router() -> APIRouter:
    """Build the /v1/knowledge router."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required for knowledge API")

    router = APIRouter(tags=["knowledge"])

    @router.post("/upload", status_code=201)
    async def upload_document(request: Request, file: UploadFile) -> dict[str, Any]:
        """Upload a file, parse, chunk, embed, store."""
        # Early size check before reading body into memory
        _validate_content_length(request.headers.get("content-length"))

        tenant_id = _get_tenant(request)
        _require_write_access(request, tenant_id)
        registry = _get_registry(request)
        store = _get_store(request)

        filename = file.filename or "unknown.txt"
        _validate_upload_type(filename, file.content_type)

        content = await _read_within_limit(file)

        from .core.retrieval.doc_parser import parse

        try:
            text = parse(content, filename)
        except (ImportError, ValueError) as e:
            raise HTTPException(422, str(e)) from e

        if not text.strip():
            raise HTTPException(422, "Parsed file contains no text")

        doc_id = uuid.uuid4().hex
        sig_meta = _signature_metadata(
            request,
            canonical_kb_payload(
                kind="upload",
                tenant_id=tenant_id,
                doc_id=doc_id,
                source=file.filename or "upload",
                content=content,
            ),
            request.headers.get("X-KB-Signature", ""),
            request.headers.get("X-KB-Key-ID", ""),
        )
        _validate_chunking(512, 64)
        loop = asyncio.get_running_loop()
        chunk_ids = await loop.run_in_executor(
            None,
            _chunk_and_store,
            text,
            doc_id,
            tenant_id,
            store,
            512,
            64,
            sig_meta,
        )
        record = registry.register(
            doc_id,
            file.filename or "upload",
            tenant_id,
            chunk_ids,
            content_hash=_content_hash(text),
        )

        return {
            "doc_id": record.doc_id,
            "source": record.source,
            "chunk_count": record.chunk_count,
            "tenant_id": tenant_id,
        }

    @router.post("/ingest", status_code=201)
    async def ingest_text(request: Request, body: IngestRequest) -> dict[str, Any]:
        """Ingest raw text â†’ chunk â†’ embed â†’ store."""
        tenant_id = _get_tenant(request)
        _require_write_access(request, tenant_id)
        registry = _get_registry(request)
        store = _get_store(request)

        doc_id = (
            _validate_optional_identifier("doc_id", body.doc_id) or uuid.uuid4().hex
        )
        source = _validate_source(body.source)
        _validate_chunking(body.chunk_size, body.overlap)
        sig_meta = _signature_metadata(
            request,
            canonical_kb_payload(
                kind="ingest",
                tenant_id=tenant_id,
                doc_id=doc_id,
                source=source,
                text=body.text,
            ),
            body.signature,
            body.signature_key_id,
        )

        if registry.exists(doc_id):
            raise HTTPException(
                409, f"Document {doc_id!r} already exists. Use PUT to update."
            )

        loop = asyncio.get_running_loop()
        chunk_ids = await loop.run_in_executor(
            None,
            _chunk_and_store,
            body.text,
            doc_id,
            tenant_id,
            store,
            body.chunk_size,
            body.overlap,
            sig_meta,
        )
        record = registry.register(
            doc_id,
            source,
            tenant_id,
            chunk_ids,
            content_hash=_content_hash(body.text),
        )

        return {
            "doc_id": record.doc_id,
            "source": record.source,
            "chunk_count": record.chunk_count,
            "tenant_id": tenant_id,
        }

    @router.get("/documents")
    async def list_documents(request: Request) -> dict[str, Any]:
        """List all documents for the current tenant."""
        tenant_id = _get_tenant(request)
        registry = _get_registry(request)
        docs = registry.list_for_tenant(tenant_id)
        return {
            "tenant_id": tenant_id,
            "count": len(docs),
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "source": d.source,
                    "chunk_count": d.chunk_count,
                    "created_at": d.created_at,
                    "updated_at": d.updated_at,
                    "content_hash": getattr(d, "content_hash", ""),
                }
                for d in docs
            ],
        }

    @router.get("/documents/{doc_id}")
    async def get_document(request: Request, doc_id: str) -> dict[str, Any]:
        """Get metadata for a specific document."""
        tenant_id = _get_tenant(request)
        doc_id = _validate_optional_identifier("doc_id", doc_id)
        registry = _get_registry(request)
        record = registry.get(doc_id, tenant_id)
        if record is None:
            raise HTTPException(404, "Document not found")
        return {
            "doc_id": record.doc_id,
            "source": record.source,
            "tenant_id": record.tenant_id,
            "chunk_count": record.chunk_count,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "content_hash": getattr(record, "content_hash", ""),
        }

    @router.delete("/documents/{doc_id}")
    async def delete_document(request: Request, doc_id: str) -> dict[str, Any]:
        """Delete a document and all its chunks."""
        tenant_id = _get_tenant(request)
        _require_write_access(request, tenant_id)
        doc_id = _validate_optional_identifier("doc_id", doc_id)
        registry = _get_registry(request)
        store = _get_store(request)

        record = registry.get(doc_id, tenant_id)
        if record is None:
            raise HTTPException(404, "Document not found")

        try:
            removed = _delete_chunks(record, store)
        except RuntimeError as exc:
            logger.exception("Failed deleting knowledge document chunks: %s", doc_id)
            raise HTTPException(503, "Unable to delete document chunks") from exc
        registry.delete(doc_id)

        return {"deleted": doc_id, "chunks_removed": removed}

    @router.put("/documents/{doc_id}")
    async def update_document(
        request: Request, doc_id: str, body: IngestRequest
    ) -> dict[str, Any]:
        """Replace a document's content — re-chunks and re-embeds."""
        tenant_id = _get_tenant(request)
        _require_write_access(request, tenant_id)
        doc_id = _validate_optional_identifier("doc_id", doc_id)
        registry = _get_registry(request)
        store = _get_store(request)
        source = _validate_source(body.source)
        _validate_chunking(body.chunk_size, body.overlap)
        sig_meta = _signature_metadata(
            request,
            canonical_kb_payload(
                kind="update",
                tenant_id=tenant_id,
                doc_id=doc_id,
                source=source,
                text=body.text,
            ),
            body.signature,
            body.signature_key_id,
        )

        record = registry.get(doc_id, tenant_id)
        if record is None:
            raise HTTPException(404, "Document not found")
        incoming_hash = _content_hash(body.text)
        if getattr(record, "content_hash", "") == incoming_hash:
            return {
                "doc_id": doc_id,
                "source": record.source,
                "chunk_count": record.chunk_count,
                "content_hash": incoming_hash,
                "unchanged": True,
            }

        loop = asyncio.get_running_loop()
        chunk_id_prefix = f"{doc_id}:rev:{uuid.uuid4().hex[:12]}"
        try:
            new_chunk_ids = await loop.run_in_executor(
                None,
                _chunk_and_store,
                body.text,
                doc_id,
                tenant_id,
                store,
                body.chunk_size,
                body.overlap,
                sig_meta,
                chunk_id_prefix,
            )
        except Exception as exc:
            logger.exception("Failed staging replacement knowledge chunks: %s", doc_id)
            raise HTTPException(503, "Unable to stage replacement chunks") from exc

        try:
            _delete_chunks(record, store)
        except RuntimeError as exc:
            _cleanup_chunk_ids(new_chunk_ids, store)
            logger.exception("Failed replacing knowledge document chunks: %s", doc_id)
            raise HTTPException(503, "Unable to replace document chunks") from exc

        record = registry.update(
            doc_id,
            new_chunk_ids,
            source=source,
            content_hash=incoming_hash,
        )

        return {
            "doc_id": doc_id,
            "source": record.source,
            "chunk_count": record.chunk_count,
            "content_hash": getattr(record, "content_hash", incoming_hash),
            "unchanged": False,
        }

    @router.get("/search")
    async def search_knowledge(
        request: Request, query: str, top_k: int = 5
    ) -> dict[str, Any]:
        """Test retrieval quality — returns matching chunks."""
        tenant_id = _get_tenant(request)
        if not query.strip():
            raise HTTPException(400, "query must be non-empty")
        if top_k < 1 or top_k > _MAX_TOP_K:
            raise HTTPException(400, f"top_k must be between 1 and {_MAX_TOP_K}")
        store = _get_store(request)

        try:
            results = store.backend.query(query, n_results=top_k, tenant_id=tenant_id)
        except TypeError:
            results = store.backend.query(query, n_results=top_k)

        return {
            "query": query,
            "tenant_id": tenant_id,
            "results": [
                {
                    "text": r.get("text", "")[:500],
                    "distance": r.get("distance"),
                    "metadata": r.get("metadata", {}),
                }
                for r in results
            ],
        }

    @router.post("/tune-embeddings")
    async def tune_embeddings_endpoint(request: Request) -> dict[str, Any]:
        """Fine-tune embedding model on ingested documents.

        Builds contrastive pairs from adjacent chunks within documents,
        trains for a few epochs, and saves the tuned model. After tuning,
        re-ingest documents to use the improved embeddings.
        """
        tenant_id = _get_tenant(request)
        _require_write_access(request, tenant_id)
        registry = _get_registry(request)
        docs = registry.list_for_tenant(tenant_id)

        if len(docs) < 2:
            raise HTTPException(
                422,
                f"Need at least 2 documents for tuning, got {len(docs)}",
            )

        store = _get_store(request)
        documents: list[list[str]] = []
        for doc in docs:
            chunks = []
            for cid in doc.chunk_ids:
                text = store.facts.get(cid, "")
                if text:
                    chunks.append(text)
            if len(chunks) >= 2:
                documents.append(chunks)

        if len(documents) < 2:
            raise HTTPException(
                422,
                "Need at least 2 documents with 2+ chunks each",
            )

        from .core.retrieval.embedding_tuner import tune_embeddings

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            tune_embeddings,
            documents,
        )

        return {
            "model_path": result.model_path,
            "train_samples": result.train_samples,
            "epochs": result.epochs,
            "message": "Re-ingest documents to use tuned embeddings",
        }

    return router
