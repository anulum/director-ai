# SPDX-License-Identifier: AGPL-3.0-or-later
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
import logging
import uuid

logger = logging.getLogger("DirectorAI.KnowledgeAPI")

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
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

try:
    from fastapi import APIRouter, HTTPException, Request, UploadFile
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

if _FASTAPI_AVAILABLE:

    class IngestRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=10_000_000)
        source: str = Field("text", max_length=500)
        doc_id: str | None = None
        chunk_size: int = Field(512, ge=64, le=4096)
        overlap: int = Field(64, ge=0, le=512)


def _get_tenant(request: Request) -> str:
    return getattr(request.state, "tenant_id", "") or request.headers.get(
        "X-Tenant-ID", ""
    )


def _get_registry(request: Request):

    reg = request.app.state._state.get("doc_registry")
    if reg is None:
        raise HTTPException(503, "Document registry not initialised")
    return reg


def _get_store(request: Request):
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


def _chunk_and_store(
    text: str, doc_id: str, tenant_id: str, store, chunk_size: int, overlap: int
) -> list[str]:
    from .core.retrieval.doc_chunker import ChunkConfig, split

    chunks = split(text, ChunkConfig(chunk_size=chunk_size, overlap=overlap))
    chunk_ids = [f"{doc_id}:chunk:{i}" for i in range(len(chunks))]

    for cid, chunk_text in zip(chunk_ids, chunks, strict=True):
        store.backend.add(
            doc_id=cid,
            text=chunk_text,
            metadata={"doc_id": doc_id, "tenant_id": tenant_id},
        )

    return chunk_ids


def _delete_chunks(record, store) -> int:
    removed = 0
    for cid in record.chunk_ids:
        try:
            store.backend.delete([cid])
            removed += 1
        except (AttributeError, TypeError):
            pass
        store.facts.pop(cid, None)
    return removed


def create_knowledge_router() -> APIRouter:
    """Build the /v1/knowledge router."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required for knowledge API")

    router = APIRouter(tags=["knowledge"])

    @router.post("/upload", status_code=201)
    async def upload_document(request: Request, file: UploadFile):
        """Upload a file, parse, chunk, embed, store."""
        # Early size check before reading body into memory
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
            )

        tenant_id = _get_tenant(request)
        registry = _get_registry(request)
        store = _get_store(request)

        filename = file.filename or "unknown.txt"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                415,
                f"File type ‘.{ext}’ not supported. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
            )

        content = await file.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
            )

        from .core.retrieval.doc_parser import parse

        try:
            text = parse(content, filename)
        except (ImportError, ValueError) as e:
            raise HTTPException(422, str(e)) from e

        if not text.strip():
            raise HTTPException(422, "Parsed file contains no text")

        doc_id = uuid.uuid4().hex
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
        )
        record = registry.register(
            doc_id, file.filename or "upload", tenant_id, chunk_ids
        )

        return {
            "doc_id": record.doc_id,
            "source": record.source,
            "chunk_count": record.chunk_count,
            "tenant_id": tenant_id,
        }

    @router.post("/ingest", status_code=201)
    async def ingest_text(request: Request, body: IngestRequest):
        """Ingest raw text â†’ chunk â†’ embed â†’ store."""
        tenant_id = _get_tenant(request)
        registry = _get_registry(request)
        store = _get_store(request)

        doc_id = body.doc_id or uuid.uuid4().hex

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
        )
        record = registry.register(doc_id, body.source, tenant_id, chunk_ids)

        return {
            "doc_id": record.doc_id,
            "source": record.source,
            "chunk_count": record.chunk_count,
            "tenant_id": tenant_id,
        }

    @router.get("/documents")
    async def list_documents(request: Request):
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
                }
                for d in docs
            ],
        }

    @router.get("/documents/{doc_id}")
    async def get_document(request: Request, doc_id: str):
        """Get metadata for a specific document."""
        tenant_id = _get_tenant(request)
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
        }

    @router.delete("/documents/{doc_id}")
    async def delete_document(request: Request, doc_id: str):
        """Delete a document and all its chunks."""
        tenant_id = _get_tenant(request)
        registry = _get_registry(request)
        store = _get_store(request)

        record = registry.get(doc_id, tenant_id)
        if record is None:
            raise HTTPException(404, "Document not found")

        removed = _delete_chunks(record, store)
        registry.delete(doc_id)

        return {"deleted": doc_id, "chunks_removed": removed}

    @router.put("/documents/{doc_id}")
    async def update_document(request: Request, doc_id: str, body: IngestRequest):
        """Replace a document's content — re-chunks and re-embeds."""
        tenant_id = _get_tenant(request)
        registry = _get_registry(request)
        store = _get_store(request)

        record = registry.get(doc_id, tenant_id)
        if record is None:
            raise HTTPException(404, "Document not found")

        loop = asyncio.get_running_loop()
        new_chunk_ids = await loop.run_in_executor(
            None,
            _chunk_and_store,
            body.text,
            doc_id,
            tenant_id,
            store,
            body.chunk_size,
            body.overlap,
        )

        _delete_chunks(record, store)
        registry.update(doc_id, new_chunk_ids)

        return {
            "doc_id": doc_id,
            "source": body.source,
            "chunk_count": len(new_chunk_ids),
        }

    @router.get("/search")
    async def search_knowledge(request: Request, query: str, top_k: int = 5):
        """Test retrieval quality — returns matching chunks."""
        tenant_id = _get_tenant(request)
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
    async def tune_embeddings_endpoint(request: Request):
        """Fine-tune embedding model on ingested documents.

        Builds contrastive pairs from adjacent chunks within documents,
        trains for a few epochs, and saves the tuned model. After tuning,
        re-ingest documents to use the improved embeddings.
        """
        tenant_id = _get_tenant(request)
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
