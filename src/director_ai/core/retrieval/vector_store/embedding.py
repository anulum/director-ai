# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Embedding-based vector backends (SentenceTransformer, Chroma)

"""Embedding-based vector backends.

``SentenceTransformerBackend`` uses ``sentence-transformers``
directly for in-process dense retrieval;
``ChromaBackend`` delegates storage and embedding to ChromaDB.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from .base import VectorBackend, logger

__all__ = ["SentenceTransformerBackend", "ChromaBackend"]


class _ChromaEmbeddingFunction(Protocol):
    """Callable embedding provider accepted by Chroma collections."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a Chroma document batch into numeric vectors."""
        ...


class _DirectorChromaEmbeddingAdapter:
    """Adapt a local callable to Chroma's embedding-function interface."""

    def __init__(self, embedding_function: _ChromaEmbeddingFunction) -> None:
        self._embedding_function = embedding_function

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Return embeddings from the wrapped local callable."""
        return self._embedding_function(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Return query embeddings from the wrapped local callable."""
        return self(input)

    @staticmethod
    def name() -> str:
        """Return the stable Chroma provider name for local callables."""
        return "director-ai-local-embedding"

    def default_space(self) -> str:
        """Return Chroma's default distance space for local embeddings."""
        return "l2"

    def supported_spaces(self) -> list[str]:
        """Return distance spaces accepted for local embeddings."""
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, str]:
        """Return serialisable Chroma configuration metadata."""
        return {"provider": self.name()}

    @staticmethod
    def build_from_config(
        config: dict[str, str],
    ) -> _DirectorChromaEmbeddingAdapter:
        """Reject deserialisation because local callables are process-owned."""
        raise ValueError(
            "director-ai local Chroma embedding functions must be provided at runtime",
        )

    def is_legacy(self) -> bool:
        """Return False because the adapter implements Chroma metadata hooks."""
        return False


def _normalise_chroma_embedding_function(
    embedding_function: _ChromaEmbeddingFunction,
) -> Any:
    """Return a Chroma-compatible embedding provider."""
    required_methods = ("name", "get_config", "build_from_config", "is_legacy")
    if all(
        callable(getattr(embedding_function, method, None))
        for method in required_methods
    ):
        return embedding_function
    return _DirectorChromaEmbeddingAdapter(embedding_function)


class SentenceTransformerBackend(VectorBackend):
    """Embedding-based backend using sentence-transformers directly.

    Recommended model: BAAI/bge-large-en-v1.5 (best quality/speed tradeoff).
    Alternative: Snowflake/snowflake-arctic-embed-l for multilingual.

    Requires ``pip install sentence-transformers``.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        """Load the embedding model and initialise the in-memory index."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerBackend requires sentence-transformers. "
                "Install with: pip install sentence-transformers",
            ) from e
        self._model = SentenceTransformer(model_name)
        self._docs: list[dict[str, Any]] = []
        self._embeddings: list[Any] = []
        self._lock = threading.Lock()

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Embed and append one document with optional metadata."""
        import numpy as _np

        emb = self._model.encode(text, normalize_embeddings=True)
        with self._lock:
            self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})
            self._embeddings.append(_np.asarray(emb, dtype=_np.float32))

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Return positive-similarity nearest documents, optionally tenant-scoped."""
        import numpy as _np

        with self._lock:
            if not self._docs:
                return []
            docs_snapshot = list(self._docs)
            embs_snapshot = list(self._embeddings)

        docs_and_embs = [
            (d, e)
            for d, e in zip(docs_snapshot, embs_snapshot, strict=True)
            if not tenant_id or d["metadata"].get("tenant_id") == tenant_id
        ]
        if not docs_and_embs:
            return []

        filtered_docs, filtered_embs = zip(*docs_and_embs, strict=True)

        raw_q_emb = self._model.encode(text, normalize_embeddings=True)
        q_emb = _np.asarray(raw_q_emb, dtype=_np.float32)
        similarities = [float(_np.dot(q_emb, e)) for e in filtered_embs]
        indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )
        results = []
        for idx in indices[:n_results]:
            if similarities[idx] > 0:
                results.append(
                    {
                        **filtered_docs[idx],
                        "distance": 1.0 - similarities[idx],
                    },
                )
        return results

    def count(self) -> int:
        """Return the number of retained in-memory documents."""
        with self._lock:
            return len(self._docs)

    def delete(self, doc_ids: list[str]) -> int:
        """Delete matching document IDs while keeping embeddings aligned."""
        if not isinstance(doc_ids, list):
            raise ValueError("doc_ids must be a list of non-empty strings")
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in doc_ids):
            raise ValueError("doc_ids must contain only non-empty strings")
        targets = set(doc_ids)
        if not targets:
            return 0
        with self._lock:
            kept_docs: list[dict[str, Any]] = []
            kept_embeddings: list[Any] = []
            removed = 0
            for doc, embedding in zip(self._docs, self._embeddings, strict=True):
                if doc["id"] in targets:
                    removed += 1
                    continue
                kept_docs.append(doc)
                kept_embeddings.append(embedding)
            self._docs = kept_docs
            self._embeddings = kept_embeddings
            return removed


class ChromaBackend(VectorBackend):
    """ChromaDB backend for production vector search.

    Director uses Chroma in embedded mode only. The backend constructs either
    ``chromadb.PersistentClient(path=...)`` for local persistence or
    ``chromadb.Client()`` for ephemeral in-process storage; it never constructs
    an HTTP client or forwards remote-code execution flags. That boundary keeps
    server-only Chroma advisories outside Director's execution path while the
    optional dependency remains open to operator-managed upgrades.

    Operators may inject a local Chroma embedding function for offline,
    deterministic, or pre-warmed deployments. That path avoids the Chroma
    default embedder and sentence-transformer model downloads while preserving
    real Chroma storage and query semantics.

    Requires ``pip install chromadb sentence-transformers``.
    """

    def __init__(
        self,
        collection_name: str = "director_ai_facts",
        persist_directory: str | None = None,
        embedding_model: str | None = None,
        embedding_function: _ChromaEmbeddingFunction | None = None,
    ) -> None:
        """Create or open a Chroma collection with optional persistence."""
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "ChromaDB backend requires chromadb. "
                "Install with: pip install director-ai[vector]",
            ) from e

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        create_kwargs: dict[str, Any] = {"name": collection_name}
        if embedding_model and embedding_function is not None:
            raise ValueError(
                "embedding_model and embedding_function are mutually exclusive",
            )
        if embedding_function is not None:
            create_kwargs["embedding_function"] = _normalise_chroma_embedding_function(
                embedding_function,
            )
        elif embedding_model:
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )

                create_kwargs["embedding_function"] = (
                    SentenceTransformerEmbeddingFunction(model_name=embedding_model)
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, using Chroma default embedder",
                )

        self._collection = self._client.get_or_create_collection(**create_kwargs)

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add one document to the Chroma collection."""
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata] if metadata else None,
        )

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Query Chroma and normalise result rows to VectorBackend shape."""
        where: dict[str, Any] | None = {"tenant_id": tenant_id} if tenant_id else None
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_texts=[text],
            n_results=min(n_results, count),
            where=where,
        )
        docs: list[dict[str, Any]] = []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        ids = results.get("ids")
        distances = results.get("distances")
        if results and documents:  # pragma: no branch
            for i, doc_text in enumerate(documents[0]):
                meta = metadatas[0][i] if metadatas else {}
                doc_id = ids[0][i] if ids else f"doc_{i}"
                dist = distances[0][i] if distances else 0.0
                docs.append(
                    {
                        "id": doc_id,
                        "text": doc_text,
                        "metadata": meta,
                        "distance": dist,
                    },
                )
        return docs

    def count(self) -> int:
        """Return Chroma's current collection count."""
        return int(self._collection.count())

    def delete(self, doc_ids: list[str]) -> int:
        """Delete document IDs from Chroma and return the removed count."""
        if not isinstance(doc_ids, list):
            raise ValueError("doc_ids must be a list of non-empty strings")
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in doc_ids):
            raise ValueError("doc_ids must contain only non-empty strings")
        if not doc_ids:
            return 0
        before = self.count()
        self._collection.delete(ids=doc_ids)
        after = self.count()
        return max(before - after, 0)
