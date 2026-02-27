# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Vector Database Backend for GroundTruthStore
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Pluggable vector database backend for embedding-based retrieval.

Provides ``VectorGroundTruthStore`` which extends ``GroundTruthStore``
with semantic similarity search via ChromaDB (local) or any backend
implementing the ``VectorBackend`` protocol.

Install with::

    pip install director-ai[vector]
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .knowledge import GroundTruthStore
from .types import EvidenceChunk

# Re-export recommended model name for documentation
RECOMMENDED_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

logger = logging.getLogger("DirectorAI.VectorStore")


class VectorBackend(ABC):
    """Protocol for vector database backends."""

    @abstractmethod
    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None: ...

    @abstractmethod
    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]: ...

    @abstractmethod
    def count(self) -> int: ...


class InMemoryBackend(VectorBackend):
    """Simple in-memory cosine-similarity backend (no external deps).

    Uses TF-IDF-like word overlap for embedding approximation.
    Suitable for testing and small fact stores.
    """

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})

    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        if not self._docs:
            return []
        query_words = set(text.lower().split())
        scored = []
        for doc in self._docs:
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words & doc_words)
            total = max(len(query_words | doc_words), 1)
            similarity = overlap / total
            scored.append((similarity, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, doc in scored[:n_results]:
            if sim > 0:
                results.append({**doc, "distance": 1.0 - sim})
        return results

    def count(self) -> int:
        return len(self._docs)


class SentenceTransformerBackend(VectorBackend):
    """Embedding-based backend using sentence-transformers directly.

    Recommended model: BAAI/bge-large-en-v1.5 (best quality/speed tradeoff).
    Alternative: Snowflake/snowflake-arctic-embed-l for multilingual.

    Requires ``pip install sentence-transformers``.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerBackend requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._docs: list[dict[str, Any]] = []
        self._embeddings: list[Any] = []

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        import numpy as _np

        emb = self._model.encode(text, normalize_embeddings=True)
        self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})
        self._embeddings.append(_np.asarray(emb, dtype=_np.float32))

    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        import numpy as _np

        if not self._docs:
            return []
        q_emb = self._model.encode(text, normalize_embeddings=True)
        q_emb = _np.asarray(q_emb, dtype=_np.float32)
        similarities = [float(_np.dot(q_emb, e)) for e in self._embeddings]
        indices = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )
        results = []
        for idx in indices[:n_results]:
            if similarities[idx] > 0:
                results.append(
                    {
                        **self._docs[idx],
                        "distance": 1.0 - similarities[idx],
                    }
                )
        return results

    def count(self) -> int:
        return len(self._docs)


class ChromaBackend(VectorBackend):
    """ChromaDB backend for production vector search.

    Requires ``pip install chromadb sentence-transformers``.
    """

    def __init__(
        self,
        collection_name: str = "director_ai_facts",
        persist_directory: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "ChromaDB backend requires chromadb. "
                "Install with: pip install director-ai[vector]"
            ) from e

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        create_kwargs: dict[str, Any] = {"name": collection_name}
        if embedding_model:
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )

                create_kwargs["embedding_function"] = (
                    SentenceTransformerEmbeddingFunction(model_name=embedding_model)
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, using Chroma default embedder"
                )

        self._collection = self._client.get_or_create_collection(**create_kwargs)

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata] if metadata else None,
        )

    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        results = self._collection.query(query_texts=[text], n_results=n_results)
        docs: list[dict[str, Any]] = []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        ids = results.get("ids")
        distances = results.get("distances")
        if results and documents:
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
                    }
                )
        return docs

    def count(self) -> int:
        return int(self._collection.count())


class VectorGroundTruthStore(GroundTruthStore):
    """Ground truth store with vector-based semantic retrieval.

    Extends the keyword-based ``GroundTruthStore`` with embedding-based
    similarity search. Falls back to keyword matching when the vector
    backend returns no results.

    Parameters
    ----------
    backend : VectorBackend — vector DB backend (default: InMemoryBackend).
    auto_index : bool — index built-in facts on init (default: True).
    """

    def __init__(
        self,
        backend: VectorBackend | None = None,
        auto_index: bool = True,
    ) -> None:
        super().__init__()
        self.backend = backend if backend is not None else InMemoryBackend()

        if auto_index:
            self._index_builtin_facts()

    def _index_builtin_facts(self) -> None:
        """Index the built-in fact dictionary into the vector backend."""
        for key, value in self.facts.items():
            self.backend.add(
                doc_id=f"builtin_{key.replace(' ', '_')}",
                text=f"{key} is {value}",
                metadata={"source": "builtin", "key": key},
            )
        logger.info("Indexed %d built-in facts into vector backend.", len(self.facts))

    def ingest(self, texts: list[str]) -> int:
        """Bulk-add plain text documents into the vector backend."""
        for i, text in enumerate(texts):
            self.backend.add(
                doc_id=f"ingest_{i}", text=text, metadata={"source": "ingest"}
            )
        logger.info("Ingested %d documents into vector backend.", len(texts))
        return len(texts)

    def add_fact(
        self, key: str, value: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a fact to both the keyword store and vector backend."""
        self.facts[key] = value
        self.backend.add(
            doc_id=f"user_{key.replace(' ', '_')}",
            text=f"{key} is {value}",
            metadata={"source": "user", "key": key, **(metadata or {})},
        )

    def retrieve_context_with_chunks(self, query: str) -> list[EvidenceChunk]:
        """Retrieve context as structured EvidenceChunk list."""
        results = self.backend.query(query, n_results=3)
        if not results:
            return []
        chunks = []
        for r in results:
            chunks.append(
                EvidenceChunk(
                    text=r["text"],
                    distance=r.get("distance", 0.0),
                    source=r.get("metadata", {}).get("source", ""),
                )
            )
        return chunks

    def retrieve_context(self, query: str) -> str | None:
        """Retrieve context using vector similarity with keyword fallback.

        1. Try vector backend (semantic similarity)
        2. Fall back to keyword matching if no results
        """
        # Try vector search first
        results = self.backend.query(query, n_results=3)
        if results:
            context = "; ".join(r["text"] for r in results)
            self.logger.info(
                "Vector retrieval: %d results for '%s'", len(results), query
            )
            return context

        # Fall back to keyword matching
        result: str | None = super().retrieve_context(query)
        return result
