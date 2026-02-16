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
            scored.append((overlap / total, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:n_results] if _ > 0]

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
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        results = self._collection.query(query_texts=[text], n_results=n_results)
        docs: list[dict[str, Any]] = []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        ids = results.get("ids")
        if results and documents:
            for i, doc_text in enumerate(documents[0]):
                meta = metadatas[0][i] if metadatas else {}
                doc_id = ids[0][i] if ids else f"doc_{i}"
                docs.append({"id": doc_id, "text": doc_text, "metadata": meta})
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
