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
    ) -> None: ...  # pragma: no cover

    @abstractmethod
    def query(
        self, text: str, n_results: int = 3
    ) -> list[dict[str, Any]]: ...  # pragma: no cover

    @abstractmethod
    def count(self) -> int: ...  # pragma: no cover


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
        q_emb = _np.asarray(q_emb, dtype=_np.float32)  # type: ignore[assignment]
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
                    }
                )
        return docs

    def count(self) -> int:
        return int(self._collection.count())


class RerankedBackend(VectorBackend):
    """Cross-encoder reranking wrapper around any VectorBackend.

    Retrieves ``top_k_multiplier * n_results`` candidates from the base
    backend, then reranks with a cross-encoder model and returns the
    top ``n_results``.

    Requires ``pip install sentence-transformers>=2.2``.
    """

    def __init__(
        self,
        base: VectorBackend,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_multiplier: int = 3,
    ) -> None:
        self._base = base
        self._multiplier = top_k_multiplier
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "RerankedBackend requires sentence-transformers. "
                "Install with: pip install director-ai[reranker]"
            ) from e
        self._reranker = CrossEncoder(reranker_model)

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self._base.add(doc_id, text, metadata)

    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        candidates = self._base.query(text, n_results=n_results * self._multiplier)
        if not candidates:
            return []
        pairs = [(text, c["text"]) for c in candidates]
        scores = self._reranker.predict(pairs)
        ranked = sorted(
            zip(scores, candidates, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )
        return [c for _, c in ranked[:n_results]]

    def count(self) -> int:
        return self._base.count()


class PineconeBackend(VectorBackend):
    """Pinecone vector database backend.

    Requires ``pip install director-ai[pinecone]``.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "",
        embed_fn: Any = None,
    ) -> None:
        try:
            import pinecone  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "PineconeBackend requires pinecone. "
                "Install with: pip install director-ai[pinecone]"
            ) from e
        self._pc = pinecone.Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self._namespace = namespace
        self._embed_fn = embed_fn
        self._texts: dict[str, str] = {}

    def _embed(self, text: str) -> list[float]:
        if self._embed_fn is None:
            raise ValueError("PineconeBackend requires embed_fn for text embedding")
        result: list[float] = self._embed_fn(text)
        return result

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        vector = self._embed(text)
        meta = {**(metadata or {}), "text": text}
        self._index.upsert(
            vectors=[(doc_id, vector, meta)],
            namespace=self._namespace,
        )
        self._texts[doc_id] = text

    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        vector = self._embed(text)
        results = self._index.query(
            vector=vector,
            top_k=n_results,
            namespace=self._namespace,
            include_metadata=True,
        )
        docs: list[dict[str, Any]] = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            docs.append(
                {
                    "id": match["id"],
                    "text": meta.get("text", ""),
                    "distance": 1.0 - match.get("score", 0.0),
                    "metadata": meta,
                }
            )
        return docs

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(self._namespace, {})
        return int(ns_stats.get("vector_count", 0))


class WeaviateBackend(VectorBackend):
    """Weaviate vector database backend.

    Requires ``pip install director-ai[weaviate]``.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: str | None = None,
        class_name: str = "DirectorFact",
        embed_fn: Any = None,
    ) -> None:
        try:
            import weaviate  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "WeaviateBackend requires weaviate-client. "
                "Install with: pip install director-ai[weaviate]"
            ) from e
        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
        self._client = weaviate.Client(url=url, auth_client_secret=auth)
        self._class_name = class_name
        self._embed_fn = embed_fn
        self._count = 0

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        props = {"text": text, "doc_id": doc_id, **(metadata or {})}
        kwargs: dict[str, Any] = {
            "data_object": props,
            "class_name": self._class_name,
            "uuid": doc_id,
        }
        if self._embed_fn:  # pragma: no branch
            kwargs["vector"] = self._embed_fn(text)
        self._client.data_object.create(**kwargs)
        self._count += 1

    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        q = self._client.query.get(self._class_name, ["text", "doc_id"])
        if self._embed_fn:
            vector = self._embed_fn(text)
            q = q.with_near_vector({"vector": vector})
        else:
            q = q.with_near_text({"concepts": [text]})
        q = q.with_limit(n_results).with_additional(["distance", "id"])
        result = q.do()
        docs: list[dict[str, Any]] = []
        items = result.get("data", {}).get("Get", {}).get(self._class_name, [])
        for item in items:
            extra = item.get("_additional", {})
            docs.append(
                {
                    "id": extra.get("id", item.get("doc_id", "")),
                    "text": item.get("text", ""),
                    "distance": float(extra.get("distance", 0.0)),
                    "metadata": {
                        k: v for k, v in item.items() if k not in ("_additional",)
                    },
                }
            )
        return docs

    def count(self) -> int:
        return self._count


class QdrantBackend(VectorBackend):
    """Qdrant vector database backend.

    Requires ``pip install director-ai[qdrant]``.
    """

    def __init__(
        self,
        url: str = "localhost",
        port: int = 6333,
        collection_name: str = "director_facts",
        embed_fn: Any = None,
        vector_size: int = 384,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "QdrantBackend requires qdrant-client. "
                "Install with: pip install director-ai[qdrant]"
            ) from e
        self._client = QdrantClient(host=url, port=port)
        self._collection = collection_name
        self._embed_fn = embed_fn
        self._vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        try:
            self._client.get_collection(self._collection)
        except Exception:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        from qdrant_client.models import PointStruct

        if self._embed_fn is None:
            raise ValueError("QdrantBackend requires embed_fn for text embedding")
        vector = self._embed_fn(text)
        payload = {"text": text, **(metadata or {})}
        point = PointStruct(id=doc_id, vector=vector, payload=payload)
        self._client.upsert(collection_name=self._collection, points=[point])

    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        if self._embed_fn is None:
            raise ValueError("QdrantBackend requires embed_fn for text embedding")
        vector = self._embed_fn(text)
        results = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=n_results,
        )
        docs: list[dict[str, Any]] = []
        for hit in results:
            payload = hit.payload or {}
            docs.append(
                {
                    "id": str(hit.id),
                    "text": payload.get("text", ""),
                    "distance": 1.0 - hit.score,
                    "metadata": payload,
                }
            )
        return docs

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return int(info.points_count)


class VectorGroundTruthStore(GroundTruthStore):
    """Ground truth store with vector-based semantic retrieval.

    Extends the keyword-based ``GroundTruthStore`` with embedding-based
    similarity search. Falls back to keyword matching when the vector
    backend returns no results.

    Parameters
    ----------
    backend : VectorBackend — vector DB backend (default: InMemoryBackend).
    """

    def __init__(
        self,
        backend: VectorBackend | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend if backend is not None else InMemoryBackend()

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
