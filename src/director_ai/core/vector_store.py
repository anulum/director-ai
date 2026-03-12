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
import threading
from abc import ABC, abstractmethod
from typing import Any

from .knowledge import GroundTruthStore
from .metrics import metrics
from .otel import trace_vector_add, trace_vector_query
from .types import EvidenceChunk

# Re-export recommended model name for documentation
RECOMMENDED_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

logger = logging.getLogger("DirectorAI.VectorStore")

# ── VectorBackend registry (mirrors backends.py pattern) ─────────

_VECTOR_REGISTRY: dict[str, type[VectorBackend]] = {}
_VECTOR_EP_LOADED = False


def register_vector_backend(name: str, cls: type[VectorBackend]) -> None:
    """Register a vector backend class under *name*."""
    if not (isinstance(cls, type) and issubclass(cls, VectorBackend)):
        raise TypeError(f"{cls!r} must be a VectorBackend subclass")
    _VECTOR_REGISTRY[name] = cls
    logger.debug("Registered vector backend: %s", name)


def get_vector_backend(name: str) -> type[VectorBackend]:
    """Look up a registered vector backend by name. Raises KeyError if unknown."""
    _load_vector_entry_points()
    if name not in _VECTOR_REGISTRY:
        raise KeyError(
            f"Unknown vector backend {name!r}. Available: {list(_VECTOR_REGISTRY)}"
        )
    return _VECTOR_REGISTRY[name]


def list_vector_backends() -> dict[str, type[VectorBackend]]:
    """Return all registered vector backends."""
    _load_vector_entry_points()
    return dict(_VECTOR_REGISTRY)


def _load_vector_entry_points() -> None:
    """Discover backends from ``director_ai.vector_backends`` entry points (once)."""
    global _VECTOR_EP_LOADED
    if _VECTOR_EP_LOADED:
        return
    _VECTOR_EP_LOADED = True
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group: list = (  # type: ignore[assignment]
            eps.get("director_ai.vector_backends", [])
            if isinstance(eps, dict)
            else eps.select(group="director_ai.vector_backends")
        )
        for ep in group:
            try:
                cls = ep.load()
                if ep.name not in _VECTOR_REGISTRY:  # pragma: no cover
                    register_vector_backend(ep.name, cls)
            except (ImportError, AttributeError, TypeError) as exc:  # pragma: no cover
                logger.warning(
                    "Failed to load vector backend entry point %s: %s", ep.name, exc
                )
    except ImportError:
        pass


class VectorBackend(ABC):
    """Protocol for vector database backends."""

    @abstractmethod
    def add(  # type: ignore[override]
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None: ...  # pragma: no cover

    @abstractmethod
    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]: ...  # pragma: no cover

    @abstractmethod
    def count(self) -> int: ...  # pragma: no cover

    async def aadd(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Async add — delegates to sync add via executor by default."""
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.add, doc_id, text, metadata)

    async def aquery(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:
        """Async query — delegates to sync query via executor by default."""
        import asyncio
        import functools

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self.query, text, n_results=n_results, tenant_id=tenant_id
            ),
        )


class InMemoryBackend(VectorBackend):
    """Simple in-memory cosine-similarity backend (no external deps).

    Uses TF-IDF-like word overlap for embedding approximation.
    Suitable for testing and small fact stores.
    """

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(  # type: ignore[override]
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:
        with self._lock:
            snapshot = list(self._docs)

        if not snapshot:
            return []
        docs = [
            d
            for d in snapshot
            if not tenant_id or d["metadata"].get("tenant_id") == tenant_id
        ]
        if not docs:
            return []
        import re

        _strip = re.compile(r"[^\w\s]")
        query_words = set(_strip.sub("", text).lower().split())
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in docs:
            doc_words = set(_strip.sub("", doc["text"]).lower().split())
            overlap = len(query_words & doc_words)
            total = max(len(query_words | doc_words), 1)
            similarity = overlap / total
            scored.append((similarity, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        scored_sliced = scored[:n_results]
        for sim, doc2 in scored_sliced:
            if sim > 0:
                results.append({**doc2, "distance": 1.0 - sim})
        return results

    def count(self) -> int:
        with self._lock:
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
        self._lock = threading.Lock()

    def add(  # type: ignore[override]
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        import numpy as _np

        emb = self._model.encode(text, normalize_embeddings=True)
        with self._lock:
            self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})
            self._embeddings.append(_np.asarray(emb, dtype=_np.float32))

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:
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

        q_emb = self._model.encode(text, normalize_embeddings=True)
        q_emb = _np.asarray(q_emb, dtype=_np.float32)  # type: ignore[assignment]
        similarities = [float(_np.dot(q_emb, e)) for e in filtered_embs]
        indices = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )
        results = []
        for idx in indices[:n_results]:
            if similarities[idx] > 0:
                results.append(
                    {
                        **filtered_docs[idx],
                        "distance": 1.0 - similarities[idx],
                    }
                )
        return results

    def count(self) -> int:
        with self._lock:
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

    def add(  # type: ignore[override]
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:

        self._collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata] if metadata else None,
        )

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:

        where: dict[str, str] | None = {"tenant_id": tenant_id} if tenant_id else None
        results = self._collection.query(
            query_texts=[text],
            n_results=n_results,
            where=where,  # type: ignore[arg-type]
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
                    }
                )
        return docs

    def count(self) -> int:
        return int(self._collection.count())


class HybridBackend(VectorBackend):
    """BM25 + dense retrieval with Reciprocal Rank Fusion (RRF).

    Wraps any VectorBackend. Maintains a parallel BM25 index over
    the same documents. At query time, runs both sparse (BM25) and
    dense (wrapped backend) retrieval, then fuses results via RRF.

    No external dependencies — uses a built-in BM25 implementation.

    RRF: score(d) = 1/(k + rank_sparse) + 1/(k + rank_dense).
    Croft et al. 2009, default k=60.
    """

    def __init__(
        self,
        base: VectorBackend,
        rrf_k: int = 60,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        fetch_multiplier: int = 3,
    ) -> None:
        self._base = base
        self._rrf_k = rrf_k
        self._sparse_w = sparse_weight
        self._dense_w = dense_weight
        self._fetch_mul = fetch_multiplier
        self._docs: list[dict[str, Any]] = []
        self._doc_tfs: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        self._total_len = 0
        self._lock = threading.Lock()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re

        return re.findall(r"\w+", text.lower())

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self._base.add(doc_id, text, metadata)
        tokens = self._tokenize(text)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        with self._lock:
            self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})
            self._doc_tfs.append(tf)
            self._total_len += len(tokens)
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

    def _bm25_query(
        self, text: str, n_results: int, tenant_id: str
    ) -> list[dict[str, Any]]:
        """BM25 scoring: k1=1.2, b=0.75."""
        import math

        query_tokens = self._tokenize(text)
        if not query_tokens:
            return []

        with self._lock:
            docs = list(self._docs)
            tfs = list(self._doc_tfs)
            df = dict(self._df)
            total_len = self._total_len

        n = len(docs)
        if n == 0:
            return []
        avgdl = total_len / n
        k1, b = 1.2, 0.75

        scores: list[tuple[float, int]] = []
        for i, (doc, tf) in enumerate(zip(docs, tfs, strict=False)):
            if tenant_id and doc["metadata"].get("tenant_id") != tenant_id:
                continue
            dl = sum(tf.values())
            score = 0.0
            for qt in query_tokens:
                f = tf.get(qt, 0)
                if f == 0:
                    continue
                idf = math.log((n - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1.0)
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:n_results]:
            if score > 0:
                results.append({**docs[idx], "distance": 1.0 / (1.0 + score)})
        return results

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:
        fetch_n = n_results * self._fetch_mul

        # Run both retrieval paths
        sparse_results = self._bm25_query(text, fetch_n, tenant_id)
        dense_results = self._base.query(text, n_results=fetch_n, tenant_id=tenant_id)

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(sparse_results):
            did = doc["id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + self._sparse_w / (
                self._rrf_k + rank + 1
            )
            doc_map[did] = doc

        for rank, doc in enumerate(dense_results):
            did = doc["id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + self._dense_w / (
                self._rrf_k + rank + 1
            )
            doc_map[did] = doc

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[did] for did, _ in ranked[:n_results]]

    def count(self) -> int:
        return self._base.count()


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

    def add(  # type: ignore[override]
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self._base.add(doc_id, text, metadata)

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:
        candidates = self._base.query(
            text, n_results=n_results * self._multiplier, tenant_id=tenant_id
        )
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

    def add(  # type: ignore[override]
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:

        vector = self._embed(text)
        meta = {**(metadata or {}), "text": text}
        self._index.upsert(
            vectors=[(doc_id, vector, meta)],
            namespace=self._namespace,
        )
        self._texts[doc_id] = text

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:

        vector = self._embed(text)

        filter_dict = None
        if tenant_id:
            filter_dict = {"tenant_id": {"$eq": tenant_id}}

        results = self._index.query(
            vector=vector,
            top_k=n_results,
            namespace=self._namespace,
            include_metadata=True,
            filter=filter_dict,
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

    def add(  # type: ignore[override]
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

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:

        where_filter = None
        if tenant_id:
            where_filter = {
                "path": ["tenant_id"],
                "operator": "Equal",
                "valueText": tenant_id,
            }

        query_builder = self._client.query.get(self._class_name, ["text", "doc_id"])
        if self._embed_fn:
            vector = self._embed_fn(text)
            query_builder = query_builder.with_near_vector({"vector": vector})
        else:
            query_builder = query_builder.with_near_text({"concepts": [text]})

        query_builder = query_builder.with_limit(n_results).with_additional(
            ["distance", "id"]
        )

        if where_filter:
            query_builder = query_builder.with_where(where_filter)

        result = query_builder.do()
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

    def add(  # type: ignore[override]
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:

        from qdrant_client.models import PointStruct

        if self._embed_fn is None:
            raise ValueError("QdrantBackend requires embed_fn for text embedding")
        vector = self._embed_fn(text)
        payload = {"text": text, **(metadata or {})}
        point = PointStruct(id=doc_id, vector=vector, payload=payload)
        self._client.upsert(collection_name=self._collection, points=[point])

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if tenant_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id),
                    )
                ]
            )

        if self._embed_fn is None:
            raise ValueError("QdrantBackend requires embed_fn for text embedding")
        vector = self._embed_fn(text)
        results = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=n_results,
            query_filter=query_filter,
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


class FAISSBackend(VectorBackend):
    """FAISS backend for in-process dense vector search.

    Operates entirely in-process with no external server. Ideal for
    edge/offline deployments where sub-millisecond retrieval matters.

    Requires ``pip install director-ai[faiss]``.
    """

    def __init__(
        self,
        embed_fn: Any = None,
        vector_size: int = 384,
        index_type: str = "flat",
    ) -> None:
        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "FAISSBackend requires faiss-cpu or faiss-gpu. "
                "Install with: pip install director-ai[faiss]"
            ) from e

        self._faiss = faiss
        if index_type == "ivf":
            quantizer = faiss.IndexFlatIP(vector_size)
            self._index = faiss.IndexIVFFlat(quantizer, vector_size, 16)
            self._needs_training = True
        else:
            self._index = faiss.IndexFlatIP(vector_size)
            self._needs_training = False

        self._embed_fn = embed_fn
        self._docs: list[dict[str, Any]] = []
        self._trained = False
        self._lock = threading.Lock()

    def _embed(self, text: str) -> Any:
        if self._embed_fn is None:
            raise ValueError("FAISSBackend requires embed_fn for text embedding")
        import numpy as np

        vec = np.asarray(self._embed_fn(text), dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(vec)
        return vec

    def add(  # type: ignore[override]
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        vec = self._embed(text)
        with self._lock:
            if self._needs_training and not self._trained:
                self._index.train(vec)
                self._trained = True
            self._index.add(vec)
            self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:
        vec = self._embed(text)
        with self._lock:
            if not self._docs:
                return []
            k = min(n_results * 3 if tenant_id else n_results, len(self._docs))
            distances, indices = self._index.search(vec, k)
            docs_snapshot = list(self._docs)

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0 or idx >= len(docs_snapshot):
                continue
            doc = docs_snapshot[idx]
            if tenant_id and doc["metadata"].get("tenant_id") != tenant_id:
                continue
            results.append({**doc, "distance": 1.0 - float(dist)})
            if len(results) >= n_results:
                break
        return results

    def count(self) -> int:
        with self._lock:
            return len(self._docs)


class ElasticsearchBackend(VectorBackend):
    """Elasticsearch backend with hybrid BM25 + dense retrieval.

    Combines lexical (BM25) and semantic (kNN) search via Elasticsearch's
    built-in vector search. Requires Elasticsearch 8.x+ with vector support.

    Requires ``pip install director-ai[elasticsearch]``.
    """

    def __init__(
        self,
        url: str = "http://localhost:9200",
        api_key: str | None = None,
        index_name: str = "director_facts",
        embed_fn: Any = None,
        vector_size: int = 384,
        hybrid_weight: float = 0.5,
    ) -> None:
        try:
            from elasticsearch import Elasticsearch  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "ElasticsearchBackend requires elasticsearch. "
                "Install with: pip install director-ai[elasticsearch]"
            ) from e

        kwargs: dict[str, Any] = {"hosts": [url]}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = Elasticsearch(**kwargs)
        self._index = index_name
        self._embed_fn = embed_fn
        self._vector_size = vector_size
        self._hybrid_weight = max(0.0, min(1.0, hybrid_weight))
        self._count = 0
        self._ensure_index()

    def _ensure_index(self) -> None:
        if self._client.indices.exists(index=self._index):
            return
        mappings: dict[str, Any] = {
            "properties": {
                "text": {"type": "text"},
                "doc_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
            }
        }
        if self._embed_fn:
            mappings["properties"]["embedding"] = {
                "type": "dense_vector",
                "dims": self._vector_size,
                "index": True,
                "similarity": "cosine",
            }
        self._client.indices.create(index=self._index, mappings=mappings)

    def add(  # type: ignore[override]
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        body: dict[str, Any] = {"text": text, "doc_id": doc_id, **(metadata or {})}
        if self._embed_fn:
            body["embedding"] = self._embed_fn(text)
        self._client.index(index=self._index, id=doc_id, document=body)
        self._count += 1

    def query(
        self, text: str, n_results: int = 3, tenant_id: str = ""
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        if tenant_id:
            filters.append({"term": {"tenant_id": tenant_id}})

        has_vector = self._embed_fn is not None

        if has_vector and self._hybrid_weight < 1.0:
            # Hybrid: RRF over BM25 + kNN sub-queries
            bm25_query: dict[str, Any] = {"match": {"text": text}}
            knn_query: dict[str, Any] = {
                "field": "embedding",
                "query_vector": self._embed_fn(text),
                "k": n_results,
                "num_candidates": n_results * 5,
            }
            if filters:
                bm25_query = {"bool": {"must": bm25_query, "filter": filters}}
                knn_query["filter"] = {"bool": {"filter": filters}}

            resp = self._client.search(
                index=self._index,
                size=n_results,
                query=bm25_query,
                knn=knn_query,
                rank={"rrf": {"window_size": n_results * 5}},
            )
        else:
            # Pure BM25 fallback
            body_query: dict[str, Any] = {"match": {"text": text}}
            if filters:
                body_query = {"bool": {"must": body_query, "filter": filters}}
            resp = self._client.search(
                index=self._index, size=n_results, query=body_query
            )

        docs: list[dict[str, Any]] = []
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            score = hit.get("_score", 0.0) or 0.0
            docs.append(
                {
                    "id": hit["_id"],
                    "text": source.get("text", ""),
                    "distance": 1.0 / (1.0 + score),
                    "metadata": {
                        k: v
                        for k, v in source.items()
                        if k not in ("text", "embedding")
                    },
                }
            )
        return docs

    def count(self) -> int:
        return self._count


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
        tenant_id: str = "",
    ) -> None:
        super().__init__()
        self.backend = backend if backend is not None else InMemoryBackend()
        self.tenant_id = tenant_id

    def add_fact(self, key: str, value: str, tenant_id: str = "") -> None:
        """Alias for add() — also populates parent keyword store."""
        self.facts[key] = value
        self.add(key, value, tenant_id=tenant_id)

    def ingest(self, texts: list[str], tenant_id: str = "") -> int:
        """Bulk-add plain text documents into the vector backend."""
        for i, text in enumerate(texts):
            self.backend.add(
                doc_id=f"ingest_{i}_{tenant_id}",
                text=text,
                metadata={"source": "ingest", "tenant_id": tenant_id},
            )
        logger.info("Ingested %d documents into vector backend.", len(texts))
        return len(texts)

    def add(  # type: ignore[override]
        self,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "",
    ) -> None:
        import time

        doc_id = f"{tenant_id}::{key}" if tenant_id else key
        combined_text = f"{key}: {value}"
        meta = {**(metadata or {}), "key": key, "value": value}
        if tenant_id:
            meta["tenant_id"] = tenant_id

        with trace_vector_add() as span:
            start_time = time.monotonic()
            metrics.inc("knowledge_adds_total")
            try:
                self.backend.add(doc_id=doc_id, text=combined_text, metadata=meta)
                duration = time.monotonic() - start_time
                metrics.observe("knowledge_add_duration_seconds", duration)
                span.set_attribute("vector.doc_id", doc_id)
                span.set_attribute("vector.tenant_id", tenant_id)
            except Exception as e:
                metrics.inc("knowledge_add_errors")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                raise ValueError(f"Failed to add to vector store: {e}") from e

    def retrieve_context(  # type: ignore[override]
        self, query: str, top_k: int = 3, tenant_id: str = ""
    ) -> str | None:
        """Retrieve context as a string (matching parent interface).

        Falls back to keyword-based parent if vector search returns nothing.
        """
        import time

        with trace_vector_query() as span:
            start_time = time.monotonic()
            metrics.inc("knowledge_queries_total")
            try:
                try:
                    results = self.backend.query(
                        query, n_results=top_k, tenant_id=tenant_id
                    )
                except TypeError:
                    # Backend doesn't accept tenant_id
                    results = self.backend.query(query, n_results=top_k)
                span.set_attribute("vector.query.k", top_k)
                span.set_attribute("vector.tenant_id", tenant_id)

                if results:
                    texts = [r["text"] for r in results]
                    duration = time.monotonic() - start_time
                    metrics.observe("knowledge_query_duration_seconds", duration)
                    return "; ".join(texts)

                duration = time.monotonic() - start_time
                metrics.observe("knowledge_query_duration_seconds", duration)
                # Fall back to keyword-based parent
                return super().retrieve_context(query, tenant_id=tenant_id)
            except Exception as e:
                metrics.inc("knowledge_query_errors")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                raise ValueError(f"Failed to query vector store: {e}") from e

    def retrieve_context_with_chunks(
        self, query: str, top_k: int = 3, tenant_id: str = ""
    ) -> list[EvidenceChunk]:
        """Retrieve context as EvidenceChunk objects."""
        import time

        with trace_vector_query() as span:
            start_time = time.monotonic()
            try:
                try:
                    results = self.backend.query(
                        query, n_results=top_k, tenant_id=tenant_id
                    )
                except TypeError:
                    # Backend doesn't accept tenant_id
                    results = self.backend.query(query, n_results=top_k)
                chunks = []
                for r in results:
                    chunks.append(
                        EvidenceChunk(
                            text=r["text"],
                            distance=r.get("distance", 0.0),
                            source=f"vector:{r['id']}",
                        )
                    )
                duration = time.monotonic() - start_time
                metrics.observe("knowledge_query_duration_seconds", duration)
                return chunks
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise ValueError(f"Failed to query vector store: {e}") from e


# ── Auto-register built-in vector backends ───────────────────────

register_vector_backend("memory", InMemoryBackend)
register_vector_backend("sentence-transformer", SentenceTransformerBackend)
register_vector_backend("hybrid", HybridBackend)

try:
    import chromadb as _chromadb  # noqa: F401

    register_vector_backend("chroma", ChromaBackend)
except ImportError:
    pass

try:
    import pinecone as _pinecone  # noqa: F401

    register_vector_backend("pinecone", PineconeBackend)
except ImportError:
    pass

try:
    import weaviate as _weaviate  # noqa: F401

    register_vector_backend("weaviate", WeaviateBackend)
except ImportError:
    pass

try:
    from qdrant_client import QdrantClient as _QdrantClient  # noqa: F401

    register_vector_backend("qdrant", QdrantBackend)
except ImportError:
    pass

try:
    import faiss as _faiss  # noqa: F401

    register_vector_backend("faiss", FAISSBackend)
except ImportError:
    pass

try:
    from elasticsearch import Elasticsearch as _Elasticsearch  # noqa: F401

    register_vector_backend("elasticsearch", ElasticsearchBackend)
except ImportError:
    pass
