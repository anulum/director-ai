# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Third-party vector backends

"""Third-party vector backends.

Each class gates its import path behind an ``ImportError`` so
users only pay the dependency cost for the vendors they enable.
Covers Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch, and
ColBERT.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any
from urllib.parse import urlparse

from .base import VectorBackend

logger = logging.getLogger("DirectorAI.VectorStore.Vendors")

__all__ = [
    "PineconeBackend",
    "WeaviateBackend",
    "QdrantBackend",
    "FAISSBackend",
    "ElasticsearchBackend",
    "ColBERTBackend",
]


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
            import pinecone
        except ImportError as e:
            raise ImportError(
                "PineconeBackend requires pinecone. "
                "Install with: pip install director-ai[pinecone]",
            ) from e
        self._pc = pinecone.Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self._namespace = namespace
        self._embed_fn = embed_fn
        self._texts: dict[str, str] = {}

    def _embed(self, text: str) -> list[float]:
        """Embed text using the configured Pinecone embedding callback."""
        if self._embed_fn is None:
            raise ValueError("PineconeBackend requires embed_fn for text embedding")
        result: list[float] = self._embed_fn(text)
        return result

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upsert one document vector into Pinecone."""
        vector = self._embed(text)
        meta = {**(metadata or {}), "text": text}
        self._index.upsert(
            vectors=[(doc_id, vector, meta)],
            namespace=self._namespace,
        )
        self._texts[doc_id] = text

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Query Pinecone and return normalised vector result dictionaries."""
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
                },
            )
        return docs

    def count(self) -> int:
        """Return Pinecone vector count for the configured namespace."""
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
        grpc_port: int = 50051,
        grpc_host: str | None = None,
    ) -> None:
        try:
            import weaviate
        except ImportError as e:
            raise ImportError(
                "WeaviateBackend requires weaviate-client. "
                "Install with: pip install director-ai[weaviate]",
            ) from e
        # weaviate-client v4 opens an HTTP + gRPC connection on construction
        # (the v3 ``weaviate.Client(url=...)`` surface was removed). Derive the
        # HTTP host/port/scheme from ``url`` and pair it with the gRPC endpoint.
        parsed = urlparse(url)
        http_secure = parsed.scheme == "https"
        http_host = parsed.hostname or "localhost"
        http_port = parsed.port or (443 if http_secure else 8080)
        auth = weaviate.classes.init.Auth.api_key(api_key) if api_key else None
        self._client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=grpc_host or http_host,
            grpc_port=grpc_port,
            grpc_secure=http_secure,
            auth_credentials=auth,
        )
        self._weaviate = weaviate
        self._class_name = class_name
        self._embed_fn = embed_fn

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert one object with an optional explicit vector (v4 collections API)."""
        props = {"text": text, "doc_id": doc_id, **(metadata or {})}
        vector = self._embed_fn(text) if self._embed_fn else None
        collection = self._client.collections.get(self._class_name)
        collection.data.insert(
            properties=props,
            uuid=self._weaviate.util.generate_uuid5(doc_id),
            vector=vector,
        )

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Near-vector or near-text search via the v4 collections API."""
        query_classes = self._weaviate.classes.query
        filters = (
            query_classes.Filter.by_property("tenant_id").equal(tenant_id)
            if tenant_id
            else None
        )
        return_metadata = query_classes.MetadataQuery(distance=True)
        collection = self._client.collections.get(self._class_name)
        if self._embed_fn:
            response = collection.query.near_vector(
                near_vector=self._embed_fn(text),
                limit=n_results,
                filters=filters,
                return_metadata=return_metadata,
            )
        else:
            response = collection.query.near_text(
                query=text,
                limit=n_results,
                filters=filters,
                return_metadata=return_metadata,
            )

        docs: list[dict[str, Any]] = []
        for obj in response.objects:
            props = dict(obj.properties)
            distance = getattr(obj.metadata, "distance", None)
            docs.append(
                {
                    "id": props.get("doc_id", str(obj.uuid)),
                    "text": props.get("text", ""),
                    "distance": float(distance) if distance is not None else 0.0,
                    "metadata": props,
                },
            )
        return docs

    def count(self) -> int:
        """Return the object count reported by the Weaviate server."""
        collection = self._client.collections.get(self._class_name)
        result = collection.aggregate.over_all(total_count=True)
        return int(result.total_count or 0)


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
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "QdrantBackend requires qdrant-client. "
                "Install with: pip install director-ai[qdrant]",
            ) from e
        self._client = QdrantClient(host=url, port=port)
        self._collection = collection_name
        self._embed_fn = embed_fn
        self._vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection when it is not present."""
        from qdrant_client.models import Distance, VectorParams

        try:
            self._client.get_collection(self._collection)
        except Exception:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upsert one embedded point into Qdrant."""
        from qdrant_client.models import PointStruct

        if self._embed_fn is None:
            raise ValueError("QdrantBackend requires embed_fn for text embedding")
        vector = self._embed_fn(text)
        payload = {"text": text, "doc_id": doc_id, **(metadata or {})}
        # Qdrant point ids must be an unsigned integer or a UUID, so an arbitrary
        # string doc id (e.g. "doc-2") is rejected by the server. Map it to a
        # deterministic UUID — the same doc id always yields the same point id,
        # so re-adding a document upserts rather than duplicating — and keep the
        # original id in the payload so ``query`` can return it verbatim.
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id)),
            vector=vector,
            payload=payload,
        )
        self._client.upsert(collection_name=self._collection, points=[point])

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Search Qdrant with an optional tenant filter."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if tenant_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id),
                    ),
                ],
            )

        if self._embed_fn is None:
            raise ValueError("QdrantBackend requires embed_fn for text embedding")
        vector = self._embed_fn(text)
        # qdrant-client >= 1.10 removed Client.search; query_points is the
        # supported API and returns a response whose .points are the hits.
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=n_results,
            query_filter=query_filter,
            with_payload=True,
        )
        docs: list[dict[str, Any]] = []
        for hit in response.points:
            payload = hit.payload or {}
            docs.append(
                {
                    "id": payload.get("doc_id", str(hit.id)),
                    "text": payload.get("text", ""),
                    "distance": 1.0 - hit.score,
                    "metadata": payload,
                },
            )
        return docs

    def count(self) -> int:
        """Return Qdrant point count for the configured collection."""
        info = self._client.get_collection(self._collection)
        return int(info.points_count or 0)


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
        ivf_nlist: int = 1,
    ) -> None:
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "FAISSBackend requires faiss-cpu or faiss-gpu. "
                "Install with: pip install director-ai[faiss]",
            ) from e

        self._faiss = faiss
        if not isinstance(ivf_nlist, int) or isinstance(ivf_nlist, bool):
            raise ValueError("ivf_nlist must be an integer")
        if ivf_nlist < 1:
            raise ValueError("ivf_nlist must be positive")

        # All index variants derive from ``faiss.Index``; annotate the base so
        # flat, bootstrap, and trained IVF branches share one declared type.
        self._index: faiss.Index
        self._ivf_index: faiss.Index | None = None
        self._ivf_nlist = ivf_nlist
        if index_type == "ivf":
            quantizer = faiss.IndexFlatIP(vector_size)
            self._ivf_index = faiss.IndexIVFFlat(quantizer, vector_size, ivf_nlist)
            self._index = faiss.IndexFlatIP(vector_size)
            self._needs_training = True
        else:
            self._index = faiss.IndexFlatIP(vector_size)
            self._needs_training = False

        self._embed_fn = embed_fn
        self._docs: list[dict[str, Any]] = []
        self._vectors: list[Any] = []
        self._trained = False
        self._lock = threading.Lock()

    def _embed(self, text: str) -> Any:
        """Embed and L2-normalise text for FAISS inner-product search."""
        if self._embed_fn is None:
            raise ValueError("FAISSBackend requires embed_fn for text embedding")
        import numpy as np

        vec = np.asarray(self._embed_fn(text), dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(vec)
        return vec

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add one embedded document to the FAISS index."""
        vec = self._embed(text)
        with self._lock:
            self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})
            if self._needs_training and not self._trained:
                self._vectors.append(vec.copy())
            if (
                self._needs_training
                and not self._trained
                and self._ivf_index is not None
                and len(self._vectors) >= self._ivf_nlist
            ):
                import numpy as np

                training_matrix = np.vstack(self._vectors).astype(
                    "float32",
                    copy=False,
                )
                self._ivf_index.train(training_matrix)
                self._ivf_index.add(training_matrix)
                self._index = self._ivf_index
                self._trained = True
                self._needs_training = False
                self._vectors.clear()
            else:
                self._index.add(vec)

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Search FAISS and apply tenant filtering in metadata.

        With a tenant filter the search over-fetch expands
        geometrically (3×, 6×, 12×, …) until ``n_results`` tenant
        documents are found or the whole index has been searched — a
        fixed over-fetch silently starves tenants whose documents sit
        beyond the first window on a shared index.
        """
        vec = self._embed(text)
        with self._lock:
            if not self._docs:
                return []
            total = len(self._docs)
            docs_snapshot = list(self._docs)
            k = min(n_results * 3 if tenant_id else n_results, total)
            while True:
                distances, indices = self._index.search(vec, k)
                results = self._filter_hits(
                    distances[0],
                    indices[0],
                    docs_snapshot,
                    n_results,
                    tenant_id,
                )
                if len(results) >= n_results or k >= total or not tenant_id:
                    return results
                k = min(k * 2, total)

    @staticmethod
    def _filter_hits(
        distances: Any,
        indices: Any,
        docs_snapshot: list[dict[str, Any]],
        n_results: int,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Map raw FAISS hits to result rows under the tenant filter."""
        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances, indices, strict=True):
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
        """Return the number of documents tracked beside the FAISS index."""
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
            from elasticsearch import Elasticsearch
        except ImportError as e:
            raise ImportError(
                "ElasticsearchBackend requires elasticsearch. "
                "Install with: pip install director-ai[elasticsearch]",
            ) from e

        kwargs: dict[str, Any] = {"hosts": [url]}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = Elasticsearch(**kwargs)
        self._index = index_name
        self._embed_fn = embed_fn
        self._vector_size = vector_size
        self._hybrid_weight = max(0.0, min(1.0, hybrid_weight))
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the Elasticsearch index and dense-vector mapping if needed."""
        if self._client.indices.exists(index=self._index):
            return
        mappings: dict[str, Any] = {
            "properties": {
                "text": {"type": "text"},
                "doc_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
            },
        }
        if self._embed_fn:
            mappings["properties"]["embedding"] = {
                "type": "dense_vector",
                "dims": self._vector_size,
                "index": True,
                "similarity": "cosine",
            }
        self._client.indices.create(index=self._index, mappings=mappings)

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index one document into Elasticsearch."""
        body: dict[str, Any] = {"text": text, "doc_id": doc_id, **(metadata or {})}
        if self._embed_fn:
            body["embedding"] = self._embed_fn(text)
        self._client.index(index=self._index, id=doc_id, document=body)

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Search Elasticsearch using hybrid or BM25 retrieval."""
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
                index=self._index,
                size=n_results,
                query=body_query,
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
                },
            )
        return docs

    def count(self) -> int:
        """Return the document count reported by the Elasticsearch server."""
        return int(self._client.count(index=self._index)["count"])


class ColBERTBackend(VectorBackend):
    """ColBERT v2 late-interaction retrieval via RAGatouille.

    Each token gets its own embedding vector. Matching uses MaxSim
    across all token pairs — much more accurate than single-vector
    bi-encoders for partial and domain-specific matches.

    Requires a separately installed safe RAGatouille stack.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        index_name: str = "director_colbert",
        persist_dir: str = "",
    ) -> None:
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError as e:
            raise ImportError(
                "ColBERTBackend requires ragatouille. "
                "Install a safe ragatouille release separately; "
                "director-ai[colbert] is intentionally empty while the "
                "upstream stack pulls a vulnerable nltk dependency.",
            ) from e
        self._model = RAGPretrainedModel.from_pretrained(model_name)
        self._index_name = index_name
        self._persist_dir = persist_dir
        self._docs: list[dict[str, Any]] = []
        self._indexed = False
        self._lock = threading.Lock()

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Stage one document for the ColBERT late-interaction index."""
        with self._lock:
            self._docs.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata or {},
                }
            )
            self._indexed = False

    def _ensure_index(self) -> None:
        """Build or refresh the ColBERT index when staged docs changed."""
        if self._indexed or not self._docs:
            return
        texts = [d["text"] for d in self._docs]
        doc_ids = [d["id"] for d in self._docs]
        kwargs: dict[str, Any] = {
            "collection": texts,
            "document_ids": doc_ids,
            "index_name": self._index_name,
            "split_documents": False,
        }
        if self._persist_dir:
            kwargs["use_faiss"] = True
        self._model.index(**kwargs)
        self._indexed = True

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Search the ColBERT index and return normalised result dictionaries."""
        with self._lock:
            self._ensure_index()
            if not self._indexed:
                return []
        results = self._model.search(text, k=n_results)
        docs: list[dict[str, Any]] = []
        for r in results:
            doc_meta = {}
            for d in self._docs:
                if d["id"] == r.get("document_id"):
                    doc_meta = d["metadata"]
                    break
            docs.append(
                {
                    "id": r.get("document_id", ""),
                    "text": r.get("content", ""),
                    "distance": 1.0 - r.get("score", 0.0),
                    "metadata": {**doc_meta, "doc_id": r.get("document_id", "")},
                }
            )
        return docs

    def count(self) -> int:
        """Return the number of staged ColBERT documents."""
        with self._lock:
            return len(self._docs)
