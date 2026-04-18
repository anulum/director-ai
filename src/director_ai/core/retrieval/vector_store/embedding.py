# SPDX-License-Identifier: AGPL-3.0-or-later
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
from typing import Any

from .base import VectorBackend, logger

__all__ = ["SentenceTransformerBackend", "ChromaBackend"]


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
                "Install with: pip install director-ai[vector]",
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
                    "sentence-transformers not installed, using Chroma default embedder",
                )

        self._collection = self._client.get_or_create_collection(**create_kwargs)

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:

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
        return int(self._collection.count())
