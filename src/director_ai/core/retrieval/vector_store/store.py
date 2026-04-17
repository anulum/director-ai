# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — VectorGroundTruthStore

"""``VectorGroundTruthStore`` — semantic RAG store on top of any
``VectorBackend``.

Extends the keyword-based :class:`GroundTruthStore` with
embedding-based similarity search and exposes a ``grounded()``
factory that wires the recommended hybrid (BM25 + dense) recipe.
"""

from __future__ import annotations

from typing import Any

from ...metrics import metrics
from ...otel import trace_vector_add, trace_vector_query
from ...types import EvidenceChunk
from ..knowledge import GroundTruthStore
from .base import (
    RECOMMENDED_EMBEDDING_MODEL,
    InMemoryBackend,
    VectorBackend,
    logger,
)
from .composite import HybridBackend
from .embedding import SentenceTransformerBackend

__all__ = ["VectorGroundTruthStore"]


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

    def _resolved_tenant_id(self, tenant_id: str = "") -> str:
        return tenant_id or self.tenant_id

    def add_fact(self, key: str, value: str, tenant_id: str = "") -> None:
        """Alias for add() — also populates parent keyword store."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        fact_key = f"{tenant_id}:{key}" if tenant_id else key
        self.facts[fact_key] = value
        self.add(key, value, tenant_id=tenant_id)

    def ingest(self, texts: list[str], tenant_id: str = "") -> int:
        """Bulk-add plain text documents into the vector backend."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        for i, text in enumerate(texts):
            metadata = {"source": "ingest"}
            if tenant_id:
                metadata["tenant_id"] = tenant_id
            self.backend.add(
                doc_id=f"ingest_{i}_{tenant_id}",
                text=text,
                metadata=metadata,
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

        tenant_id = self._resolved_tenant_id(tenant_id)
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
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "",
    ) -> str | None:
        """Retrieve context as a string (matching parent interface).

        Falls back to keyword-based parent if vector search returns nothing.
        """
        import time

        tenant_id = self._resolved_tenant_id(tenant_id)
        with trace_vector_query() as span:
            start_time = time.monotonic()
            metrics.inc("knowledge_queries_total")
            try:
                try:
                    results = self.backend.query(
                        query,
                        n_results=top_k,
                        tenant_id=tenant_id,
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
                return super().retrieve_context(
                    query,
                    tenant_id=tenant_id,
                    top_k=top_k,
                )
            except Exception as e:
                metrics.inc("knowledge_query_errors")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                raise ValueError(f"Failed to query vector store: {e}") from e

    @classmethod
    def grounded(
        cls,
        embedding_model: str = RECOMMENDED_EMBEDDING_MODEL,
        use_hybrid: bool = True,
        rrf_k: int = 60,
        tenant_id: str = "",
    ) -> VectorGroundTruthStore:
        """Factory for the recommended grounded retrieval recipe.

        Sets up hybrid retrieval (BM25 + dense) with a sentence-transformer
        embedding model. This is the intended production path for domain
        profiles (medical, finance, legal) where NLI-only scoring has
        100% FPR without KB grounding.

        Usage::

            store = VectorGroundTruthStore.grounded()
            store.ingest(["Your product documentation...", ...])
            scorer = CoherenceScorer(ground_truth_store=store, use_nli=True)

        Parameters
        ----------
        embedding_model : str
            HuggingFace model ID for dense embeddings.
            Default: ``BAAI/bge-large-en-v1.5``.
        use_hybrid : bool
            Wrap dense backend with BM25 + RRF fusion (default True).
        rrf_k : int
            Reciprocal Rank Fusion parameter (default 60).
        tenant_id : str
            Default tenant scope for multi-tenant deployments.
        """
        dense: VectorBackend
        try:
            dense = SentenceTransformerBackend(model_name=embedding_model)
        except Exception:
            logger.warning(
                "sentence-transformers not available, falling back to InMemoryBackend. "
                "Install with: pip install director-ai[vector]"
            )
            dense = InMemoryBackend()

        backend = HybridBackend(base=dense, rrf_k=rrf_k) if use_hybrid else dense

        return cls(backend=backend, tenant_id=tenant_id)

    def retrieve_context_with_chunks(
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "",
    ) -> list[EvidenceChunk]:
        """Retrieve context as EvidenceChunk objects."""
        import time

        tenant_id = self._resolved_tenant_id(tenant_id)
        with trace_vector_query() as span:
            start_time = time.monotonic()
            try:
                try:
                    results = self.backend.query(
                        query,
                        n_results=top_k,
                        tenant_id=tenant_id,
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
                        ),
                    )
                duration = time.monotonic() - start_time
                metrics.observe("knowledge_query_duration_seconds", duration)
                if not chunks:
                    return super().retrieve_context_with_chunks(
                        query,
                        top_k=top_k,
                        tenant_id=tenant_id,
                    )
                return chunks
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise ValueError(f"Failed to query vector store: {e}") from e
