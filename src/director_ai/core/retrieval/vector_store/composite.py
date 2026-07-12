# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Composite vector backends (Hybrid, Reranked)

"""Composite vector backends that decorate a base backend.

``HybridBackend`` fuses BM25 sparse retrieval with the wrapped
backend via Reciprocal Rank Fusion.
``RerankedBackend`` post-processes base results with a
cross-encoder.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

from director_ai.core.accelerator_fallback import sum_i64

try:  # pragma: no cover - optional acceleration
    from backfire_kernel import rust_sum_i64

    _RUST_COMPOSITE = True
except ImportError:  # pragma: no cover - bit-exact pure-Python fallback (ADR-0001)
    rust_sum_i64 = sum_i64

    _RUST_COMPOSITE = False


from ...mandatory import mandatory_execution
from .base import RECOMMENDED_RERANKER_MODEL, VectorBackend
from .fusion import fuse_results, validate_fusion_method

__all__ = ["HybridBackend", "RerankedBackend"]

logger = logging.getLogger("DirectorAI.VectorStore")


class _RerankerModel(Protocol):
    """Cross-encoder reranker contract used by ``RerankedBackend``."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return relevance scores for query/document pairs."""
        ...


class _BM25State:
    """BM25 term statistics shared between ``HybridBackend`` views.

    ``HybridBackend.with_fusion`` derives query views over the same
    index; sharing one state object keeps documents, term counts and
    the average document length coherent no matter which view adds.
    """

    __slots__ = ("df", "doc_tfs", "docs", "lock", "total_len")

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self.doc_tfs: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        self.total_len = 0
        self.lock = threading.Lock()


class HybridBackend(VectorBackend):
    """BM25 + dense retrieval with a pluggable fusion strategy.

    Wraps any VectorBackend. Maintains a parallel BM25 index over
    the same documents. At query time, runs both sparse (BM25) and
    dense (wrapped backend) retrieval, then fuses the runs with the
    configured method — see
    :mod:`director_ai.core.retrieval.vector_store.fusion` for the
    available strategies and their references.

    No external dependencies — uses a built-in BM25 implementation.

    Default fusion is weighted Reciprocal Rank Fusion:
    score(d) = w_sparse/(k + rank_sparse) + w_dense/(k + rank_dense)
    (Cormack, Clarke & Büttcher, SIGIR 2009; default k=60).
    """

    def __init__(
        self,
        base: VectorBackend,
        rrf_k: int = 60,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        fetch_multiplier: int = 3,
        fusion_method: str = "rrf",
    ) -> None:
        if base is None:
            raise ValueError("base backend is required")
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool):
            raise ValueError("rrf_k must be an integer")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        sparse_weight = _validate_non_negative_weight(sparse_weight, "sparse_weight")
        dense_weight = _validate_non_negative_weight(dense_weight, "dense_weight")
        if sparse_weight == 0.0 and dense_weight == 0.0:
            raise ValueError("at least one retrieval weight must be positive")
        if not isinstance(fetch_multiplier, int) or isinstance(fetch_multiplier, bool):
            raise ValueError("fetch_multiplier must be an integer")
        if fetch_multiplier < 1:
            raise ValueError("fetch_multiplier must be at least 1")
        self._base = base
        self._rrf_k = rrf_k
        self._sparse_w = sparse_weight
        self._dense_w = dense_weight
        self._fetch_mul = fetch_multiplier
        self._fusion = validate_fusion_method(fusion_method)
        self._bm25 = _BM25State()

    def with_fusion(
        self,
        fusion_method: str,
        *,
        rrf_k: int | None = None,
        sparse_weight: float | None = None,
        dense_weight: float | None = None,
    ) -> HybridBackend:
        """Return a view over the same base backend and BM25 index.

        The view shares document and term statistics with this
        backend — adding through either stays coherent — so fusion
        strategies can be compared without re-indexing the corpus.
        """
        view = HybridBackend(
            base=self._base,
            rrf_k=self._rrf_k if rrf_k is None else rrf_k,
            sparse_weight=self._sparse_w if sparse_weight is None else sparse_weight,
            dense_weight=self._dense_w if dense_weight is None else dense_weight,
            fetch_multiplier=self._fetch_mul,
            fusion_method=fusion_method,
        )
        view._bm25 = self._bm25
        return view

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re

        return re.findall(r"\w+", text.lower())

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index the document in the base store and the BM25 term table."""
        self._base.add(doc_id, text, metadata)
        tokens = self._tokenize(text)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        state = self._bm25
        with state.lock:
            state.docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})
            state.doc_tfs.append(tf)
            state.total_len += len(tokens)
            for term in set(tokens):
                state.df[term] = state.df.get(term, 0) + 1

    def _bm25_query(
        self,
        text: str,
        n_results: int,
        tenant_id: str,
    ) -> list[tuple[dict[str, Any], float]]:
        """BM25 scoring (k1=1.2, b=0.75) → (row, native score) pairs."""
        import math

        query_tokens = self._tokenize(text)
        if not query_tokens:
            return []

        state = self._bm25
        with state.lock:
            docs = list(state.docs)
            tfs = list(state.doc_tfs)
            df = dict(state.df)
            total_len = state.total_len

        n = len(docs)
        if n == 0:
            return []
        avgdl = total_len / n
        k1, b = 1.2, 0.75

        scores: list[tuple[float, int]] = []
        for i, (doc, tf) in enumerate(zip(docs, tfs, strict=False)):
            if tenant_id and doc["metadata"].get("tenant_id") != tenant_id:
                continue
            dl = _sum_int(list(tf.values()))
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
                results.append(({**docs[idx], "distance": 1.0 / (1.0 + score)}, score))
        return results

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Return the top ``n_results`` by fused dense and BM25 scores."""
        fetch_n = n_results * self._fetch_mul

        # Run both retrieval paths
        sparse_run = self._bm25_query(text, fetch_n, tenant_id)
        dense_rows = self._base.query(text, n_results=fetch_n, tenant_id=tenant_id)
        # Dense native score: every shipped backend emits distance = 1 − sim.
        dense_run = [(row, 1.0 - float(row.get("distance", 1.0))) for row in dense_rows]

        fused = fuse_results(
            self._fusion,
            sparse_run,
            dense_run,
            rrf_k=self._rrf_k,
            sparse_weight=self._sparse_w,
            dense_weight=self._dense_w,
        )
        return fused[:n_results]

    def count(self) -> int:
        """Return the number of documents in the underlying base store."""
        return self._base.count()


class RerankedBackend(VectorBackend):
    """Cross-encoder reranking wrapper around any VectorBackend.

    Retrieves ``top_k_multiplier * n_results`` candidates from the base
    backend, then reranks with a cross-encoder model and returns the
    top ``n_results``.

    Operators may inject a preloaded reranker object that implements
    ``predict(list[tuple[str, str]]) -> list[float]``. This keeps embedded,
    offline, or model-managed deployments on the same retrieval path without
    requiring ``RerankedBackend`` to own model loading.

    Requires ``pip install sentence-transformers>=4``.
    """

    def __init__(
        self,
        base: VectorBackend,
        reranker_model: str = RECOMMENDED_RERANKER_MODEL,
        reranker_revision: str | None = None,
        top_k_multiplier: int = 3,
        reranker: _RerankerModel | None = None,
    ) -> None:
        """Create a reranking wrapper around an existing vector backend."""
        if base is None:
            raise ValueError("base backend is required")
        if not isinstance(reranker_model, str) or not reranker_model.strip():
            raise ValueError("reranker_model must be a non-empty string")
        reranker_model = reranker_model.strip()
        if not isinstance(top_k_multiplier, int) or isinstance(top_k_multiplier, bool):
            raise ValueError("top_k_multiplier must be an integer")
        if top_k_multiplier < 1:
            raise ValueError("top_k_multiplier must be at least 1")
        self._base = base
        self._multiplier = top_k_multiplier
        if reranker is not None:
            if not callable(getattr(reranker, "predict", None)):
                raise ValueError("reranker must provide a callable predict method")
            self._reranker = reranker
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "RerankedBackend requires sentence-transformers. "
                "Install with: pip install director-ai[reranker]",
            ) from e
        from ..._device import release_torch_cuda, select_torch_device

        device = select_torch_device()
        try:
            self._reranker = CrossEncoder(
                reranker_model,
                device=device,
                revision=reranker_revision,
            )
        except RuntimeError as exc:
            if device == "cpu" or not _is_cuda_oom(exc):
                raise
            logger.warning(
                "CUDA out of memory while loading reranker %s; retrying on CPU",
                reranker_model,
            )
            release_torch_cuda()
            self._reranker = CrossEncoder(
                reranker_model,
                device="cpu",
                revision=reranker_revision,
            )

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index the document in the wrapped base store (no reranker state)."""
        self._base.add(doc_id, text, metadata)

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Over-fetch from the base store, then cross-encoder rerank to ``n_results``."""
        candidates = self._base.query(
            text,
            n_results=n_results * self._multiplier,
            tenant_id=tenant_id,
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
        """Return the number of documents in the underlying base store."""
        return self._base.count()


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "torch.outofmemoryerror" in msg


def _validate_non_negative_weight(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _sum_int(values: list[int]) -> int:
    if _RUST_COMPOSITE:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return int(rust_sum_i64(values))
    # kernel absent: use the bit-exact pure-Python fallback (ADR-0001)
    return sum_i64(values)
