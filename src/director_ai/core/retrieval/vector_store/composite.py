# SPDX-License-Identifier: AGPL-3.0-or-later
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
from typing import Any

try:  # pragma: no cover - optional acceleration
    from backfire_kernel import rust_sum_i64

    _RUST_COMPOSITE = True
except ImportError:  # pragma: no cover - mandatory accelerator guard
    _RUST_COMPOSITE = True

    def rust_sum_i64(_values: list[int]) -> int:
        raise RuntimeError("backfire_kernel rust_sum_i64 is unavailable")


from ...mandatory import mandatory_execution
from .base import VectorBackend

__all__ = ["HybridBackend", "RerankedBackend"]

logger = logging.getLogger("DirectorAI.VectorStore")


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
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
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
        self,
        text: str,
        n_results: int,
        tenant_id: str,
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
                results.append({**docs[idx], "distance": 1.0 / (1.0 + score)})
        return results

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
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

    Requires ``pip install sentence-transformers>=4``.
    """

    def __init__(
        self,
        base: VectorBackend,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_revision: str | None = None,
        top_k_multiplier: int = 3,
    ) -> None:
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
        self._base.add(doc_id, text, metadata)

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
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
    return sum(values)
