# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Query decomposition backend
"""Query decomposition: split compound queries, retrieve for each, merge.

Complex queries like "What is the refund policy and how long does
shipping take?" contain multiple information needs. This backend
decomposes them into sub-queries, retrieves for each independently,
and merges results via Reciprocal Rank Fusion (RRF).

Two decomposition strategies:
- ``"heuristic"`` (default): split on conjunctions, semicolons,
  question marks. Zero LLM calls, <0.1ms overhead.
- ``"llm"``: use a provided LLM to generate focused sub-queries.
  Higher quality but adds LLM latency.

Usage::

    from director_ai.core.retrieval.query_decomposition import (
        QueryDecompositionBackend,
    )

    base = ChromaBackend()
    backend = QueryDecompositionBackend(base)
    results = backend.query("What is the refund policy and shipping time?")
    # Internally: ["What is the refund policy", "shipping time"] → 2 retrievals → RRF merge
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from director_ai.core.retrieval.vector_store import VectorBackend

logger = logging.getLogger("DirectorAI.QueryDecomp")

__all__ = ["QueryDecompositionBackend"]

# Heuristic split patterns: conjunctions, semicolons, numbered lists
_SPLIT_PATTERN = re.compile(
    r"\s*(?:"
    r"\band\b|\bor\b|\balso\b|\badditionally\b|"
    r"[;]\s*|"
    r"[?]\s+(?=[A-Z])|"  # question mark followed by capital (new question)
    r"\d+[.)]\s"  # numbered lists: "1. ", "2) "
    r")\s*",
    re.IGNORECASE,
)

# Minimum sub-query length to avoid degenerate splits
_MIN_SUBQUERY_LEN = 8


def _heuristic_decompose(query: str) -> list[str]:
    """Split a compound query into sub-queries using patterns.

    Returns a list of sub-queries. Single-intent queries return
    ``[query]`` unchanged.
    """
    parts = _SPLIT_PATTERN.split(query)
    sub_queries = [p.strip().rstrip("?").strip() for p in parts if p and p.strip()]
    # Filter degenerate fragments
    sub_queries = [q for q in sub_queries if len(q) >= _MIN_SUBQUERY_LEN]
    return sub_queries if sub_queries else [query]


class QueryDecompositionBackend(VectorBackend):
    """Decorator: decompose compound queries, retrieve for each, merge.

    Parameters
    ----------
    base : VectorBackend
        Underlying backend for storage and retrieval.
    strategy : str
        ``"heuristic"`` (default) or ``"llm"``.
    generator : Callable[[str], str] | None
        LLM callable for ``"llm"`` strategy. Ignored for heuristic.
    rrf_k : int
        RRF parameter (default 60). Higher = smoother fusion.
    max_sub_queries : int
        Maximum sub-queries to generate (caps LLM verbosity).
    """

    def __init__(
        self,
        base: VectorBackend,
        strategy: str = "heuristic",
        generator: Callable[[str], str] | None = None,
        rrf_k: int = 60,
        max_sub_queries: int = 5,
    ) -> None:
        self._base = base
        self._strategy = strategy
        self._generator = generator
        self._rrf_k = rrf_k
        self._max_sub_queries = max_sub_queries

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to base — documents are indexed normally."""
        self._base.add(doc_id, text, metadata)

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Decompose query, retrieve for each sub-query, merge via RRF."""
        sub_queries = self._decompose(text)

        if len(sub_queries) <= 1:
            # Single intent — pass through directly
            return self._base.query(text, n_results=n_results, tenant_id=tenant_id)

        logger.debug("Decomposed %r into %d sub-queries", text[:60], len(sub_queries))

        # Retrieve for each sub-query
        all_results: list[list[dict[str, Any]]] = []
        fetch_per = max(n_results, 3)  # minimum 3 per sub-query
        for sq in sub_queries[: self._max_sub_queries]:
            results = self._base.query(sq, n_results=fetch_per, tenant_id=tenant_id)
            all_results.append(results)

        # RRF merge
        return self._rrf_merge(all_results, n_results)

    def count(self) -> int:
        """Delegate to base."""
        return self._base.count()

    def _decompose(self, query: str) -> list[str]:
        """Decompose query using configured strategy."""
        if self._strategy == "llm" and self._generator is not None:
            return self._llm_decompose(query)
        return _heuristic_decompose(query)

    def _llm_decompose(self, query: str) -> list[str]:
        """Use LLM to generate focused sub-queries."""
        assert self._generator is not None
        prompt = (
            "Break the following question into independent sub-questions.\n"
            "Return one sub-question per line. Do not number them.\n"
            f"Question: {query}\n"
            "Sub-questions:"
        )
        try:
            response = self._generator(prompt)
            lines = [
                ln.strip().lstrip("- ").strip()
                for ln in response.strip().split("\n")
                if ln.strip() and len(ln.strip()) >= _MIN_SUBQUERY_LEN
            ]
            return lines[: self._max_sub_queries] if lines else [query]
        except Exception as exc:
            logger.warning("LLM decomposition failed, using heuristic: %s", exc)
            return _heuristic_decompose(query)

    def _rrf_merge(
        self,
        result_lists: list[list[dict[str, Any]]],
        n_results: int,
    ) -> list[dict[str, Any]]:
        """Merge multiple result lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc.get("id", "")
                rrf_score = 1.0 / (self._rrf_k + rank + 1)
                scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Sort by fused score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in ranked[:n_results]]
