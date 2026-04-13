# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HyDE (Hypothetical Document Embeddings) backend
"""HyDE retrieval: embed a pseudo-answer instead of the raw query.

Instead of embedding the user's question directly, an LLM generates a
hypothetical answer (pseudo-document), which is then embedded and used
for dense retrieval. This bridges the lexical gap between questions
and answers, improving recall for abstract or indirect queries.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (2022). arXiv:2212.10496.

This is a ``VectorBackend`` decorator: it wraps any base backend and
intercepts ``query()`` to generate a pseudo-document before retrieval.
``add()`` delegates directly — documents are indexed normally.

Usage::

    from director_ai.core.retrieval.hyde import HyDEBackend

    def my_llm(prompt: str) -> str:
        return openai.chat.completions.create(...).choices[0].message.content

    base = ChromaBackend()
    backend = HyDEBackend(base, generator=my_llm)
    backend.add("doc1", "Paris is the capital of France.")
    results = backend.query("What is the capital of France?")
    # Internally: LLM generates "The capital of France is Paris."
    # → embeds that → retrieves from Chroma
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from director_ai.core.retrieval.vector_store import VectorBackend

logger = logging.getLogger("DirectorAI.HyDE")

__all__ = ["HyDEBackend"]

_DEFAULT_TEMPLATE = (
    "Write a short factual passage that answers the following question.\n"
    "Question: {query}\n"
    "Passage:"
)


class HyDEBackend(VectorBackend):
    """Decorator: generate pseudo-document before retrieval.

    Parameters
    ----------
    base : VectorBackend
        Underlying backend for storage and retrieval.
    generator : Callable[[str], str]
        Function that takes a prompt string and returns an LLM response.
        Can be any LLM client wrapper (OpenAI, Anthropic, local, etc.).
    template : str
        Prompt template with ``{query}`` placeholder. The LLM generates
        a hypothetical answer for this prompt.
    fallback_to_raw : bool
        If True (default), falls back to raw query when LLM generation
        fails. If False, raises the exception.
    cache_ttl : float
        Seconds to cache pseudo-documents for identical queries.
        0 disables caching.
    """

    def __init__(
        self,
        base: VectorBackend,
        generator: Callable[[str], str] | None = None,
        template: str = _DEFAULT_TEMPLATE,
        fallback_to_raw: bool = True,
        cache_ttl: float = 300.0,
    ) -> None:
        self._base = base
        self._generator = generator
        self._template = template
        self._fallback_to_raw = fallback_to_raw
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[str, float]] = {}

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
        """Generate pseudo-document, then retrieve using it."""
        pseudo_doc = self._generate_pseudo_doc(text)
        results = self._base.query(pseudo_doc, n_results=n_results, tenant_id=tenant_id)

        # Annotate results with HyDE metadata
        for r in results:
            meta = r.get("metadata", {})
            meta["hyde_pseudo_doc"] = pseudo_doc[:200]  # truncate for audit
            meta["hyde_original_query"] = text[:200]
            r["metadata"] = meta

        return results

    def count(self) -> int:
        """Delegate to base."""
        return self._base.count()

    def _generate_pseudo_doc(self, query: str) -> str:
        """Generate a hypothetical document for the query.

        Uses the configured LLM generator. Falls back to raw query
        on failure when ``fallback_to_raw=True``.
        """
        if self._generator is None:
            return query  # No LLM → use raw query (graceful degradation)

        # Check cache
        now = time.monotonic()
        if self._cache_ttl > 0 and query in self._cache:
            cached_doc, cached_at = self._cache[query]
            if now - cached_at < self._cache_ttl:
                return cached_doc

        prompt = self._template.format(query=query)
        try:
            pseudo_doc = self._generator(prompt)
            if not pseudo_doc or not pseudo_doc.strip():
                logger.warning("HyDE generator returned empty response")
                return query

            # Cache the result
            if self._cache_ttl > 0:
                self._cache[query] = (pseudo_doc.strip(), now)
                # Evict stale entries (simple, not LRU)
                if len(self._cache) > 1000:
                    self._cache = {
                        k: (v, t)
                        for k, (v, t) in self._cache.items()
                        if now - t < self._cache_ttl
                    }

            logger.debug(
                "HyDE: query=%r → pseudo_doc=%r",
                query[:80],
                pseudo_doc[:80],
            )
            return pseudo_doc.strip()

        except Exception as exc:
            if self._fallback_to_raw:
                logger.warning("HyDE generation failed, using raw query: %s", exc)
                return query
            raise
