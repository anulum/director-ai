# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Contextual compression backend
"""Contextual compression: keep only query-relevant parts of passages.

After retrieval, an LLM (or heuristic) compresses each result to
retain only the parts relevant to the original query. This reduces
noise in the context window and improves downstream NLI scoring.

Two compression strategies:
- ``"heuristic"`` (default): sentence-level filtering via keyword
  overlap with the query. Zero LLM calls, <1ms overhead.
- ``"llm"``: LLM-based extraction of relevant sentences.

Usage::

    from director_ai.core.retrieval.contextual_compression import (
        ContextualCompressionBackend,
    )

    base = ChromaBackend()
    backend = ContextualCompressionBackend(base)
    results = backend.query("refund policy")
    # Each result["text"] is compressed to query-relevant sentences
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from director_ai.core.retrieval.vector_store import VectorBackend

logger = logging.getLogger("DirectorAI.Compression")

__all__ = ["ContextualCompressionBackend"]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _keyword_overlap(query: str, sentence: str) -> float:
    """Jaccard word overlap between query and sentence."""
    q_words = set(query.lower().split())
    s_words = set(sentence.lower().split())
    if not q_words or not s_words:
        return 0.0
    return len(q_words & s_words) / len(q_words | s_words)


def _heuristic_compress(query: str, text: str, threshold: float) -> str:
    """Keep sentences with sufficient keyword overlap with query."""
    sentences = _SENT_SPLIT.split(text)
    if len(sentences) <= 1:
        return text  # single sentence — keep as-is

    kept = [s for s in sentences if _keyword_overlap(query, s) >= threshold]
    if not kept:
        # No sentence passes threshold — return highest-overlap sentence
        best = max(sentences, key=lambda s: _keyword_overlap(query, s))
        return best
    return " ".join(kept)


class ContextualCompressionBackend(VectorBackend):
    """Decorator: compress retrieved passages to query-relevant parts.

    Parameters
    ----------
    base : VectorBackend
        Underlying backend for storage and retrieval.
    strategy : str
        ``"heuristic"`` (default) or ``"llm"``.
    generator : Callable[[str], str] | None
        LLM callable for ``"llm"`` strategy.
    overlap_threshold : float
        Minimum keyword overlap for heuristic sentence filtering.
    min_compressed_len : int
        Minimum output length. If compression reduces below this,
        return the original text.
    """

    def __init__(
        self,
        base: VectorBackend,
        strategy: str = "heuristic",
        generator: Callable[[str], str] | None = None,
        overlap_threshold: float = 0.1,
        min_compressed_len: int = 20,
    ) -> None:
        self._base = base
        self._strategy = strategy
        self._generator = generator
        self._overlap_threshold = overlap_threshold
        self._min_compressed_len = min_compressed_len

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to base."""
        self._base.add(doc_id, text, metadata)

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Retrieve, then compress each result."""
        results = self._base.query(text, n_results=n_results, tenant_id=tenant_id)

        compressed = []
        for r in results:
            original = r.get("text", "")
            short = self._compress(text, original)

            # Preserve original in metadata for audit trail
            meta = dict(r.get("metadata", {}))
            if short != original:
                meta["original_text"] = original[:500]
                meta["compression_ratio"] = (
                    len(short) / len(original) if original else 1.0
                )

            compressed.append({**r, "text": short, "metadata": meta})

        return compressed

    def count(self) -> int:
        """Delegate to base."""
        return self._base.count()

    def _compress(self, query: str, text: str) -> str:
        """Compress text using configured strategy."""
        if self._strategy == "llm" and self._generator is not None:
            return self._llm_compress(query, text)
        result = _heuristic_compress(query, text, self._overlap_threshold)
        if len(result) < self._min_compressed_len:
            return text  # compression too aggressive → keep original
        return result

    def _llm_compress(self, query: str, text: str) -> str:
        """Use LLM to extract query-relevant sentences."""
        assert self._generator is not None
        prompt = (
            "Extract only the sentences from the passage that are "
            "relevant to answering the question. Return just those "
            "sentences, nothing else.\n\n"
            f"Question: {query}\n\n"
            f"Passage: {text}\n\n"
            "Relevant sentences:"
        )
        try:
            result = self._generator(prompt)
            if result and len(result.strip()) >= self._min_compressed_len:
                return result.strip()
            return text
        except Exception as exc:
            logger.warning("LLM compression failed: %s", exc)
            return _heuristic_compress(query, text, self._overlap_threshold)
