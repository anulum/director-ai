# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Multi-vector retrieval backend
"""Multi-vector retrieval: store multiple representations per document.

Each document is indexed under multiple embedding representations:
- **content**: the full text (default)
- **summary**: a condensed version (first N sentences or LLM summary)
- **title**: extracted or generated title line

Queries match against all representations. Results are deduplicated
by document ID and ranked by best-match score across representations.

Usage::

    from director_ai.core.retrieval.multi_vector import MultiVectorBackend

    base = ChromaBackend()
    backend = MultiVectorBackend(base)
    backend.add("doc1", "Long document text about refund policies...")
    results = backend.query("refund")
    # Matches against content + summary + title representations
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from typing import Any

from director_ai.core.retrieval.vector_store import VectorBackend

logger = logging.getLogger("DirectorAI.MultiVector")

__all__ = ["MultiVectorBackend"]

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _extract_title(text: str) -> str:
    """Heuristic title extraction: first sentence or first line."""
    first_line = text.split("\n")[0].strip()
    if len(first_line) > 10:
        return first_line[:200]
    sentences = _SENT_RE.split(text)
    return sentences[0][:200] if sentences else text[:200]


def _extract_summary(text: str, max_sentences: int = 3) -> str:
    """Heuristic summary: first N sentences."""
    sentences = _SENT_RE.split(text)
    return " ".join(sentences[:max_sentences])


class MultiVectorBackend(VectorBackend):
    """Decorator: index multiple representations, query across all.

    Parameters
    ----------
    base : VectorBackend
        Underlying backend for storage and retrieval.
    representations : list[str]
        Which representations to generate. Default: all three.
        Options: ``"content"``, ``"summary"``, ``"title"``.
    summary_generator : Callable[[str], str] | None
        Optional LLM-based summary generator. If None, uses
        heuristic (first N sentences).
    summary_sentences : int
        Number of sentences for heuristic summary.
    """

    def __init__(
        self,
        base: VectorBackend,
        representations: list[str] | None = None,
        summary_generator: Callable[[str], str] | None = None,
        summary_sentences: int = 3,
    ) -> None:
        self._base = base
        self._representations = representations or ["content", "summary", "title"]
        self._summary_generator = summary_generator
        self._summary_sentences = summary_sentences
        self._doc_texts: dict[str, str] = {}  # doc_id → original text
        self._lock = threading.Lock()

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index document under multiple representations."""
        meta = dict(metadata) if metadata else {}

        with self._lock:
            self._doc_texts[doc_id] = text

        reps = self._generate_representations(text)
        for rep_name, rep_text in reps.items():
            rep_id = f"{doc_id}::{rep_name}"
            rep_meta = {
                **meta,
                "doc_id": doc_id,
                "representation": rep_name,
            }
            self._base.add(rep_id, rep_text, rep_meta)

        logger.debug(
            "Added %s with %d representations",
            doc_id,
            len(reps),
        )

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Query across all representations, deduplicate by doc_id."""
        # Fetch more to account for deduplication
        fetch_n = n_results * len(self._representations) * 2
        raw = self._base.query(text, n_results=fetch_n, tenant_id=tenant_id)

        # Deduplicate: keep best score per doc_id
        best: dict[str, dict[str, Any]] = {}
        for r in raw:
            meta = r.get("metadata", {})
            doc_id = meta.get("doc_id", r.get("id", ""))
            dist = r.get("distance", 1.0)

            if doc_id not in best or dist < best[doc_id].get("distance", 1.0):
                # Replace text with original content
                with self._lock:
                    original = self._doc_texts.get(doc_id)
                if original is not None:
                    r = {**r, "text": original}
                best[doc_id] = r

        # Sort by distance (ascending = most relevant first)
        ranked = sorted(best.values(), key=lambda x: x.get("distance", 1.0))
        return ranked[:n_results]

    def count(self) -> int:
        """Number of indexed representations (not unique documents)."""
        return self._base.count()

    @property
    def document_count(self) -> int:
        """Number of unique documents."""
        with self._lock:
            return len(self._doc_texts)

    def _generate_representations(self, text: str) -> dict[str, str]:
        """Generate configured representations for a document."""
        reps: dict[str, str] = {}

        if "content" in self._representations:
            reps["content"] = text

        if "summary" in self._representations:
            if self._summary_generator is not None:
                try:
                    reps["summary"] = self._summary_generator(text)
                except Exception as exc:
                    logger.warning("Summary generation failed: %s", exc)
                    reps["summary"] = _extract_summary(text, self._summary_sentences)
            else:
                reps["summary"] = _extract_summary(text, self._summary_sentences)

        if "title" in self._representations:
            reps["title"] = _extract_title(text)

        return reps
