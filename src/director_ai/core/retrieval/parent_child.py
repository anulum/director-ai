# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Parent-child chunking backend
"""Parent-child chunking: index small children, return large parents.

Small child chunks are indexed for **retrieval precision** — they match
narrow queries better than large passages. When a child matches, the
full parent chunk is returned for **context completeness**.

This is a ``VectorBackend`` decorator: it wraps any base backend and
intercepts ``add()`` to split text into parent → child chunks, and
``query()`` to deduplicate and return parent text.

Usage::

    from director_ai.core.retrieval.parent_child import ParentChildBackend
    from director_ai.core.retrieval.vector_store import ChromaBackend

    base = ChromaBackend()
    backend = ParentChildBackend(base, parent_size=2048, child_size=256)
    backend.add("doc1", long_document_text)
    results = backend.query("specific detail")
    # results[0]["text"] is the full parent chunk, not the small child
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from director_ai.core.retrieval.doc_chunker import ChunkConfig, split
from director_ai.core.retrieval.vector_store import VectorBackend

logger = logging.getLogger("DirectorAI.ParentChild")

__all__ = ["ParentChildBackend"]


def _chunk_id(doc_id: str, idx: int, level: str) -> str:
    """Deterministic chunk ID: ``doc1::parent-0``, ``doc1::child-0-2``."""
    return f"{doc_id}::{level}-{idx}"


class ParentChildBackend(VectorBackend):
    """Decorator: index child chunks, return parent chunks.

    Parameters
    ----------
    base : VectorBackend
        Underlying backend for actual storage and retrieval.
    parent_size : int
        Character size for parent chunks (returned on match).
    child_size : int
        Character size for child chunks (indexed for retrieval).
    parent_overlap : int
        Overlap between parent chunks.
    child_overlap : int
        Overlap between child chunks within a parent.
    """

    def __init__(
        self,
        base: VectorBackend,
        parent_size: int = 2048,
        child_size: int = 256,
        parent_overlap: int = 128,
        child_overlap: int = 32,
    ) -> None:
        self._base = base
        self._parent_size = parent_size
        self._child_size = child_size
        self._parent_overlap = parent_overlap
        self._child_overlap = child_overlap
        self._parents: dict[str, str] = {}  # parent_id → parent_text
        self._lock = threading.Lock()

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Split into parents, sub-split into children, index children."""
        meta = dict(metadata) if metadata else {}

        parent_cfg = ChunkConfig(
            chunk_size=self._parent_size,
            overlap=self._parent_overlap,
        )
        parents = split(text, parent_cfg)

        child_cfg = ChunkConfig(
            chunk_size=self._child_size,
            overlap=self._child_overlap,
        )

        with self._lock:
            for p_idx, parent_text in enumerate(parents):
                pid = _chunk_id(doc_id, p_idx, "parent")
                self._parents[pid] = parent_text

                children = split(parent_text, child_cfg)
                for c_idx, child_text in enumerate(children):
                    cid = _chunk_id(doc_id, c_idx, f"child-{p_idx}")
                    child_meta = {
                        **meta,
                        "parent_id": pid,
                        "child_index": c_idx,
                        "doc_id": doc_id,
                    }
                    self._base.add(cid, child_text, child_meta)

        logger.debug(
            "Added %s: %d parents, %d total children",
            doc_id,
            len(parents),
            sum(len(split(p, child_cfg)) for p in parents),
        )

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Retrieve children, deduplicate by parent, return parent text."""
        # Fetch more children than needed for deduplication headroom
        fetch_n = n_results * 5
        children = self._base.query(text, n_results=fetch_n, tenant_id=tenant_id)

        seen_parents: set[str] = set()
        results: list[dict[str, Any]] = []

        with self._lock:
            for child in children:
                child_meta = child.get("metadata", {})
                pid = child_meta.get("parent_id", "")

                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent_text = self._parents.get(pid)
                if parent_text is None:
                    # Fallback: return child text if parent not found
                    results.append(child)
                else:
                    results.append(
                        {
                            "id": pid,
                            "text": parent_text,
                            "distance": child["distance"],
                            "metadata": {
                                **child_meta,
                                "matched_child_id": child["id"],
                                "matched_child_text": child["text"],
                            },
                        }
                    )

                if len(results) >= n_results:
                    break

        return results

    def count(self) -> int:
        """Number of indexed children (not parents)."""
        return self._base.count()

    @property
    def parent_count(self) -> int:
        """Number of stored parent chunks."""
        with self._lock:
            return len(self._parents)
