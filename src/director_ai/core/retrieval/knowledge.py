# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Ground Truth Store (RAG Interface)

"""Ground-truth knowledge store: the RAG interface backing grounded answers."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from ..text_overlap import word_overlap

if TYPE_CHECKING:
    from ..types import EvidenceChunk

__all__ = ["GroundTruthStore"]

# Common English function words filtered before value-content matching, so a
# query never matches a fact on stop words alone. Deliberately small and
# recall-first — this only gates the degraded keyword fallback, not the primary
# vector path.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "did",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "many",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)


class GroundTruthStore:
    """In-memory fact store with keyword matching retrieval.

    Stores keyâ†’value fact pairs and retrieves by word overlap between
    the query and fact keys. No embeddings or semantic similarity.

    For vector-based retrieval, use ``VectorGroundTruthStore`` with a
    ``VectorBackend`` (Chroma, Pinecone, Qdrant, FAISS, etc.).
    """

    _DEMO_FACTS = {
        "sky color": "blue",
        "scpn layers": "16",
        "layer 1": "quantum biological",
        "layer 16": "director",
        "sec metric": "sustainable ethical coherence",
        "backfire limit": "entropy threshold",
        "vibrana symmetry": "13-fold",
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger("DirectorAI.GroundTruthStore")
        self.facts: dict[str, str] = {}
        self._revision = 0

    @classmethod
    def with_demo_facts(cls) -> GroundTruthStore:
        """Return a store pre-loaded with demo facts (for tests and --demo)."""
        store = cls()
        store.facts.update(cls._DEMO_FACTS)
        return store

    def add(
        self,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "",
    ) -> None:
        """Add or update a fact in the store.

        ``metadata`` is accepted for Liskov compatibility with
        :class:`VectorGroundTruthStore.add` but ignored here — the
        keyword store keeps no structured metadata.
        """
        _ = metadata  # intentional no-op for LSP compat
        key = _require_non_empty_string("key", key)
        value = _require_non_empty_string("value", value)
        tenant_id = tenant_id.strip()
        full_key = f"{tenant_id}:{key}" if tenant_id else key
        self.facts[full_key] = value
        self._bump_revision()

    def add_fact(self, key: str, value: str, tenant_id: str = "") -> None:
        """Alias for add() — used by some callers."""
        self.add(key, value, tenant_id=tenant_id)

    @property
    def revision(self) -> int:
        """Monotonic store revision used to scope score-cache entries."""
        return self._revision

    def _bump_revision(self) -> None:
        self._revision += 1

    def cache_scope(self, tenant_id: str = "") -> str:
        """Return a stable cache scope for facts visible to a tenant."""
        return f"{type(self).__name__}:tenant={tenant_id}:revision={self._revision}"

    def retrieve_context_with_chunks(
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "",
    ) -> list[EvidenceChunk]:
        """Return the keyword-retrieved context wrapped as evidence chunks."""
        from ..types import EvidenceChunk

        context_str = self.retrieve_context(query, tenant_id=tenant_id)
        if not context_str:
            return []
        return [EvidenceChunk(text=context_str, distance=0.0, source="keyword")]

    def retrieve_context(
        self, query: str, top_k: int = 0, tenant_id: str = ""
    ) -> str | None:
        """Retrieve relevant facts matching *query*.

        Returns a semicolon-separated context string, or ``None`` if
        no relevant facts are found.
        """
        query = _require_non_empty_string("query", query)
        tenant_id = tenant_id.strip()
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0; got {top_k!r}")
        if not self.facts:
            self.logger.info(
                "GroundTruthStore is empty — add facts via .add() "
                "or use VectorGroundTruthStore.ingest()",
            )
            return None

        query_lower = query.lower()
        query_tokens = {w for w in query_lower.split() if w not in _STOPWORDS}
        ranked: list[tuple[float, str]] = []

        for key, value in self.facts.items():
            search_key = key
            if tenant_id:
                prefix = f"{tenant_id}:"
                if not key.startswith(prefix):
                    continue
                search_key = key[len(prefix) :]
            key_words = search_key.lower().split()
            if any(word in query_lower for word in key_words):
                # Curated, semantically-keyed fact: rank on the key (unchanged).
                ranked.append((_word_overlap(query, search_key), value))
                continue
            # Value-aware fallback for content-keyed facts — e.g. API-ingested
            # chunks keyed by an opaque cid ("doc:chunk:0"), which has no query
            # words, so key matching never reaches the chunk text. Match the
            # value, stopword-filtered so a query cannot match on function words
            # alone, and rank below key hits so curated facts always win.
            value_tokens = {w for w in value.lower().split() if w not in _STOPWORDS}
            if query_tokens & value_tokens:
                ranked.append((0.5 * _word_overlap(query, value), value))

        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            context = [value for _, value in ranked]
            if top_k > 0:
                context = context[:top_k]
            retrieved = "; ".join(context)
            qhash = hashlib.sha256(query.encode()).hexdigest()[:12]
            self.logger.info(
                "RAG Retrieval: %d facts matched (query=%s, len=%d)",
                len(context),
                qhash,
                len(query),
            )
            return retrieved

        return None


def _require_non_empty_string(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _word_overlap(text_a: str, text_b: str) -> float:
    """Return lexical Jaccard overlap in ``[0, 1]``.

    Delegates to the shared measured-fast-path helper (pure Python below a large
    -input threshold, Rust above it). See :mod:`director_ai.core.text_overlap`.
    """
    return word_overlap(text_a, text_b, logger_name=__name__)
