# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Ground Truth Store (RAG Interface)

from __future__ import annotations

import hashlib
import logging

__all__ = ["GroundTruthStore"]


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

    def __init__(self):
        self.logger = logging.getLogger("DirectorAI.GroundTruthStore")
        self.facts = {}

    @classmethod
    def with_demo_facts(cls) -> GroundTruthStore:
        """Return a store pre-loaded with demo facts (for tests and --demo)."""
        store = cls()
        store.facts.update(cls._DEMO_FACTS)
        return store

    def add(self, key: str, value: str, tenant_id: str = "") -> None:
        """Add or update a fact in the store."""
        full_key = f"{tenant_id}:{key}" if tenant_id else key
        self.facts[full_key] = value

    def add_fact(self, key: str, value: str, tenant_id: str = "") -> None:
        """Alias for add() — used by some callers."""
        self.add(key, value, tenant_id=tenant_id)

    def retrieve_context_with_chunks(
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "",
    ) -> list:
        from ..types import EvidenceChunk

        context_str = self.retrieve_context(query, tenant_id=tenant_id)
        if not context_str:
            return []
        return [EvidenceChunk(text=context_str, distance=0.0, source="keyword")]

    def retrieve_context(
        self, query: str, tenant_id: str = "", top_k: int = 0
    ) -> str | None:
        """Retrieve relevant facts matching *query*.

        Returns a semicolon-separated context string, or ``None`` if
        no relevant facts are found.
        """
        if not self.facts:
            self.logger.info(
                "GroundTruthStore is empty — add facts via .add() "
                "or use VectorGroundTruthStore.ingest()",
            )
            return None

        query_lower = query.lower()
        context = []

        for key, value in self.facts.items():
            search_key = key
            if tenant_id:
                prefix = f"{tenant_id}:"
                if not key.startswith(prefix):
                    continue
                search_key = key[len(prefix) :]
            key_words = search_key.lower().split()
            if any(word in query_lower for word in key_words):
                context.append(value)

        if context:
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
