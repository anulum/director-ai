# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Ground Truth Store (RAG Interface)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging

__all__ = ["GroundTruthStore"]


class GroundTruthStore:
    """
    Retrieval-Augmented Generation (RAG) ground truth store.

    In production this connects to a vector database (Pinecone, Milvus,
    etc.) holding verified facts.  The prototype uses an in-memory dict
    for deterministic testing.
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
    def with_demo_facts(cls) -> "GroundTruthStore":
        """Return a store pre-loaded with demo facts (for tests and --demo)."""
        store = cls()
        store.facts.update(cls._DEMO_FACTS)
        return store

    async def add(self, key: str, value: str, tenant_id: str = "") -> None:
        """Add or update a fact in the store."""
        full_key = f"{tenant_id}:{key}" if tenant_id else key
        self.facts[full_key] = value

    async def retrieve_context(self, query: str, tenant_id: str = "") -> str | None:
        """
        Retrieve relevant facts matching *query*.

        Returns a semicolon-separated context string, or ``None`` if
        no relevant facts are found.
        """
        if not self.facts:
            self.logger.info(
                "GroundTruthStore is empty — add facts via .add() "
                "or use VectorGroundTruthStore.ingest()"
            )
            return None

        query_lower = query.lower()
        context = []

        for key, value in self.facts.items():
            if tenant_id and not key.startswith(f"{tenant_id}:"):
                continue
            key_words = key.lower().split()
            if any(word in query_lower for word in key_words):
                context.append(value)

        if context:
            retrieved = "; ".join(context)
            self.logger.info(
                f"RAG Retrieval: Found context '{retrieved}' for query '{query}'"
            )
            return retrieved

        return None
