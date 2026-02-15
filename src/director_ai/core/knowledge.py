# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Ground Truth Store (RAG Interface)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging


class GroundTruthStore:
    """
    Retrieval-Augmented Generation (RAG) ground truth store.

    In production this connects to a vector database (Pinecone, Milvus,
    etc.) holding verified facts.  The prototype uses an in-memory dict
    for deterministic testing.
    """

    def __init__(self):
        self.logger = logging.getLogger("DirectorAI.GroundTruthStore")
        # Mock ground truth database
        self.facts = {
            "sky color": "blue",
            "scpn layers": "16",
            "layer 1": "quantum biological",
            "layer 16": "director",
            "sec metric": "sustainable ethical coherence",
            "backfire limit": "entropy threshold",
            "vibrana symmetry": "13-fold",
        }

    def retrieve_context(self, query):
        """
        Retrieve relevant facts matching *query*.

        Returns a semicolon-separated context string, or ``None`` if
        no relevant facts are found.
        """
        query_lower = query.lower()
        context = []

        for key, value in self.facts.items():
            key_words = key.split()
            if any(word in query_lower for word in key_words):
                context.append(f"{key} is {value}")

        if context:
            retrieved = "; ".join(context)
            self.logger.info(f"RAG Retrieval: Found context '{retrieved}' for query '{query}'")
            return retrieved

        return None
