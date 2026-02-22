# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Ground Truth Store (RAG Interface)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging
from typing import Any

#: Sample facts for demos and tests.  Not loaded by default — pass
#: ``facts=SAMPLE_FACTS`` to ``GroundTruthStore()`` when you want them.
SAMPLE_FACTS: dict[str, str] = {
    "sky color": "blue",
    "scpn layers": "16",
    "layer 1": "quantum biological",
    "layer 16": "director",
    "sec metric": "sustainable ethical coherence",
    "backfire limit": "entropy threshold",
    "vibrana symmetry": "13-fold",
}


class GroundTruthStore:
    """
    Retrieval-Augmented Generation (RAG) ground truth store.

    Holds a ``{key: value}`` dictionary of verified facts and retrieves
    matching entries via keyword overlap.

    Parameters
    ----------
    facts : dict[str, str] | None
        Initial fact dictionary.  Defaults to *empty*.  Pass
        ``SAMPLE_FACTS`` for the built-in demo dataset.
    """

    def __init__(self, facts: dict[str, Any] | None = None):
        self.logger = logging.getLogger("DirectorAI.GroundTruthStore")
        self.facts: dict[str, str] = dict(facts) if facts else {}

    def add(self, key: str, value: str) -> None:
        """Insert or update a single fact."""
        self.facts[key] = value

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
            self.logger.info(
                f"RAG Retrieval: Found context '{retrieved}' for query '{query}'"
            )
            return retrieved

        return None
