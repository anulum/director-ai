# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Embedding-based scorer backend
"""Embedding similarity scorer using sentence-transformers.

Encodes premise and hypothesis into dense vectors and returns their
cosine similarity as a groundedness score. Dramatically better than
word-overlap heuristics (~65-68% BA on AggreFact vs ~55%) at 3ms
latency on CPU.

This is Tier 3 in the 5-tier scoring pyramid — between rule-based
(Tier 2, <1ms, no ML) and full NLI (Tier 5, 14.6ms, 0.4B params).

Install::

    pip install director-ai[embed]

Usage::

    from director_ai.core.scoring.embed_scorer import EmbedBackend

    backend = EmbedBackend()  # loads bge-small-en-v1.5 (33M, 67 MB)
    score = backend.score("Water boils at 100°C.", "Water boils at 500°C.")
    # score ≈ 0.85 (high similarity — embedding catches topic but not factual error)
"""

from __future__ import annotations

import logging

logger = logging.getLogger("DirectorAI.EmbedScorer")

# Default model: fast, small, good quality for similarity
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class EmbedBackend:
    """Embedding cosine-similarity scorer.

    Implements the same interface as ``ScorerBackend`` from
    ``backends.py``. Lazy-loads the sentence-transformer model on
    first call to avoid import-time overhead.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for sentence-transformers.
    device : str
        ``"cpu"`` or ``"cuda"``. Default ``"cpu"`` — the small models
        are fast enough on CPU.
    cache_dir : str | None
        Cache directory for downloaded model weights.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model = None  # lazy

    def _ensure_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "EmbedBackend requires sentence-transformers. "
                "Install with: pip install director-ai[embed]"
            ) from exc

        logger.info(
            "Loading embedding model: %s (device=%s)",
            self._model_name,
            self._device,
        )
        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
            cache_folder=self._cache_dir,
        )
        logger.info("Embedding model loaded")

    def score(self, premise: str, hypothesis: str) -> float:
        """Cosine similarity between premise and hypothesis embeddings.

        Returns a value in [0, 1] where 1 = identical meaning and
        0 = completely unrelated. Note: embedding similarity captures
        *topic* similarity, not factual correctness — "boils at 100°C"
        and "boils at 500°C" will score high because they share topic.
        For factual checking, use NLI (Tier 5).
        """
        self._ensure_model()
        embeddings = self._model.encode(
            [premise, hypothesis],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarity = float(embeddings[0] @ embeddings[1])
        # Clamp to [0, 1] — cosine with normalised vecs is already in [-1, 1]
        return max(0.0, min(1.0, similarity))

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch scoring with efficient encoding.

        Encodes all premises and hypotheses in two batches, then
        computes pairwise cosine similarities.
        """
        if not pairs:
            return []
        self._ensure_model()
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        # Batch encode both sets
        p_embs = self._model.encode(
            premises,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        h_embs = self._model.encode(
            hypotheses,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        # Pairwise cosine (dot product of normalised vectors)
        scores = []
        for i in range(len(pairs)):
            sim = float(p_embs[i] @ h_embs[i])
            scores.append(max(0.0, min(1.0, sim)))
        return scores
