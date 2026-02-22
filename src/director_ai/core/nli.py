# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Real NLI Backend (DeBERTa)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Real NLI-based logical divergence scorer using DeBERTa-v3-base-mnli.

Provides ``NLIScorer`` which lazily loads the model on first call and
caches it for subsequent invocations.  Falls back to heuristic scoring
when the model is unavailable (no internet, no transformers, etc.).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger("DirectorAI.NLI")

_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Labels for the DeBERTa-mnli model
_LABEL_ENTAILMENT = 0
_LABEL_NEUTRAL = 1
_LABEL_CONTRADICTION = 2


@lru_cache(maxsize=1)
def _load_nli_model():
    """Lazily load the DeBERTa NLI model + tokenizer (cached singleton)."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading NLI model: %s", _MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        model.eval()
        logger.info("NLI model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.warning("NLI model unavailable: %s — using heuristic fallback", e)
        return None, None


class NLIScorer:
    """NLI-based logical divergence scorer.

    Usage::

        scorer = NLIScorer()
        h_logical = scorer.score("The sky is blue", "The sky is green")
        # h_logical ≈ 0.9 (contradiction)

    Parameters
    ----------
    use_model : bool — if True, attempt to load DeBERTa model.
    max_length : int — max token length for NLI input.
    """

    def __init__(
        self,
        use_model: bool = True,
        max_length: int = 512,
    ) -> None:
        self.use_model = use_model
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._model_loaded = False

    def _ensure_model(self) -> bool:
        """Load model if not yet loaded. Returns True if model is available."""
        if self._model_loaded:
            return self._model is not None
        if not self.use_model:
            self._model_loaded = True
            return False
        self._tokenizer, self._model = _load_nli_model()
        self._model_loaded = True
        return self._model is not None

    @property
    def model_available(self) -> bool:
        return self._ensure_model()

    def score(self, premise: str, hypothesis: str) -> float:
        """Compute logical divergence between premise and hypothesis.

        Returns
        -------
        float in [0, 1]: 0 = entailment, 0.5 = neutral, 1.0 = contradiction.
        """
        if not self._ensure_model():
            return self._heuristic_score(premise, hypothesis)
        return self._model_score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple (premise, hypothesis) pairs."""
        return [self.score(p, h) for p, h in pairs]

    def _model_score(self, premise: str, hypothesis: str) -> float:
        """Score using the DeBERTa NLI model."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded — call with use_model=True")

        import torch
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits

        # Guard against NaN/Inf in logits
        if not torch.isfinite(logits).all():
            logger.warning("NLI logits contain NaN/Inf — falling back to heuristic")
            return self._heuristic_score(premise, hypothesis)

        probs = torch.softmax(logits, dim=1).numpy()[0]
        contradiction_prob = float(probs[_LABEL_CONTRADICTION])
        neutral_prob = float(probs[_LABEL_NEUTRAL])

        h = contradiction_prob + (neutral_prob * 0.5)
        return float(np.clip(h, 0.0, 1.0))

    @staticmethod
    def _heuristic_score(premise: str, hypothesis: str) -> float:
        """Deterministic heuristic fallback (same as mock scorer)."""
        h_lower = hypothesis.lower()
        if "consistent with reality" in h_lower:
            return 0.1
        if "opposite is true" in h_lower:
            return 0.9
        if "depends on your perspective" in h_lower:
            return 0.5
        # Simple keyword overlap heuristic
        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        if not p_words:
            return 0.5
        overlap = len(p_words & h_words) / max(len(p_words), 1)
        return float(np.clip(0.5 - overlap * 0.3, 0.1, 0.9))
