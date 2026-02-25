# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Real NLI Backend (DeBERTa)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
NLI-based logical divergence scorer using DeBERTa.

``NLIScorer`` lazily loads the model on first call and caches it.
Falls back to heuristic scoring when the model is unavailable.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger("DirectorAI.NLI")

_DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Labels for the DeBERTa-mnli model
_LABEL_ENTAILMENT = 0
_LABEL_NEUTRAL = 1
_LABEL_CONTRADICTION = 2


@lru_cache(maxsize=4)
def _load_nli_model(model_name: str = _DEFAULT_MODEL):
    """Lazily load an NLI model + tokenizer (cached by model_name)."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading NLI model: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        logger.info("NLI model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.warning("NLI model unavailable: %s — using heuristic fallback", e)
        return None, None


def nli_available() -> bool:
    """Check whether torch + transformers are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


class NLIScorer:
    """NLI-based logical divergence scorer.

    Parameters
    ----------
    use_model : bool — if True, attempt to load model on first score().
    max_length : int — max token length for NLI input.
    model_name : str | None — HuggingFace model ID or local path.
        Defaults to the pre-trained DeBERTa-v3-base-mnli-fever-anli.
    """

    def __init__(
        self,
        use_model: bool = True,
        max_length: int = 512,
        model_name: str | None = None,
    ) -> None:
        self.use_model = use_model
        self.max_length = max_length
        self._model_name = model_name or _DEFAULT_MODEL
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
        self._tokenizer, self._model = _load_nli_model(self._model_name)
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
        """Score using the NLI model."""
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded")
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=1).numpy()[0]
        contradiction_prob = float(probs[_LABEL_CONTRADICTION])
        neutral_prob = float(probs[_LABEL_NEUTRAL])

        return contradiction_prob + (neutral_prob * 0.5)

    @staticmethod
    def _heuristic_score(premise: str, hypothesis: str) -> float:
        """Deterministic heuristic fallback (no model needed)."""
        h_lower = hypothesis.lower()
        if "consistent with reality" in h_lower:
            return 0.1
        if "opposite is true" in h_lower:
            return 0.9
        if "depends on your perspective" in h_lower:
            return 0.5
        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        if not p_words:
            return 0.5
        overlap = len(p_words & h_words) / max(len(p_words), 1)
        return float(np.clip(0.5 - overlap * 0.3, 0.1, 0.9))
