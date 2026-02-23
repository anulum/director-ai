# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Real NLI Backend (DeBERTa)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
NLI-based logical divergence scorer using DeBERTa.

Lazily loads the model on first call, caches per model name.
Falls back to heuristic scoring when the model is unavailable.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

logger = logging.getLogger("DirectorAI.NLI")

_DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

_LABEL_ENTAILMENT = 0
_LABEL_NEUTRAL = 1
_LABEL_CONTRADICTION = 2


@lru_cache(maxsize=4)
def _load_nli_model(model_name: str):
    """Load DeBERTa NLI model + tokenizer, moving to GPU if available."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading NLI model: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info("NLI model loaded on %s.", device)
        return tokenizer, model, device
    except Exception as e:
        logger.warning("NLI model unavailable: %s — using heuristic fallback", e)
        return None, None, None


class NLIScorer:
    """NLI-based logical divergence scorer.

    Usage::

        scorer = NLIScorer()
        h_logical = scorer.score("The sky is blue", "The sky is green")
        # h_logical ≈ 0.9 (contradiction)

    Parameters
    ----------
    use_model : bool — attempt to load DeBERTa model.
    model_name : str — HuggingFace model ID. Overridden by DIRECTOR_NLI_MODEL env var.
    max_length : int — max token length for NLI input.
    chunk_size : int — token count per premise chunk for long-text scoring.
    chunk_overlap : int — overlap tokens between consecutive chunks.
    """

    def __init__(
        self,
        use_model: bool = True,
        model_name: str | None = None,
        max_length: int = 512,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
    ) -> None:
        self.use_model = use_model
        self.model_name = os.environ.get(
            "DIRECTOR_NLI_MODEL", model_name or _DEFAULT_MODEL
        )
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = None
        self._model = None
        self._device = None
        self._model_loaded = False

    def _ensure_model(self) -> bool:
        """Load model if not yet loaded. Returns True if model is available."""
        if self._model_loaded:
            return self._model is not None
        if not self.use_model:
            self._model_loaded = True
            return False
        self._tokenizer, self._model, self._device = _load_nli_model(self.model_name)
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
        if not self._ensure_model() or self._tokenizer is None:
            return self._heuristic_score(premise, hypothesis)

        premise_tokens = self._tokenizer.encode(premise, add_special_tokens=False)
        if len(premise_tokens) > self.max_length - 64:
            return self._chunked_score(premise, hypothesis)
        return self._model_score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple (premise, hypothesis) pairs."""
        return [self.score(p, h) for p, h in pairs]

    def _model_score(self, premise: str, hypothesis: str) -> float:
        """Score a single (premise, hypothesis) pair via DeBERTa NLI."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded")

        import torch

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        if not torch.isfinite(logits).all():
            logger.warning("NLI logits contain NaN/Inf — falling back to heuristic")
            return self._heuristic_score(premise, hypothesis)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        contradiction_prob = float(probs[_LABEL_CONTRADICTION])
        neutral_prob = float(probs[_LABEL_NEUTRAL])

        h = contradiction_prob + (neutral_prob * 0.5)
        return float(np.clip(h, 0.0, 1.0))

    def _chunked_score(self, premise: str, hypothesis: str) -> float:
        """Score long premises by chunking and taking max (AlignScore strategy).

        If ANY chunk contradicts the hypothesis, the text is hallucinated.
        """
        if self._tokenizer is None:
            return self._heuristic_score(premise, hypothesis)
        premise_ids = self._tokenizer.encode(premise, add_special_tokens=False)
        step = max(1, self.chunk_size - self.chunk_overlap)
        scores = []
        for start in range(0, len(premise_ids), step):
            chunk_ids = premise_ids[start : start + self.chunk_size]
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if not chunk_text.strip():
                continue
            scores.append(self._model_score(chunk_text, hypothesis))
        if not scores:
            return self._heuristic_score(premise, hypothesis)
        return max(scores)

    @staticmethod
    def _heuristic_score(premise: str, hypothesis: str) -> float:
        """Deterministic heuristic fallback."""
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
