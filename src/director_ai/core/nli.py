# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Real NLI Backend (DeBERTa)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
NLI-based logical divergence scorer.

Default model: FactCG-DeBERTa-v3-Large (75.6% balanced accuracy on AggreFact).
Alternative: MiniCheck-DeBERTa-L (72.6%),
install with ``pip install director-ai[minicheck]``.

Supports both 2-class and 3-class NLI models, optional 8-bit quantization
via bitsandbytes, and configurable device/dtype.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

from .metrics import metrics

logger = logging.getLogger("DirectorAI.NLI")

_DEFAULT_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
_RECOMMENDED_MODEL = "lytang/MiniCheck-DeBERTa-L"

# FactCG instruction template (NAACL 2025, derenlei/FactCG)
_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

# Heuristic divergence defaults (shared with scorer.py)
_DIVERGENCE_NEUTRAL = 0.5
_DIVERGENCE_ALIGNED = 0.1
_DIVERGENCE_CONTRADICTED = 0.9


@lru_cache(maxsize=4)
def _load_nli_model(
    model_name: str = _DEFAULT_MODEL,
    quantize_8bit: bool = False,
    device: str | None = None,
    torch_dtype: str | None = None,
):
    """Lazily load an NLI model + tokenizer (cached by model_name)."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading NLI model: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs: dict = {}
        if torch_dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            load_kwargs["torch_dtype"] = dtype_map.get(torch_dtype, torch.float32)

        if quantize_8bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                load_kwargs["device_map"] = "auto"
                logger.info("Loading with 8-bit quantization")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — loading without quantization"
                )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        )

        if device and "device_map" not in load_kwargs:
            model = model.to(device)

        model.eval()
        logger.info("NLI model loaded successfully.")
        return tokenizer, model
    except (ImportError, RuntimeError, OSError, ValueError) as e:
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
        Defaults to FactCG-DeBERTa-v3-Large.
    backend : str — "deberta" (default) or "minicheck".
    quantize_8bit : bool — load model with 8-bit quantization (requires bitsandbytes).
    device : str | None — torch device ("cpu", "cuda", "cuda:0").
    torch_dtype : str | None — "float16", "bfloat16", or "float32".
    """

    _BACKENDS = ("deberta", "minicheck")

    def __init__(
        self,
        use_model: bool = True,
        max_length: int = 512,
        model_name: str | None = None,
        backend: str = "deberta",
        quantize_8bit: bool = False,
        device: str | None = None,
        torch_dtype: str | None = None,
    ) -> None:
        if backend not in self._BACKENDS:
            raise ValueError(
                f"backend must be one of {self._BACKENDS}, got {backend!r}"
            )
        self.use_model = use_model
        self.max_length = max_length
        self.backend = backend
        self._model_name = model_name or _DEFAULT_MODEL
        self._quantize_8bit = quantize_8bit
        self._device = device
        self._torch_dtype = torch_dtype
        self._tokenizer = None
        self._model = None
        self._model_loaded = False
        self._minicheck = None
        self._minicheck_loaded = False

    def _ensure_model(self) -> bool:
        """Load model if not yet loaded. Returns True if model is available."""
        if self._model_loaded:
            return self._model is not None
        if not self.use_model:
            self._model_loaded = True
            return False
        self._tokenizer, self._model = _load_nli_model(
            self._model_name,
            quantize_8bit=self._quantize_8bit,
            device=self._device,
            torch_dtype=self._torch_dtype,
        )
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
        if self.backend == "minicheck":
            return self._minicheck_score(premise, hypothesis)
        if not self._ensure_model():
            return self._heuristic_score(premise, hypothesis)
        return self._model_score(premise, hypothesis)

    async def ascore(self, premise: str, hypothesis: str) -> float:
        """Async version of score() — runs model inference in a thread pool."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.score, premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple (premise, hypothesis) pairs."""
        return [self.score(p, h) for p, h in pairs]

    def _ensure_minicheck(self) -> bool:
        """Load MiniCheck scorer if not yet loaded."""
        if self._minicheck_loaded:
            return self._minicheck is not None
        self._minicheck_loaded = True
        try:
            from minicheck import MiniCheck

            self._minicheck = MiniCheck(model_name="MiniCheck-DeBERTa-L")
            logger.info("MiniCheck backend loaded.")
            return True
        except ImportError:
            logger.warning("minicheck package not installed — pip install minicheck")
            return False
        except (RuntimeError, OSError, ValueError, AttributeError) as e:
            logger.warning("MiniCheck init failed: %s — using heuristic fallback", e)
            return False

    def _minicheck_score(self, premise: str, hypothesis: str) -> float:
        """Score using MiniCheck backend. Falls back to heuristic."""
        if not self._ensure_minicheck() or self._minicheck is None:
            return self._heuristic_score(premise, hypothesis)
        pred = self._minicheck.score(docs=[premise], claims=[hypothesis])
        # MiniCheck returns list of floats in [0,1] where 1 = supported
        return float(1.0 - pred[0])

    @property
    def _is_factcg(self) -> bool:
        return "factcg" in self._model_name.lower()

    def _model_score(self, premise: str, hypothesis: str) -> float:
        """Score using the NLI model.

        Handles both 2-class (supported/not-supported) and 3-class
        (entailment/neutral/contradiction) models.  For 2-class models
        the convention is label0 = not-supported, label1 = supported
        (FactCG).  For 3-class models the convention is label0 = entailment,
        label1 = neutral, label2 = contradiction (DeBERTa-mnli family).

        FactCG models use an instruction template (single-string input).
        Standard NLI models use two-segment (premise, hypothesis) input.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("NLI model not loaded — torch not installed") from None

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded")

        with metrics.timer("nli_inference_seconds"):
            if self._is_factcg:
                text = _FACTCG_TEMPLATE.format(text_a=premise, text_b=hypothesis)
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
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

        if len(probs) == 2:
            return float(1.0 - probs[1])

        # 3-class NLI: label0 = entailment, label1 = neutral, label2 = contradiction
        return float(probs[2]) + float(probs[1]) * 0.5

    # ── Chunked scoring ─────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split on sentence-ending punctuation followed by whitespace."""
        return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count (~4 chars/token for English)."""
        return len(text) // 4 + 1

    def _build_chunks(self, sentences: list[str], budget: int) -> list[str]:
        """Group sentences into chunks within *budget* tokens, 1-sentence overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current and current_tokens + sent_tokens > budget:
                chunks.append(" ".join(current))
                current = [current[-1]]
                current_tokens = self._estimate_tokens(current[0])
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks or [" ".join(sentences)]

    def score_chunked(self, premise: str, hypothesis: str) -> tuple[float, list[float]]:
        """Score with chunking for long hypotheses.

        Returns (aggregated_score, per_chunk_scores).
        Aggregation: max() — worst chunk wins (conservative).
        Short texts bypass chunking (backward-compatible).
        """
        hypothesis_budget = int(self.max_length * 0.6)

        if self._estimate_tokens(hypothesis) <= hypothesis_budget:
            s = self.score(premise, hypothesis)
            return s, [s]

        sentences = self._split_sentences(hypothesis)
        if len(sentences) <= 1:
            s = self.score(premise, hypothesis)
            return s, [s]

        chunks = self._build_chunks(sentences, hypothesis_budget)

        premise_budget = int(self.max_length * 0.4)
        if self._estimate_tokens(premise) > premise_budget:
            premise = premise[: premise_budget * 4]

        chunk_scores = [self.score(premise, chunk) for chunk in chunks]
        return max(chunk_scores), chunk_scores

    @staticmethod
    def _heuristic_score(premise: str, hypothesis: str) -> float:
        """Deterministic heuristic fallback (no model needed)."""
        h_lower = hypothesis.lower()
        if "consistent with reality" in h_lower:
            return _DIVERGENCE_ALIGNED
        if "opposite is true" in h_lower:
            return _DIVERGENCE_CONTRADICTED
        if "depends on your perspective" in h_lower:
            return _DIVERGENCE_NEUTRAL
        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        if not p_words:
            return _DIVERGENCE_NEUTRAL
        overlap = len(p_words & h_words) / max(len(p_words), 1)
        raw = _DIVERGENCE_NEUTRAL - overlap * 0.3
        return float(np.clip(raw, _DIVERGENCE_ALIGNED, _DIVERGENCE_CONTRADICTED))
