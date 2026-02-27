# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Real NLI Backend (DeBERTa)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
NLI-based logical divergence scorer.

Recommended model: MiniCheck-DeBERTa-L (72.6% balanced accuracy on AggreFact).
Default model: DeBERTa-v3-base-mnli-fever-anli (66.2% — use MiniCheck for
production; install with ``pip install director-ai[minicheck]``).

Supports both 2-class and 3-class NLI models, optional 8-bit quantization
via bitsandbytes, and configurable device/dtype.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger("DirectorAI.NLI")

_DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
_RECOMMENDED_MODEL = "lytang/MiniCheck-DeBERTa-L"


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
        Defaults to DeBERTa-v3-base-mnli-fever-anli.
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
        except Exception as e:
            logger.warning("MiniCheck init failed: %s — using heuristic fallback", e)
            return False

    def _minicheck_score(self, premise: str, hypothesis: str) -> float:
        """Score using MiniCheck backend. Falls back to heuristic."""
        if not self._ensure_minicheck() or self._minicheck is None:
            return self._heuristic_score(premise, hypothesis)
        pred = self._minicheck.score(docs=[premise], claims=[hypothesis])
        # MiniCheck returns list of floats in [0,1] where 1 = supported
        return float(1.0 - pred[0])

    def _model_score(self, premise: str, hypothesis: str) -> float:
        """Score using the NLI model.

        Handles both 2-class (supported/not-supported) and 3-class
        (entailment/neutral/contradiction) models.  For 2-class models
        the convention is label0 = not-supported, label1 = supported
        (FactCG).  For 3-class models the convention is label0 = entailment,
        label1 = neutral, label2 = contradiction (DeBERTa-mnli family).
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("NLI model not loaded — torch not installed") from None

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

        if len(probs) == 2:
            # Binary: label0 = not-supported, label1 = supported
            return float(1.0 - probs[1])

        # 3-class NLI: label0 = entailment, label1 = neutral, label2 = contradiction
        return float(probs[2]) + float(probs[1]) * 0.5

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
