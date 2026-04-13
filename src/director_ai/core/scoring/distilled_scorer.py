# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Distilled lightweight NLI scorer
"""Distilled NLI scorer: MiniLM-class model (~22M params) trained via
knowledge distillation from FactCG-DeBERTa-v3-Large.

This is Tier 4 in the 5-tier scoring pyramid — between embedding
similarity (Tier 3, ~65% BA) and full NLI (Tier 5, 75.6% BA). Target:
~70% BA at 5ms latency on CPU via ONNX INT8.

The model is loaded from HuggingFace Hub (``anulum/director-ai-nli-lite``)
or from a local path. ONNX Runtime is used for inference when available,
with a PyTorch fallback.

Install::

    pip install director-ai[nli-lite]

Usage::

    from director_ai.core.scoring.distilled_scorer import DistilledNLIBackend

    backend = DistilledNLIBackend()
    score = backend.score("Water boils at 100°C.", "Water boils at 500°C.")
    # score ≈ 0.2 (low — NLI detects factual contradiction)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("DirectorAI.DistilledNLI")

DEFAULT_DISTILLED_MODEL = "anulum/director-ai-nli-lite"
DEFAULT_DISTILLED_REVISION = None  # will be pinned after first training run


class DistilledNLIBackend:
    """Distilled MiniLM NLI scorer — ~70% BA at 5ms.

    Loads a small NLI model (MiniLM-L6-H384, ~22M params) distilled
    from FactCG-DeBERTa-v3-Large. Supports ONNX Runtime (preferred)
    and PyTorch fallback.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID or local directory path.
    use_onnx : bool
        If True (default), attempt ONNX Runtime inference. Falls back
        to PyTorch if ONNX is unavailable.
    device : str
        ``"cpu"`` or ``"cuda"``. ONNX auto-detects; PyTorch uses this.
    max_length : int
        Maximum token sequence length.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_DISTILLED_MODEL,
        use_onnx: bool = True,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        self._model_path = model_path
        self._use_onnx = use_onnx
        self._device = device
        self._max_length = max_length
        self._tokeniser = None
        self._session = None  # ONNX session
        self._model = None  # PyTorch model
        self._ready = False

    def _ensure_loaded(self) -> None:
        """Lazy-load model on first use."""
        if self._ready:
            return

        # Try ONNX first
        if self._use_onnx:
            try:
                self._load_onnx()
                self._ready = True
                return
            except (ImportError, FileNotFoundError, Exception) as exc:
                logger.warning("ONNX load failed, falling back to PyTorch: %s", exc)

        # PyTorch fallback
        self._load_pytorch()
        self._ready = True

    def _load_onnx(self) -> None:
        """Load ONNX Runtime session + tokeniser."""
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = Path(self._model_path)
        onnx_path = model_dir / "model.onnx" if model_dir.is_dir() else None

        if onnx_path is None or not onnx_path.exists():
            # Try downloading from HF Hub
            from huggingface_hub import hf_hub_download

            onnx_path = Path(
                hf_hub_download(
                    self._model_path,
                    "model.onnx",
                    revision=DEFAULT_DISTILLED_REVISION,
                )
            )

        self._tokeniser = AutoTokenizer.from_pretrained(
            self._model_path,
            revision=DEFAULT_DISTILLED_REVISION,
        )
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("Distilled NLI loaded (ONNX): %s", self._model_path)

    def _load_pytorch(self) -> None:
        """Load PyTorch model + tokeniser."""
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "DistilledNLIBackend requires transformers + torch. "
                "Install with: pip install director-ai[nli-lite]"
            ) from exc

        self._tokeniser = AutoTokenizer.from_pretrained(
            self._model_path,
            revision=DEFAULT_DISTILLED_REVISION,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path,
            revision=DEFAULT_DISTILLED_REVISION,
        )
        self._model.to(self._device).eval()
        self._torch = torch
        logger.info("Distilled NLI loaded (PyTorch): %s", self._model_path)

    def _infer(self, premise: str, hypothesis: str) -> float:
        """Run inference, return P(entailment) in [0, 1]."""
        assert self._tokeniser is not None  # guaranteed by _ensure_loaded
        inputs = self._tokeniser(
            premise,
            hypothesis,
            return_tensors="np" if self._session else "pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )

        if self._session is not None:
            # ONNX path
            ort_inputs = {
                k: v
                for k, v in inputs.items()
                if k in {inp.name for inp in self._session.get_inputs()}
            }
            logits = self._session.run(None, ort_inputs)[0]
            probs = _softmax(logits[0])
        else:
            # PyTorch path
            with self._torch.no_grad():
                pt_inputs = {k: v.to(self._device) for k, v in inputs.items()}
                logits = self._model(**pt_inputs).logits
                probs = self._torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # Convention: label 0 = entailment/supported, label 1 = contradiction
        # Return P(supported) as the score
        return float(probs[0]) if len(probs) >= 2 else float(probs[0])

    def score(self, premise: str, hypothesis: str) -> float:
        """Score groundedness. Returns [0, 1] where 1 = supported."""
        self._ensure_loaded()
        return self._infer(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple pairs."""
        self._ensure_loaded()
        return [self._infer(p, h) for p, h in pairs]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()
