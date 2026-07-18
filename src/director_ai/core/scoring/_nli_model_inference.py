# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Inference Backends (NLIScorer mixin)

"""PyTorch and ONNX inference backends for the NLI scorer.

:class:`ModelInferenceMixin` owns the tokenizer-based inference surface of
:class:`~director_ai.core.scoring.nli.NLIScorer`: serialised tokenisation,
the FactCG instruction template, single-pair and batched PyTorch forward
passes, batched ONNX Runtime inference, and the confidence-returning
variants of both. The mixin owns no ``__init__`` — the composing scorer
initialises every attribute declared on the class body.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ..metrics import metrics
from ..safety.sanitizer import InputSanitizer
from ._nli_numeric import (
    _probs_to_confidence,
    _probs_to_divergence,
    _softmax_np,
)

__all__ = ["ModelInferenceMixin"]

# FactCG instruction template (NAACL 2025, derenlei/FactCG)
_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

#: Upper bound on pairs per batched forward pass. A single forward over an
#: unbounded claims×chunks pair list allocates activations proportional to
#: the pair count — a long RAGTruth sample OOMed a 24 GB A30 (2026-07-18).
#: Rows are scored independently (padding is attention-masked), so chunking
#: bounds memory without changing any score.
_MAX_PAIRS_PER_FORWARD = 32


def _normalise_nli_text(text: str) -> str:
    """Strip zero-width/confusable perturbations from model input (KIMI2-J).

    An invisible character (e.g. U+200B) inside an otherwise-true claim splits
    a word for the tokenizer and inflates the divergence, false-halting a true
    input (measured on GPU 2026-07-16: a single zero-width space pushed a true
    claim's support from 0.87 to 0.44). ASCII text cannot carry such a
    perturbation and NFKC is a no-op on it, so the scan is skipped on the
    common path; non-ASCII text is NFKC-normalised with control/format
    characters removed via the canonical input sanitiser (no confusable
    folding, so legitimate non-Latin scripts keep their letters).
    """
    if text.isascii():
        return text
    return InputSanitizer.scrub(text)


def _normalise_nli_arg(value: Any) -> Any:
    """Normalise a positional tokenizer argument (str or list[str])."""
    if isinstance(value, str):
        return _normalise_nli_text(value)
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return [_normalise_nli_text(v) for v in value]
    return value


class ModelInferenceMixin:
    """PyTorch/ONNX inference surface of :class:`NLIScorer`.

    All state is initialised by the composing scorer's ``__init__``; the
    annotations below declare that shared contract for static analysis
    without creating attributes.
    """

    # Shared state initialised by the composing scorer.
    max_length: int
    _model_name: str
    _tokenizer: Any
    _tokenizer_lock: threading.Lock
    _model: Any
    _onnx_session: Any
    _last_token_count: int
    _label_indices: tuple[int, int] | None

    @property
    def _is_factcg(self) -> bool:
        """Return whether the configured model expects the FactCG prompt template."""
        return "factcg" in self._model_name.lower()

    def _tokenize(self, *args: Any, **kwargs: Any) -> Any:
        """Serialise access to the shared tokenizer.

        The fast (Rust) tokenizer mutates its truncation/padding state on
        every call, so concurrent invocations from the parallel logical and
        factual divergence futures raise ``RuntimeError("Already borrowed")``
        on fast hardware. Guard only the encode step — the model forward
        stays outside the lock and runs in parallel — so scores are identical
        and throughput is preserved.
        """
        if self._tokenizer is None:
            raise RuntimeError("NLI model not loaded")
        # Scrub zero-width/confusable perturbations from the text arguments
        # before tokenisation (KIMI2-J); done outside the lock since it is
        # pure per-string work. Positional args carry the text (a template
        # string, a premise/hypothesis pair, or batched lists); kwargs carry
        # tokenizer options and are left untouched.
        args = tuple(_normalise_nli_arg(a) for a in args)
        with self._tokenizer_lock:
            return self._tokenizer(*args, **kwargs)

    def _model_score(self, premise: str, hypothesis: str) -> float:
        """Single-pair PyTorch inference.

        Handles 2-class (supported/not-supported) and 3-class
        (entailment/neutral/contradiction) models. FactCG uses an
        instruction template; standard NLI uses two-segment input.
        """
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded")

        import torch

        device = next(self._model.parameters()).device

        with metrics.timer("nli_inference_seconds"):
            if self._is_factcg:
                text = _FACTCG_TEMPLATE.format(text_a=premise, text_b=hypothesis)
                inputs = self._tokenize(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                inputs = self._tokenize(
                    premise,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )

            self._last_token_count += inputs["input_ids"].numel()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        if len(probs) == 2:
            return float(1.0 - probs[1])
        ci, ni = self._label_indices or (2, 1)
        return float(probs[ci]) + float(probs[ni]) * 0.5

    def _model_score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batched PyTorch inference in memory-bounded chunks.

        At most :data:`_MAX_PAIRS_PER_FORWARD` pairs go through one forward
        pass; an empty batch short-circuits to an empty result.
        """
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded")

        results: list[float] = []
        with metrics.timer("nli_batch_inference_seconds"):
            for start in range(0, len(pairs), _MAX_PAIRS_PER_FORWARD):
                results.extend(
                    self._model_score_batch_chunk(
                        pairs[start : start + _MAX_PAIRS_PER_FORWARD]
                    )
                )
        return results

    def _model_score_batch_chunk(self, pairs: list[tuple[str, str]]) -> list[float]:
        """One bounded PyTorch forward pass over *pairs*."""
        import torch

        device = next(self._model.parameters()).device

        if self._is_factcg:
            texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
            inputs = self._tokenize(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
        else:
            premises = [p for p, _ in pairs]
            hypotheses = [h for _, h in pairs]
            inputs = self._tokenize(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )

        self._last_token_count += inputs["input_ids"].numel()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()

        return _probs_to_divergence(probs, self._label_indices)

    def _onnx_score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batched ONNX Runtime inference in memory-bounded chunks."""
        if self._tokenizer is None or self._onnx_session is None:
            raise RuntimeError("ONNX session not loaded")

        results: list[float] = []
        with metrics.timer("nli_onnx_batch_seconds"):
            for start in range(0, len(pairs), _MAX_PAIRS_PER_FORWARD):
                results.extend(
                    self._onnx_score_batch_chunk(
                        pairs[start : start + _MAX_PAIRS_PER_FORWARD]
                    )
                )
        return results

    def _onnx_score_batch_chunk(self, pairs: list[tuple[str, str]]) -> list[float]:
        """One bounded ONNX Runtime forward pass over *pairs*."""
        if self._is_factcg:
            texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
            inputs = self._tokenize(
                texts,
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
        else:
            premises = [p for p, _ in pairs]
            hypotheses = [h for _, h in pairs]
            inputs = self._tokenize(
                premises,
                hypotheses,
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )

        self._last_token_count += inputs["input_ids"].size
        # Feed only inputs the ONNX graph expects, cast to int64
        expected = {i.name for i in self._onnx_session.get_inputs()}
        feed = {
            k: v.astype(np.int64) if v.dtype != np.int64 else v
            for k, v in inputs.items()
            if k in expected
        }
        logits = self._onnx_session.run(None, feed)[0]

        return _probs_to_divergence(_softmax_np(logits), self._label_indices)

    def _model_score_batch_with_confidence(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        """Chunked PyTorch inference returning (divergence, confidence)."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model not loaded")

        results: list[tuple[float, float]] = []
        with metrics.timer("nli_batch_inference_seconds"):
            for start in range(0, len(pairs), _MAX_PAIRS_PER_FORWARD):
                results.extend(
                    self._model_score_batch_with_confidence_chunk(
                        pairs[start : start + _MAX_PAIRS_PER_FORWARD]
                    )
                )
        return results

    def _model_score_batch_with_confidence_chunk(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        """One bounded PyTorch forward returning (divergence, confidence)."""
        import torch

        device = next(self._model.parameters()).device

        if self._is_factcg:
            texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
            inputs = self._tokenize(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
        else:
            premises = [p for p, _ in pairs]
            hypotheses = [h for _, h in pairs]
            inputs = self._tokenize(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )

        self._last_token_count += inputs["input_ids"].numel()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        divergences = _probs_to_divergence(probs, self._label_indices)
        confidences = _probs_to_confidence(probs)
        return list(zip(divergences, confidences, strict=True))

    def _onnx_score_batch_with_confidence(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        """Chunked ONNX inference returning (divergence, confidence)."""
        if self._tokenizer is None or self._onnx_session is None:
            raise RuntimeError("ONNX session not loaded")

        results: list[tuple[float, float]] = []
        with metrics.timer("nli_onnx_batch_seconds"):
            for start in range(0, len(pairs), _MAX_PAIRS_PER_FORWARD):
                results.extend(
                    self._onnx_score_batch_with_confidence_chunk(
                        pairs[start : start + _MAX_PAIRS_PER_FORWARD]
                    )
                )
        return results

    def _onnx_score_batch_with_confidence_chunk(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        """One bounded ONNX forward returning (divergence, confidence)."""
        if self._is_factcg:
            texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
            inputs = self._tokenize(
                texts,
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
        else:
            premises = [p for p, _ in pairs]
            hypotheses = [h for _, h in pairs]
            inputs = self._tokenize(
                premises,
                hypotheses,
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )

        self._last_token_count += inputs["input_ids"].size
        expected = {i.name for i in self._onnx_session.get_inputs()}
        feed = {
            k: v.astype(np.int64) if v.dtype != np.int64 else v
            for k, v in inputs.items()
            if k in expected
        }
        logits = self._onnx_session.run(None, feed)[0]

        sm = _softmax_np(logits)
        divergences = _probs_to_divergence(sm, self._label_indices)
        confidences = _probs_to_confidence(sm)
        return list(zip(divergences, confidences, strict=True))
