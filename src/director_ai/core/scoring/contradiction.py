# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — contradiction probability for the streaming halt

"""A three-class NLI contradiction signal for the streaming halt.

The streaming halt cannot use the coherence score (``1 - divergence``): that
score folds NEUTRAL — a correct-but-unsupported claim — into "divergence"
(``divergence = P(contradiction) + 0.5·P(neutral)``), and the default
FactCG grounding model is two-class (supported / not-supported) with no
contradiction class at all. Either way an answer that is correct but not in the
knowledge base scores low and false-halts. Measured: that path halts 50-92% of
factually-correct streaming text.

A halt should fire on a claim that *contradicts* governed facts, not on one that
is merely *unsupported*. :class:`ContradictionScorer` exposes the raw
``P(contradiction)`` from a three-class NLI model (entailment / neutral /
contradiction), so the halt can act on contradiction alone. On the false-halt
benchmark this separates cleanly — correct primary claims sit at p95 ≈ 0.01
contradiction while hallucinations land at 0.37-0.98.

Default model: ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``
(MIT, ungated, with an ONNX export for low-latency serving). On the streaming
false-halt benchmark it catches 2 of 3 contradiction passages at a 1.5%
false-halt rate. A contradiction LoRA fine-tune
(``anulum/director-contradiction-deberta-v3-large``) scores AUC 0.989 on its own
held-out split but does *not* generalise to this gate's input — short retrieved
fact fragments as the premise — where it returns P(contradiction) ~0.01 even for
blatant contradictions and catches 0 of 3, so it is published but not the default.
Revisions are pinned via the model-revision registry. ``transformers`` is an
optional import — installs without it keep the rest of the package working and
get a clear error if they ask for the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..model_revisions import resolve_model_revision

__all__ = [
    "ContradictionResult",
    "ContradictionScorer",
    "DEFAULT_CONTRADICTION_MODEL",
]

DEFAULT_CONTRADICTION_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# Default halt threshold: correct claims sit far below this on the benchmark
# (p95 ≈ 0.01); hallucinations land well above it.
DEFAULT_CONTRADICTION_THRESHOLD = 0.2

_CONTRADICTION_LABELS = frozenset({"contradiction", "contradict", "refuted"})


@dataclass(frozen=True)
class ContradictionResult:
    """A single premise→hypothesis contradiction judgement."""

    contradiction: float
    contradicts: bool


class ContradictionScorer:
    """Raw ``P(contradiction)`` from a three-class NLI model.

    Parameters
    ----------
    model:
        A HuggingFace sequence-classification model with a contradiction class.
    tokenizer:
        Its tokenizer.
    contradiction_idx:
        The index of the contradiction logit (resolved from ``id2label`` by
        :meth:`from_pretrained`).
    threshold:
        ``P(contradiction)`` at or above which :meth:`contradicts` is true.
    max_length:
        Tokenizer truncation length for the (premise, hypothesis) pair.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        contradiction_idx: int,
        threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
        max_length: int = 256,
    ) -> None:
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required")
        self._model = model
        self._tokenizer = tokenizer
        self._ci = int(contradiction_idx)
        self._threshold = float(threshold)
        self._max_length = int(max_length)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_CONTRADICTION_MODEL,
        *,
        revision: str | None = None,
        device: int = -1,
        threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
        max_length: int = 256,
    ) -> ContradictionScorer:
        """Load a three-class NLI model and resolve its contradiction index.

        Raises :class:`ImportError` if ``transformers``/``torch`` are missing,
        and :class:`ValueError` if the model has no contradiction class.
        """
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "ContradictionScorer.from_pretrained requires transformers and "
                "torch. Install with: pip install director-ai[nli]",
            ) from exc

        pinned = resolve_model_revision(model_id, revision)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=pinned)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=pinned
        )
        model.eval()
        if device >= 0 and torch.cuda.is_available():
            model.to(f"cuda:{device}")

        ci = cls._resolve_contradiction_idx(model)
        if ci is None:
            raise ValueError(
                f"{model_id} has no contradiction class in id2label "
                f"({getattr(model.config, 'id2label', None)}); a three-class "
                "NLI model (entailment/neutral/contradiction) is required.",
            )
        return cls(
            model,
            tokenizer,
            contradiction_idx=ci,
            threshold=threshold,
            max_length=max_length,
        )

    @staticmethod
    def _resolve_contradiction_idx(model: Any) -> int | None:
        id2label = getattr(getattr(model, "config", None), "id2label", None)
        if not id2label:
            return None
        for idx, label in id2label.items():
            if str(label).strip().lower() in _CONTRADICTION_LABELS:
                return int(idx)
        return None

    def contradiction(self, premise: str, hypothesis: str) -> float:
        """Return ``P(contradiction)`` in ``[0, 1]`` for premise→hypothesis."""
        if (
            not premise
            or not premise.strip()
            or not hypothesis
            or not hypothesis.strip()
        ):
            return 0.0
        return self.contradiction_batch([(premise, hypothesis)])[0]

    def contradiction_batch(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Vectorised ``P(contradiction)`` for many (premise, hypothesis) pairs."""
        if not pairs:
            return []
        import torch

        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        enc = self._tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, self._ci]
        return [float(x) for x in probs.tolist()]

    def contradicts(self, premise: str, hypothesis: str) -> bool:
        """Report whether ``P(contradiction)`` meets the halt threshold."""
        return self.contradiction(premise, hypothesis) >= self._threshold

    def judge(self, premise: str, hypothesis: str) -> ContradictionResult:
        """Return both the probability and the boolean decision."""
        p = self.contradiction(premise, hypothesis)
        return ContradictionResult(contradiction=p, contradicts=p >= self._threshold)

    @property
    def threshold(self) -> float:
        """Return the contradiction-probability halt threshold."""
        return self._threshold
