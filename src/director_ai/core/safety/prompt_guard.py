# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — model-backed prompt-injection / jailbreak input screen

"""Model-backed jailbreak and prompt-injection input screening.

The pattern :class:`~director_ai.core.safety.sanitizer.InputSanitizer` catches
the documented jailbreak vocabulary (DAN, "ignore previous instructions",
base64 wrappers) but is blind to adaptive attacks that carry no trigger words —
gradient-optimised suffixes (GCG), paraphrased persuasion (PAIR), or
cipher/encoding evasion. A learned text classifier closes that gap: on the
JailbreakBench artifacts it flags a majority of GCG prompts and a third of PAIR
prompts that the patterns miss entirely, at no measured benign cost.

:class:`PromptInjectionModel` wraps any HuggingFace text-classification model
that emits an "injection"/"safe" decision; the default is ProtectAI's
``deberta-v3-base-prompt-injection-v2`` (Apache-2.0, ungated). The model is an
optional dependency — installs that do not pull ``transformers`` keep the
pattern guard and get a clear error if they ask for the model.

:class:`LayeredPromptGuard` composes the fast pattern stage with the model
stage: a prompt is blocked when *either* fires, so the patterns keep their
zero-latency coverage of known attacks while the model adds adaptive-attack
recall. The decision records which stage fired.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..model_revisions import resolve_model_revision
from .sanitizer import InputSanitizer

__all__ = [
    "PromptScreenResult",
    "PromptInjectionModel",
    "LayeredPromptGuard",
]

DEFAULT_PROMPT_GUARD_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

# Labels different prompt-injection checkpoints use for the positive class.
_INJECTION_LABELS = frozenset({"injection", "jailbreak", "label_1", "1", "unsafe"})


@dataclass(frozen=True)
class PromptScreenResult:
    """Outcome of screening one prompt for injection / jailbreak intent.

    ``blocked`` is the final decision; ``score`` is the injection probability
    from whichever signal is strongest; ``stage`` records which layer fired
    (``"pattern"``, ``"model"``, ``"both"`` or ``""``); ``pattern_reason`` is
    the dominant pattern name when the pattern stage fired.
    """

    blocked: bool
    score: float
    stage: str = ""
    pattern_reason: str = ""

    @property
    def reason(self) -> str:
        """Human-readable cause, compatible with ``SanitizeResult.reason``."""
        if self.stage == "pattern":
            return self.pattern_reason or "pattern"
        if self.stage == "model":
            return "model_classifier"
        return ""


class PromptInjectionModel:
    """Model-backed injection/jailbreak classifier for input prompts.

    Parameters
    ----------
    classifier:
        A callable returning HuggingFace ``text-classification`` output —
        either ``[{"label": ..., "score": ...}]`` or that dict directly. Inject
        a stub in tests; use :meth:`from_pretrained` in production.
    threshold:
        Injection probability at or above which the prompt is flagged.
    injection_labels:
        Label strings (lower-cased) that denote the positive/injection class.
    """

    def __init__(
        self,
        classifier: Any,
        *,
        threshold: float = 0.5,
        injection_labels: frozenset[str] = _INJECTION_LABELS,
    ) -> None:
        if classifier is None:
            raise ValueError("classifier is required")
        self._classifier = classifier
        self._threshold = float(threshold)
        self._injection_labels = injection_labels

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_PROMPT_GUARD_MODEL,
        *,
        revision: str | None = None,
        device: int = -1,
        threshold: float = 0.5,
        max_length: int = 512,
    ) -> PromptInjectionModel:
        """Build a classifier from a HuggingFace model id.

        Raises :class:`ImportError` with install instructions when
        ``transformers`` is unavailable. ``device=-1`` runs on CPU. *revision*
        (a branch, tag, or commit SHA) pins the downloaded weights; when
        ``None`` the immutable revision is resolved from the model-revision
        registry, so a known remote model is always pinned for supply-chain
        reproducibility rather than tracking a moving branch.
        """
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "PromptInjectionModel.from_pretrained requires transformers. "
                "Install with: pip install director-ai[nli]",
            ) from exc
        pinned = resolve_model_revision(model_id, revision)
        clf = pipeline(
            "text-classification",
            model=model_id,
            revision=pinned,
            truncation=True,
            max_length=max_length,
            device=device,
        )
        return cls(clf, threshold=threshold)

    def score(self, text: str) -> float:
        """Return the injection probability in ``[0, 1]`` for *text*.

        A safe-class prediction is mapped to ``1 - score`` so the return value
        is always the injection probability regardless of which class the model
        reports as the argmax.
        """
        if not text or not text.strip():
            return 0.0
        raw = self._classifier(text)
        record = raw[0] if isinstance(raw, (list, tuple)) else raw
        label = str(record.get("label", "")).strip().lower()
        score = float(record.get("score", 0.0))
        if label in self._injection_labels:
            return score
        # Argmax was the safe class — invert to the injection probability.
        return 1.0 - score

    def screen(self, text: str) -> PromptScreenResult:
        """Screen one prompt; blocked when injection probability ≥ threshold."""
        prob = self.score(text)
        return PromptScreenResult(
            blocked=prob >= self._threshold,
            score=prob,
            stage="model" if prob >= self._threshold else "",
        )


class LayeredPromptGuard:
    """Pattern sanitizer first, model classifier second.

    A prompt is blocked when *either* stage fires. The pattern stage runs
    first and short-circuits the model on a known attack (zero added latency);
    otherwise the model — when configured — adds adaptive-attack recall. With
    no model attached this degrades to the pattern guard alone.
    """

    def __init__(
        self,
        sanitizer: InputSanitizer | None = None,
        model: PromptInjectionModel | None = None,
    ) -> None:
        self._sanitizer = sanitizer if sanitizer is not None else InputSanitizer()
        self._model = model

    def screen(self, text: str) -> PromptScreenResult:
        """Screen one prompt through pattern and optional model stages."""
        pattern = self._sanitizer.score(text)
        if pattern.blocked:
            # Known attack — do not pay for model inference.
            return PromptScreenResult(
                blocked=True,
                score=max(pattern.suspicion_score, 0.0),
                stage="pattern",
                pattern_reason=pattern.pattern,
            )
        if self._model is None:
            return PromptScreenResult(
                blocked=False,
                score=pattern.suspicion_score,
                stage="",
            )
        model_result = self._model.screen(text)
        if model_result.blocked:
            return PromptScreenResult(
                blocked=True,
                score=model_result.score,
                stage="model",
            )
        return PromptScreenResult(
            blocked=False,
            score=max(pattern.suspicion_score, model_result.score),
            stage="",
        )

    def screen_many(self, prompts: Sequence[str]) -> list[PromptScreenResult]:
        """Screen prompts in order and return one result per prompt."""
        return [self.screen(p) for p in prompts]

    def check(self, text: str) -> PromptScreenResult:
        """Alias :meth:`screen` for sanitizer-compatible request paths.

        The result exposes ``blocked`` and ``reason``, so a
        :class:`LayeredPromptGuard` is a drop-in replacement for an
        :class:`InputSanitizer` on the server request path.
        """
        return self.screen(text)
