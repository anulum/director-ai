# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — toxicity detectors

"""Two toxicity detectors that implement :class:`ModerationDetector`.

* :class:`KeywordToxicityDetector` — zero-dep seed list of slurs
  and attack-style phrases. Not a general-purpose classifier; the
  shipped list is intentionally minimal and documented as a
  starting point that operators extend with their own policy terms.
* :class:`DetoxifyDetector` — wraps Unitary's ``detoxify`` model
  bundle. Reports per-category scores (``toxicity``, ``severe``,
  ``obscene``, ``threat``, ``insult``, ``identity_hate``) and
  generates a match per category that exceeds ``score_threshold``.

Neither detector redacts — they report findings; the Policy layer
decides whether to block, warn, or strip.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .detectors import ModerationDetector, ModerationMatch, ModerationResult

# A short, deliberately non-exhaustive seed list. Operators who need
# a stronger baseline should either extend ``extra_keywords`` or
# switch to :class:`DetoxifyDetector` with a proper model. Every
# entry is lowercased at compile time; ``\b`` word boundaries avoid
# substring false positives ("class" containing "ass").
_DEFAULT_TOXICITY_SEEDS: tuple[str, ...] = (
    "go kill yourself",
    "kys",
    "worthless piece of",
    "i hate you",
    "nobody wants you",
    "you deserve to die",
    "die already",
)

_ATTACK_CATEGORIES: dict[str, tuple[str, ...]] = {
    "threat": (
        r"i will (?:kill|hurt|stab|shoot) you",
        r"going to (?:find|hunt) you",
    ),
    "self_harm_encouragement": (
        r"kill yourself",
        r"end your (?:life|existence)",
    ),
}


class KeywordToxicityDetector(ModerationDetector):
    """Word-boundary keyword matcher with a small default seed list.

    Accepts ``extra_keywords`` and ``extra_patterns`` for operator
    extensions. ``case_sensitive=False`` by default; flip it for
    languages where case is meaningful (e.g. German nouns).
    """

    name = "toxicity_keyword"

    def __init__(
        self,
        *,
        extra_keywords: Iterable[str] | None = None,
        extra_patterns: Iterable[tuple[str, str]] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        flags = 0 if case_sensitive else re.IGNORECASE
        seeds = list(_DEFAULT_TOXICITY_SEEDS) + list(extra_keywords or [])
        # Deduplicate while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for s in seeds:
            key = s if case_sensitive else s.lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(s)
        self._keyword_patterns: list[tuple[str, re.Pattern[str]]] = [
            ("keyword", re.compile(rf"\b{re.escape(k)}\b", flags))
            for k in deduped
        ]
        for category, patterns in _ATTACK_CATEGORIES.items():
            for pattern in patterns:
                self._keyword_patterns.append(
                    (category, re.compile(pattern, flags))
                )
        for category, pattern in extra_patterns or []:
            try:
                self._keyword_patterns.append(
                    (category, re.compile(pattern, flags))
                )
            except re.error as exc:
                raise ValueError(
                    f"invalid regex for toxicity category {category!r}: {exc}"
                ) from exc

    def analyse(self, text: str) -> ModerationResult:
        matches: list[ModerationMatch] = []
        if not text:
            return ModerationResult(detector=self.name, matches=matches)
        for category, pattern in self._keyword_patterns:
            for m in pattern.finditer(text):
                matches.append(
                    ModerationMatch(
                        detector=self.name,
                        category=category,
                        start=m.start(),
                        end=m.end(),
                        text=text,
                    )
                )
        return ModerationResult(detector=self.name, matches=matches)


class DetoxifyDetector(ModerationDetector):
    """Adapter around a ``detoxify``-shaped classifier.

    The caller supplies an object whose ``predict(text)`` method
    returns a ``dict[str, float]`` keyed by category (``toxicity``,
    ``severe_toxicity``, ``obscene``, ``threat``, ``insult``,
    ``identity_attack``). Every category above ``score_threshold``
    emits one :class:`ModerationMatch` spanning the full input.

    Categories the caller explicitly wants to ignore can be
    suppressed via ``categories`` — pass ``{"toxicity", "threat"}``
    to report only those two.
    """

    name = "toxicity_detoxify"

    def __init__(
        self,
        classifier: Any,
        *,
        score_threshold: float = 0.6,
        categories: Iterable[str] | None = None,
    ) -> None:
        if classifier is None:
            raise ValueError("classifier is required")
        self._classifier = classifier
        self._score_threshold = float(score_threshold)
        self._categories = set(categories) if categories is not None else None

    @classmethod
    def from_default_model(
        cls,
        model_type: str = "original",
        **kwargs: Any,
    ) -> DetoxifyDetector:
        """Instantiate a :class:`~detoxify.Detoxify` classifier.

        Raises :class:`ImportError` with install instructions if
        ``detoxify`` is not available. ``model_type`` is forwarded
        verbatim (``"original"`` / ``"unbiased"`` / ``"multilingual"``).
        """
        try:
            from detoxify import Detoxify
        except ImportError as exc:
            raise ImportError(
                "DetoxifyDetector.from_default_model requires "
                "detoxify. Install with: pip install director-ai[toxicity]",
            ) from exc
        return cls(classifier=Detoxify(model_type), **kwargs)

    def analyse(self, text: str) -> ModerationResult:
        if not text:
            return ModerationResult(detector=self.name, matches=[])
        try:
            scores = self._classifier.predict(text)
        except Exception:  # pragma: no cover — defensive
            return ModerationResult(detector=self.name, matches=[])
        matches: list[ModerationMatch] = []
        for category, raw in scores.items():
            if (
                self._categories is not None
                and category not in self._categories
            ):
                continue
            try:
                score = float(raw)
            except (TypeError, ValueError):  # pragma: no cover — defensive
                continue
            if score < self._score_threshold:
                continue
            matches.append(
                ModerationMatch(
                    detector=self.name,
                    category=category,
                    start=0,
                    end=len(text),
                    text=text,
                    score=score,
                )
            )
        return ModerationResult(detector=self.name, matches=matches)
