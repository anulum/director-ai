# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PII detectors (regex + Presidio)

"""Two PII detectors that implement :class:`ModerationDetector`.

* :class:`RegexPIIDetector` extends the pattern list shipped by
  :class:`director_ai.enterprise.redactor.PIIRedactor` with IBAN,
  passport, and IPv4 addresses; it has no optional dependencies and
  is the production default.
* :class:`PresidioPIIDetector` wraps Microsoft Presidio
  (``analyzer``). Presidio catches NER-recognised entities (PERSON,
  LOCATION, NRP, ORGANIZATION) that regex cannot. Install with
  ``pip install director-ai[presidio]``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .detectors import ModerationDetector, ModerationMatch, ModerationResult

_DEFAULT_REGEX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
    ("credit_card", re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "phone",
        re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        ),
    ),
    (
        "phi",
        re.compile(r"\b(?:MRN|DOB|NHS)[\s:]*[A-Z0-9-]+\b", re.IGNORECASE),
    ),
    (
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b"),
    ),
    (
        "passport",
        re.compile(r"\b[A-PR-WY][1-9]\d\s?\d{4}[1-9]\b"),
    ),
    (
        "ipv4",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
        ),
    ),
]


class RegexPIIDetector(ModerationDetector):
    """Regex PII detector with an optional Rust fast-path.

    Shares its pattern list with :class:`PIIRedactor` but exposes
    structured :class:`ModerationMatch` results with start/end
    offsets so the Policy layer can report *where* a finding sits —
    the redactor only rewrites the string in place.

    When ``backfire_kernel`` is importable the detector delegates
    scanning to :class:`backfire_kernel.PiiScanner`, which batches
    every pattern behind a single ``RegexSet`` and skips patterns
    that cannot match. Set ``prefer_rust=False`` to force the pure
    Python path (useful for benchmarking or debugging).
    """

    name = "pii_regex"

    def __init__(
        self,
        extra_patterns: Iterable[tuple[str, str]] | None = None,
        *,
        prefer_rust: bool = True,
    ) -> None:
        compiled = list(_DEFAULT_REGEX_PATTERNS)
        source_patterns: list[tuple[str, str]] = [
            (category, pattern.pattern) for category, pattern in _DEFAULT_REGEX_PATTERNS
        ]
        for category, pattern in extra_patterns or []:
            try:
                compiled.append((category, re.compile(pattern)))
            except re.error as exc:
                raise ValueError(
                    f"invalid regex for category {category!r}: {exc}"
                ) from exc
            source_patterns.append((category, pattern))
        self._patterns = compiled
        self._rust_scanner = (
            _build_rust_scanner(source_patterns) if prefer_rust else None
        )

    def analyse(self, text: str) -> ModerationResult:
        if not text:
            return ModerationResult(detector=self.name, matches=[])
        if self._rust_scanner is not None:
            raw = self._rust_scanner.scan(text)
            rust_matches = [
                ModerationMatch(
                    detector=self.name,
                    category=category,
                    start=start,
                    end=end,
                    text=text,
                )
                for category, start, end in raw
            ]
            return ModerationResult(detector=self.name, matches=rust_matches)
        matches: list[ModerationMatch] = []
        for category, pattern in self._patterns:
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

    @property
    def backend(self) -> str:
        """Which path is active — ``"rust"`` or ``"python"``.

        Useful for tests that want to assert the fast path is wired
        up, and for logging that a given deployment is actually
        exercising the accelerated scanner.
        """
        return "rust" if self._rust_scanner is not None else "python"


def _build_rust_scanner(patterns: list[tuple[str, str]]) -> Any | None:
    """Build a ``backfire_kernel.PiiScanner`` when the Rust extension
    is importable. Returns ``None`` otherwise so the Python regex
    path remains the authoritative fallback — this keeps the
    detector operable on any machine, with or without the Rust
    wheel installed.
    """
    try:
        from backfire_kernel import PiiScanner as _PiiScanner
    except ImportError:
        return None
    try:
        return _PiiScanner(patterns)
    except Exception:  # pragma: no cover — defensive
        return None


class PresidioPIIDetector(ModerationDetector):
    """Adapter around a Presidio ``AnalyzerEngine``-shaped client.

    Accepts the real ``presidio_analyzer.AnalyzerEngine`` instance
    at construction time. Tests inject a stub with the same
    ``analyze(text, language, entities) -> list`` surface so the
    adapter is exercisable without the ~600 MB Presidio stack.

    ``score_threshold`` is forwarded to Presidio; matches with a
    score below it are dropped before being returned.
    """

    name = "pii_presidio"

    def __init__(
        self,
        analyzer: Any,
        *,
        language: str = "en",
        entities: Iterable[str] | None = None,
        score_threshold: float = 0.4,
    ) -> None:
        if analyzer is None:
            raise ValueError("analyzer is required")
        self._analyzer = analyzer
        self._language = language
        self._entities = list(entities) if entities is not None else None
        self._score_threshold = float(score_threshold)

    @classmethod
    def from_default_engine(cls, **kwargs: Any) -> PresidioPIIDetector:
        """Build an adapter backed by a stock
        ``presidio_analyzer.AnalyzerEngine``. Raises
        :class:`ImportError` with install instructions if Presidio is
        not available."""
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:
            raise ImportError(
                "PresidioPIIDetector.from_default_engine requires "
                "presidio-analyzer. Install with: "
                "pip install director-ai[presidio]",
            ) from exc
        return cls(analyzer=AnalyzerEngine(), **kwargs)

    def analyse(self, text: str) -> ModerationResult:
        if not text:
            return ModerationResult(detector=self.name, matches=[])
        call_kwargs: dict[str, Any] = {
            "text": text,
            "language": self._language,
        }
        if self._entities is not None:
            call_kwargs["entities"] = self._entities
        try:
            raw_matches = self._analyzer.analyze(**call_kwargs)
        except TypeError:
            # Some analyzer stubs reject ``entities=`` — retry without it.
            call_kwargs.pop("entities", None)
            raw_matches = self._analyzer.analyze(**call_kwargs)

        out: list[ModerationMatch] = []
        for m in raw_matches:
            score = float(getattr(m, "score", 1.0))
            if score < self._score_threshold:
                continue
            start = int(getattr(m, "start", 0))
            end = int(getattr(m, "end", 0))
            category = str(getattr(m, "entity_type", "unknown")).lower()
            out.append(
                ModerationMatch(
                    detector=self.name,
                    category=category,
                    start=start,
                    end=end,
                    text=text,
                    score=score,
                )
            )
        return ModerationResult(detector=self.name, matches=out)
