# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PromptRiskScorer

"""Lightweight prompt-risk scorer.

Combines three cheap signals:

* **Length / structure** — very long prompts, prompts with deep
  nesting or unusual punctuation density, and prompts that contain
  system-style markers ("IGNORE PREVIOUS", "SYSTEM:", …) score
  higher. Closed-form, no dependencies.
* **Sanitiser risk** — the existing
  :class:`InputSanitizer` returns a 0-1 risk score; we pass it
  through when the caller has a sanitiser instance.
* **Injection risk** — optional,
  :class:`InjectionDetector` returns an adversarial-signal score
  that complements the sanitiser.

The three signals are combined as a weighted maximum rather than a
linear blend: a single red flag must not be washed out by three
neutral scores. The default weights assume the sanitiser is
authoritative (weight 0.5), injection is strong (weight 0.35), and
the heuristic is a baseline nudge (weight 0.15) — operators tune
them per deployment.

The scorer is deliberately **synchronous and fast**. The heuristic
runs in ``O(len(prompt))`` with a fixed pattern list; even on a
10 000-character prompt it finishes well under 1 ms, so the
gateway can score every request without paying the cost of the
full scoring pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("DirectorAI.Routing.Scorer")

# Cheap-to-match markers of system-prompt exfiltration attempts.
# The list is intentionally narrow — it is not a replacement for
# :class:`InputSanitizer`, only a signal for the length heuristic.
_SYSTEM_STYLE_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore (?:all |the |your |previous )", re.IGNORECASE),
    re.compile(r"\bSYSTEM\s*:", re.IGNORECASE),
    re.compile(r"\[\s*system\s*\]", re.IGNORECASE),
    re.compile(r"(?:you are|act as) (?:a|an) (?:admin|root|developer)", re.IGNORECASE),
    re.compile(r"\bdelimiter\s+collision\b", re.IGNORECASE),
)

# Characters whose count feeds the structure risk — unbalanced
# delimiters and bracket soup are common in prompt-injection payloads.
_STRUCTURAL_CHARS = "[]{}<>|`"


@dataclass(frozen=True)
class RiskComponents:
    """Per-signal breakdown — exposed so callers can log which
    channel drove a halt."""

    heuristic: float
    sanitiser: float
    injection: float
    combined: float

    def as_dict(self) -> dict[str, float]:
        return {
            "heuristic": self.heuristic,
            "sanitiser": self.sanitiser,
            "injection": self.injection,
            "combined": self.combined,
        }


class PromptRiskScorer:
    """Produce a ``[0, 1]`` risk score for a prompt.

    Parameters
    ----------
    sanitiser :
        An object exposing ``.score(prompt) -> float`` (the existing
        :class:`InputSanitizer` fits); ``None`` disables the channel.
    injection_detector :
        An object exposing ``.detect(output=None, intent=prompt)``
        returning something with a ``risk`` float attribute. The
        public :class:`InjectionDetector` fits. ``None`` disables
        the channel.
    weights :
        ``(heuristic, sanitiser, injection)``; must sum to 1.0.
    max_safe_length :
        Prompt length at which the length heuristic saturates at 1.0.
    prefer_rust :
        When ``True`` (the default) and ``backfire_kernel`` is
        importable, the system-marker regex walk delegates to
        :class:`backfire_kernel.PiiScanner` so the whole marker set
        is checked in a single pass with a ``RegexSet`` prefilter.
        Set ``False`` to force the pure-Python path for
        benchmarking or in environments without the Rust wheel.
    """

    def __init__(
        self,
        *,
        sanitiser: Any | None = None,
        injection_detector: Any | None = None,
        weights: tuple[float, float, float] = (0.15, 0.5, 0.35),
        max_safe_length: int = 8000,
        prefer_rust: bool = True,
    ) -> None:
        if max_safe_length <= 0:
            raise ValueError(f"max_safe_length must be positive; got {max_safe_length}")
        total = sum(weights)
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"weights must sum to 1.0; got {total}")
        self._sanitiser = sanitiser
        self._injection = injection_detector
        self._weights = weights
        self._max_safe_length = max_safe_length
        self._rust_scanner = _build_rust_marker_scanner() if prefer_rust else None

    def score(self, prompt: str) -> RiskComponents:
        """Return a :class:`RiskComponents` breakdown for ``prompt``.

        Empty or whitespace-only prompts score zero across every
        channel — the scoring pipeline cannot process them anyway
        and a high risk score here would starve real traffic.
        """
        if not prompt or not prompt.strip():
            return RiskComponents(0.0, 0.0, 0.0, 0.0)

        heuristic = self._heuristic(prompt)
        sanitiser = self._sanitiser_signal(prompt)
        injection = self._injection_signal(prompt)
        w_h, w_s, w_i = self._weights
        # A single strong channel must be able to trip the router
        # on its own — weighting only breaks ties between multiple
        # moderate signals. Combined = max of raw channels and the
        # weighted linear combination.
        linear = heuristic * w_h + sanitiser * w_s + injection * w_i
        combined = max(heuristic, sanitiser, injection, linear)
        combined = min(1.0, max(0.0, combined))
        return RiskComponents(heuristic, sanitiser, injection, combined)

    # ------------------------------------------------------------------

    def _heuristic(self, prompt: str) -> float:
        length = len(prompt)
        length_ratio = min(1.0, length / float(self._max_safe_length))
        structural = sum(prompt.count(ch) for ch in _STRUCTURAL_CHARS)
        structural_density = structural / max(length, 1)
        structural_risk = min(1.0, structural_density * 40.0)
        marker_hits = self._count_marker_hits(prompt)
        marker_risk = min(1.0, marker_hits * 0.35)
        risk = max(
            0.4 * length_ratio,
            0.7 * structural_risk,
            marker_risk,
        )
        return min(1.0, risk)

    def _count_marker_hits(self, prompt: str) -> int:
        """Delegate the system-marker regex walk to the Rust scanner
        when available. The scanner reports one match per hit across
        all patterns; we only need a distinct-category count, so
        collapse by category name before returning."""
        if self._rust_scanner is not None:
            matches = self._rust_scanner.scan(prompt)
            if not matches:
                return 0
            return len({m[0] for m in matches})
        return sum(1 for p in _SYSTEM_STYLE_MARKERS if p.search(prompt))

    @property
    def backend(self) -> str:
        """``"rust"`` when the fast-path scanner is active, otherwise
        ``"python"``. Exposed for tests and operator-facing logs."""
        return "rust" if self._rust_scanner is not None else "python"

    def _sanitiser_signal(self, prompt: str) -> float:
        if self._sanitiser is None:
            return 0.0
        try:
            raw = self._sanitiser.score(prompt)
        except Exception:  # pragma: no cover — defensive
            logger.debug("sanitiser.score failed; treating as zero")
            return 0.0
        return _clip01(raw)

    def _injection_signal(self, prompt: str) -> float:
        if self._injection is None:
            return 0.0
        try:
            result = self._injection.detect(output="", intent=prompt)
        except Exception:  # pragma: no cover — defensive
            logger.debug("injection detector failed; treating as zero")
            return 0.0
        risk = getattr(result, "risk", None)
        if risk is None:
            return 0.0
        return _clip01(risk)


def _clip01(value: Any) -> float:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return 0.0
    if as_float < 0.0:
        return 0.0
    if as_float > 1.0:
        return 1.0
    return as_float


def _build_rust_marker_scanner() -> Any | None:
    """Return a ``backfire_kernel.PiiScanner`` loaded with the
    system-style marker patterns, or ``None`` when the Rust wheel
    is not importable. The scanner works on any regex batch, not
    just PII — the name reflects its original use case but the
    implementation is a generic ``RegexSet`` walker.
    """
    try:
        from backfire_kernel import PiiScanner as _RustScanner
    except ImportError:
        return None
    # The Python patterns are compiled with ``re.IGNORECASE``. Rust
    # ``regex::RegexSet`` does not carry Python flags, so we prepend
    # the ``(?i)`` inline flag on each source pattern.
    patterns: list[tuple[str, str]] = [
        (f"marker_{i}", "(?i)" + p.pattern) for i, p in enumerate(_SYSTEM_STYLE_MARKERS)
    ]
    try:
        return _RustScanner(patterns)
    except Exception:  # pragma: no cover — defensive
        return None
