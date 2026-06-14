# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — informal-fallacy detection for chain-of-thought

"""Flag informal logical fallacies in a reasoning chain.

The step-logic verifier catches *structural* faults — a conclusion that does not
follow, a circular step — but not the named *informal* fallacies a fluent answer
slips in: attacking the person, appealing to the crowd, forcing a false choice.
This module flags those from their characteristic phrasing.

It is a high-signal **heuristic**, not a proof: it matches the surface markers of
each fallacy family (``everyone knows``, ``you're just biased``, ``will
inevitably lead to``) and reports candidates for review. It deliberately omits
circular reasoning, which the reasoning-chain verifier already detects by step
overlap, and it accepts that genuine appeals to authority (a cited study) read
like the fallacious kind — a match means "look here", not "this is fallacious".

The marker scan runs through the Rust ``rust_detect_fallacies`` kernel when the
compiled extension is installed and an identical pure-Python regex pass
otherwise; the two return the same matches in the same order (the patterns are
deliberately lookaround- and backreference-free so both regex engines agree), so
the dispatch is purely a speed choice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "FallacyMatch",
    "FallacyResult",
    "detect_fallacies",
    "FALLACY_EXPLANATIONS",
]

try:
    from backfire_kernel import rust_detect_fallacies

    _RUST_FALLACY = True
except ImportError:  # pragma: no cover - exercised on installs without the kernel
    rust_detect_fallacies = None
    _RUST_FALLACY = False

# (fallacy_type, pattern). Patterns are case-insensitive and free of lookaround
# and backreferences so the Python ``re`` and Rust ``regex`` engines produce
# identical matches. The order is the scan order on both backends.
_FALLACY_SPECS: list[tuple[str, str]] = [
    (
        "ad_hominem",
        r"\b(?:you|he|she|they)(?:'re| are| is|'s)?\s+(?:just\s+)?(?:an?\s+)?"
        r"(?:too\s+)?(?:idiot|idiots|stupid|fool|fools|ignorant|incompetent|"
        r"biased|liar|liars|clueless|moron|morons|dishonest)\b",
    ),
    (
        "appeal_to_authority",
        r"\b(?:because|since)\s+(?:an?\s+|the\s+)?(?:expert|experts|authority|"
        r"authorities|professor|professors|doctor|doctors|scientist|scientists)\s+"
        r"(?:say|says|said|claim|claims|agree|agrees|believe|believes)\b",
    ),
    (
        "bandwagon",
        r"\b(?:everyone|everybody|nobody|no one)\s+"
        r"(?:knows|agrees|believes|thinks|does it)\b",
    ),
    (
        "false_dichotomy",
        r"\b(?:only\s+two\s+(?:options|choices|possibilities)|"
        r"either\s+with\s+(?:us|me)\s+or\s+against|"
        r"either\s+\w+\s+or\s+nothing)\b",
    ),
    (
        "hasty_generalization",
        r"\b(?:proves|shows|means)\s+(?:that\s+)?"
        r"(?:all|every|everyone|no one|nobody|always|never)\b",
    ),
    (
        "slippery_slope",
        r"\b(?:will|would|could)\s+(?:inevitably|eventually|ultimately)\s+lead\s+to\b"
        r"|\bnext\s+thing\s+you\s+know\b",
    ),
    (
        "appeal_to_emotion",
        r"\bthink\s+of\s+the\s+children\b|\byou\s+should\s+be\s+ashamed\b"
        r"|\bhow\s+would\s+you\s+feel\b",
    ),
    (
        "post_hoc",
        r"\bcorrelat\w+[^.?!]{0,30}?\bcaus\w+"
        r"|\bafter\s+\w+[^.?!]{0,40}?\btherefore\b",
    ),
]

FALLACY_EXPLANATIONS: dict[str, str] = {
    "ad_hominem": "attacks the person rather than the argument",
    "appeal_to_authority": "treats a claim as true merely because an authority stated it",
    "bandwagon": "appeals to popularity instead of evidence",
    "false_dichotomy": "presents only two options when more exist",
    "hasty_generalization": "generalises from insufficient evidence",
    "slippery_slope": "assumes one step inevitably leads to an extreme outcome",
    "appeal_to_emotion": "substitutes an emotional appeal for an argument",
    "post_hoc": "infers causation from mere sequence or correlation",
}

_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _FALLACY_SPECS
]


@dataclass(frozen=True)
class FallacyMatch:
    """One detected fallacy marker."""

    fallacy_type: str
    text: str
    explanation: str


@dataclass
class FallacyResult:
    """Informal-fallacy scan over a text."""

    matches: list[FallacyMatch] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.matches

    @property
    def types(self) -> list[str]:
        seen: list[str] = []
        for match in self.matches:
            if match.fallacy_type not in seen:
                seen.append(match.fallacy_type)
        return seen


def _scan_python(text: str) -> list[tuple[str, str]]:
    """Pure-Python marker scan mirroring the Rust kernel exactly."""
    found: list[tuple[str, str]] = []
    for name, pattern in _COMPILED:
        for match in pattern.finditer(text):
            found.append((name, match.group(0)))
    return found


def _scan(text: str) -> list[tuple[str, str]]:
    if _RUST_FALLACY and rust_detect_fallacies is not None:
        return [(t, s) for t, s in rust_detect_fallacies(text)]
    return _scan_python(text)


def detect_fallacies(text: str) -> FallacyResult:
    """Detect informal-fallacy markers in *text*.

    Returns a :class:`FallacyResult` with one :class:`FallacyMatch` per marker, in
    scan order (fallacy family order, then left-to-right within the text).
    """
    matches = [
        FallacyMatch(
            fallacy_type=name,
            text=span,
            explanation=FALLACY_EXPLANATIONS[name],
        )
        for name, span in _scan(text)
    ]
    return FallacyResult(matches=matches)
