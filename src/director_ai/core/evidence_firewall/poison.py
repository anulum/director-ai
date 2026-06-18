# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence firewall poisoning scan

"""Dependency-free indirect-prompt-injection scan for retrieved chunks.

A poisoned chunk carries instructions aimed at the model rather than facts aimed
at the user — "ignore the previous instructions", "you are now …", an injected
tool call, or a leaked system-prompt fragment. This module scores how strongly a
chunk's text reads as such an instruction, with no model dependency, so the
firewall can run it on every chunk on the hot path. The model-backed
:class:`~director_ai.core.safety.injection.InjectionDetector` can be injected in
its place when a heavier signal is wanted.
"""

from __future__ import annotations

import re

__all__ = ["PoisonScanner", "default_poison_scan"]

# Marker phrases grouped by how strongly each signals an instruction aimed at
# the model. Weights are summed and squashed; no single weak phrase trips the
# default threshold, but a strong override phrase on its own does.
_STRONG_MARKERS: tuple[str, ...] = (
    "ignore the previous instructions",
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the above",
    "disregard previous instructions",
    "forget the previous instructions",
    "you are now",
    "from now on you are",
    "new instructions:",
    "system prompt:",
    "override the system prompt",
    "reveal your system prompt",
    "print your instructions",
)
_MEDIUM_MARKERS: tuple[str, ...] = (
    "do not tell the user",
    "without telling the user",
    "act as",
    "pretend to be",
    "bypass",
    "jailbreak",
    "developer mode",
    "respond only with",
    "your real instructions",
)
_WEAK_MARKERS: tuple[str, ...] = (
    "as an ai",
    "language model",
    "assistant:",
    "user:",
    "<|im_start|>",
    "[system]",
)

_STRONG_WEIGHT = 0.7
_MEDIUM_WEIGHT = 0.3
_WEAK_WEIGHT = 0.12

# A tool/function call literal embedded in retrieved text is a strong signal of
# an injected action rather than a stored fact.
_TOOL_CALL = re.compile(
    r"""(?ix)
    \b(tool_call|function_call|call\s+the\s+tool|invoke\s+tool)\b
    | "name"\s*:\s*"[a-z_]+"\s*,\s*"arguments"
    """,
)


def default_poison_scan(text: str) -> float:
    """Return an indirect-injection score in ``[0, 1]`` for ``text``.

    The score sums per-marker weights for every distinct marker phrase present
    (case-insensitive) plus a bonus for an embedded tool-call literal, then caps
    at ``1.0``. Distinct markers — not repeat counts — drive the score so a
    single legitimately quoted phrase is bounded.
    """
    if not text:
        return 0.0
    lowered = text.lower()
    score = 0.0
    for marker in _STRONG_MARKERS:
        if marker in lowered:
            score += _STRONG_WEIGHT
    for marker in _MEDIUM_MARKERS:
        if marker in lowered:
            score += _MEDIUM_WEIGHT
    for marker in _WEAK_MARKERS:
        if marker in lowered:
            score += _WEAK_WEIGHT
    if _TOOL_CALL.search(text):
        score += _STRONG_WEIGHT
    return min(score, 1.0)


class PoisonScanner:
    """Callable wrapper that adds a hard allow/deny verdict to the score.

    Parameters
    ----------
    threshold:
        Score in ``[0, 1]`` at or above which :meth:`is_poisoned` returns
        ``True``. The firewall uses the score directly via ``__call__``; this
        wrapper is for standalone callers that want a boolean.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def __call__(self, text: str) -> float:
        """Return the raw poison score for ``text`` in ``[0, 1]``."""
        return default_poison_scan(text)

    def is_poisoned(self, text: str) -> bool:
        """Return whether ``text`` scores at or above the threshold."""
        return default_poison_scan(text) >= self.threshold
