# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — bit-exact pure-Python ports of the Rust text-segmentation kernels
"""Bit-exact pure-Python ports of the ``backfire_kernel`` text-segmentation kernels.

Reproduces ``backfire-core``'s ``compute::extract_reasoning_steps`` and
``compute::split_sentences`` **byte-for-byte** so the reasoning verifier can run on
the pure-Python floor (ADR-0001) — the base ``pip install director-ai`` has no
compiled kernel. ``tests/test_text_segmentation_parity.py`` proves each function
against the real kernel over a randomised corpus (including the abbreviation and
Unicode edges), so the kernel-present and floor segmentations are identical.

The reasoning verifier's earlier regex fallback drifted from the kernel — most
visibly it kept a trailing ``.`` on the last sentence (``re.split`` on
punctuation-plus-whitespace) where the kernel's ``split(['.', '!', '?'])`` strips
it — which shifts the whitespace-token overlap that drives the step verdicts.
"""

from __future__ import annotations

import re

__all__ = ["extract_reasoning_steps", "split_sentences"]

# Mirror ``compute::{NUMBERED_SPLIT_RE, BULLET_STEP_RE, NL_STEP_RE}`` exactly. The
# numbered/NL patterns anchor on start-of-string or a newline (not a bare ``^``),
# so a base install must not enable ``re.MULTILINE`` on them.
_NUMBERED_SPLIT_RE = re.compile(r"(?:^|\n)\s*(?:Step\s+)?\d+[.):]")
_BULLET_STEP_RE = re.compile(r"^\s*[-*•]\s+(.+)$", re.MULTILINE)
_NL_STEP_RE = re.compile(
    r"(?:^|\n)(?:First|Second|Third|Next|Then|Finally|Therefore|Thus|Hence|So)[,]?\s+",
    re.IGNORECASE,
)

# Mirror ``compute::CHUNK_ABBREVIATIONS`` — tokens ending in ``.`` that do NOT end a
# sentence. Membership is tested after trimming wrapping brackets/quotes and
# lower-casing (``is_abbreviation_token``).
_CHUNK_ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "inc.",
        "ltd.",
        "corp.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "u.s.",
        "u.k.",
    }
)
_ABBREV_TRIM = "()[]{}\"'`"


def extract_reasoning_steps(text: str) -> list[str]:
    """Extract reasoning steps, bit-exact with ``rust_extract_reasoning_steps``.

    Tries numbered-step boundaries, then bullets, then natural-language markers,
    then a sentence fallback (splitting on every ``.``/``!``/``?`` and keeping
    fragments whose UTF-8 byte length exceeds ten). The first strategy that yields
    at least two non-empty pieces wins; otherwise the list is empty.
    """
    numbered = [s.strip() for s in _NUMBERED_SPLIT_RE.split(text) if s.strip()]
    if len(numbered) >= 2:
        return numbered

    bullets = [m.strip() for m in _BULLET_STEP_RE.findall(text)]
    if len(bullets) >= 2:
        return bullets

    nl_steps = [s.strip() for s in _NL_STEP_RE.split(text) if s.strip()]
    if len(nl_steps) >= 2:
        return nl_steps

    # ``str.len()`` in Rust is the UTF-8 byte length; match it, not the code-point
    # count, so multi-byte fragments are filtered identically.
    sentences = [
        stripped
        for s in re.split(r"[.!?]", text)
        if len((stripped := s.strip()).encode("utf-8")) > 10
    ]
    if len(sentences) >= 2:
        return sentences

    return []


def _is_abbreviation_token(token: str) -> bool:
    return token.strip(_ABBREV_TRIM).lower() in _CHUNK_ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Split into sentences, bit-exact with ``rust_split_sentences``.

    Whitespace-tokenises the trimmed text and starts a new sentence after any token
    ending in ``?``/``!``, or in ``.`` unless the token is a known abbreviation;
    terminal punctuation is retained on the token. Empty pieces are dropped.
    """
    trimmed = text.strip()
    if not trimmed:
        return []

    out: list[str] = []
    current: list[str] = []
    for token in trimmed.split():
        current.append(token)
        if token.endswith("?") or token.endswith("!"):
            boundary = True
        elif token.endswith("."):
            boundary = not _is_abbreviation_token(token)
        else:
            boundary = False
        if boundary:
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))

    return [s for s in out if s.strip()]
