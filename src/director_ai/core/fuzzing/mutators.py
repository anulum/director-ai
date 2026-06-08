# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — fuzzing mutation operators

"""Mutation operators that perturb an attack string while preserving intent.

Each operator takes a string and a seeded ``random.Random`` and returns a variant
that a human still reads as the same attack — homoglyphs, zero-width characters,
leetspeak, injected delimiters. The point is to find the variant a guard fails to
flag: an obfuscation that slips past pattern/keyword matching while still carrying
the malicious instruction. Operators are deterministic given the RNG, so any
bypass they find is replayable.
"""

from __future__ import annotations

import random
from collections.abc import Callable

__all__ = ["MUTATORS", "Mutator"]

Mutator = Callable[[str, random.Random], str]

_ZERO_WIDTH = ("​", "‌", "‍", "﻿")
_HOMOGLYPHS = {
    "a": "а",  # Cyrillic a
    "e": "е",  # Cyrillic e
    "o": "о",  # Cyrillic o
    "p": "р",  # Cyrillic er
    "c": "с",  # Cyrillic es
    "x": "х",  # Cyrillic ha
    "i": "і",  # Cyrillic byelorussian i
}
_LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
_DELIMITERS = ("</s>", "[INST]", "###", "<|im_start|>", "```", "<!--")


def _split_indices(text: str, rng: random.Random, fraction: float) -> set[int]:
    """Pick roughly ``fraction`` of the character positions, at least one."""
    if not text:
        return set()
    count = max(1, int(len(text) * fraction))
    return set(rng.sample(range(len(text)), min(count, len(text))))


def case_flip(text: str, rng: random.Random) -> str:
    """Flip the case of a sample of alphabetic characters."""
    targets = _split_indices(text, rng, 0.3)
    return "".join(
        ch.swapcase() if i in targets and ch.isalpha() else ch
        for i, ch in enumerate(text)
    )


def whitespace_inject(text: str, rng: random.Random) -> str:
    """Insert stray whitespace between characters."""
    out = []
    for ch in text:
        out.append(ch)
        if rng.random() < 0.2:
            out.append(rng.choice((" ", "\t", "  ", "\n")))
    return "".join(out)


def zero_width_inject(text: str, rng: random.Random) -> str:
    """Sprinkle zero-width characters through the text."""
    out = []
    for ch in text:
        out.append(ch)
        if rng.random() < 0.25:
            out.append(rng.choice(_ZERO_WIDTH))
    return "".join(out)


def homoglyph_substitute(text: str, rng: random.Random) -> str:
    """Replace some Latin letters with confusable Cyrillic homoglyphs."""
    return "".join(
        _HOMOGLYPHS[ch.lower()]
        if ch.lower() in _HOMOGLYPHS and rng.random() < 0.5
        else ch
        for ch in text
    )


def leetspeak(text: str, rng: random.Random) -> str:
    """Substitute letters with leet digits."""
    return "".join(
        _LEET[ch.lower()] if ch.lower() in _LEET and rng.random() < 0.6 else ch
        for ch in text
    )


def char_duplicate(text: str, rng: random.Random) -> str:
    """Duplicate a sample of characters."""
    targets = _split_indices(text, rng, 0.2)
    return "".join(ch * 2 if i in targets else ch for i, ch in enumerate(text))


def delimiter_inject(text: str, rng: random.Random) -> str:
    """Wrap or splice the text with chat/template delimiters."""
    delimiter = rng.choice(_DELIMITERS)
    if not text:
        return delimiter
    cut = rng.randrange(len(text) + 1)
    return text[:cut] + delimiter + text[cut:]


MUTATORS: dict[str, Mutator] = {
    "case_flip": case_flip,
    "whitespace_inject": whitespace_inject,
    "zero_width_inject": zero_width_inject,
    "homoglyph_substitute": homoglyph_substitute,
    "leetspeak": leetspeak,
    "char_duplicate": char_duplicate,
    "delimiter_inject": delimiter_inject,
}
