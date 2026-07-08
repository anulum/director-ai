# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — bit-exact pure-Python ports of the Rust lexical signals
"""Bit-exact pure-Python ports of the ``backfire_kernel`` lexical signals.

These reproduce ``backfire-core``'s ``signals::{entity_overlap,
numerical_consistency, negation_flip}`` and ``compute::word_overlap`` kernels
**byte-for-byte** so the rule/verified scorers can run on the pure-Python floor
(ADR-0001) — the base ``pip install director-ai`` has no compiled kernel.

The scorers previously carried *drifted* fallbacks: they tokenised with word
regexes and computed different quantities (entity **recall** instead of Jaccard,
a numeric **ratio** instead of a shared-number boolean, negation without the
three-content-word gate). The mandatory-accelerator flag hid the drift by never
running them. This module mirrors the kernel's whitespace tokenisation and set
logic exactly; ``tests/test_lexical_signals_parity.py`` proves each function
against the real kernel over a randomised corpus.
"""

from __future__ import annotations

__all__ = [
    "entity_overlap",
    "numerical_consistency",
    "negation_flip",
    "word_overlap_jaccard",
    "extract_entities",
    "extract_numbers",
]

_ASCII_DIGITS = frozenset("0123456789")

# Mirrors ``backfire-core``'s ``signals::NEG_WORDS`` exactly (order irrelevant).
_NEG_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "cannot",
        "can't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "doesn't",
        "didn't",
        "hasn't",
        "haven't",
        "hadn't",
        "without",
        "none",
        "nobody",
    }
)


def _is_capitalised_word(word: str) -> bool:
    """First character upper-case and the second lower-case (Rust ``is_capitalized_word``)."""
    if len(word) < 2:
        return False
    return word[0].isupper() and word[1].islower()


def extract_entities(text: str) -> set[str]:
    """Group runs of capitalised whitespace tokens into proper-noun entities.

    Mirrors ``signals::extract_entities``: consecutive capitalised
    whitespace-split tokens (punctuation retained) join with a single space into
    one multi-word entity; a non-capitalised token flushes the run.
    """
    entities: set[str] = set()
    current = ""
    for word in text.split():
        if _is_capitalised_word(word):
            current = f"{current} {word}" if current else word
        elif current:
            entities.add(current)
            current = ""
    if current:
        entities.add(current)
    return entities


def extract_numbers(text: str) -> set[str]:
    """Extract digit runs with embedded ``,``/``.`` (Rust ``signals::extract_numbers``).

    A number is a maximal run of ASCII digits, into which ``,`` and ``.`` are
    absorbed only while already inside a run; trailing ``,``/``.`` are trimmed and
    empty results dropped.
    """
    nums: set[str] = set()
    current = ""
    in_num = False
    for ch in text:
        if ch in _ASCII_DIGITS:
            current += ch
            in_num = True
        elif in_num and (ch == "," or ch == "."):
            current += ch
        else:
            if in_num and current:
                trimmed = current.rstrip(",.")
                if trimmed:
                    nums.add(trimmed)
                current = ""
            in_num = False
    if in_num and current:
        trimmed = current.rstrip(",.")
        if trimmed:
            nums.add(trimmed)
    return nums


def entity_overlap(text_a: str, text_b: str) -> float:
    """Jaccard proper-noun overlap; 1.0 when neither text has entities."""
    ents_a = extract_entities(text_a)
    ents_b = extract_entities(text_b)
    if not ents_a and not ents_b:
        return 1.0
    union = len(ents_a | ents_b)
    if union == 0:
        return 1.0
    return len(ents_a & ents_b) / union


def numerical_consistency(text_a: str, text_b: str) -> bool | None:
    """Whether the two texts share at least one number; ``None`` if either has none."""
    nums_a = extract_numbers(text_a)
    nums_b = extract_numbers(text_b)
    if not nums_a and not nums_b:
        return None
    if not nums_a or not nums_b:
        return None
    return not nums_a.isdisjoint(nums_b)


def negation_flip(claim: str, source: str) -> bool:
    """Polarity asymmetry gated by three shared content words (Rust ``negation_flip``)."""
    claim_words = {w.lower() for w in claim.split()}
    source_words = {w.lower() for w in source.split()}
    claim_has_neg = any(w in _NEG_WORDS for w in claim_words)
    source_has_neg = any(w in _NEG_WORDS for w in source_words)
    if claim_has_neg == source_has_neg:
        return False
    content_a = {w for w in claim_words if w not in _NEG_WORDS}
    content_b = {w for w in source_words if w not in _NEG_WORDS}
    return len(content_a & content_b) >= 3


def word_overlap_jaccard(text_a: str, text_b: str) -> float:
    """Jaccard overlap of whitespace-split lower-cased tokens (Rust ``compute::word_overlap``).

    0.0 when either side is empty. This is the same quantity the shared
    :func:`director_ai.core.text_overlap.word_overlap` fast-path computes; kept
    here so the rule scorer's parity is self-contained.
    """
    words_a = {w.lower() for w in text_a.split()}
    words_b = {w.lower() for w in text_b.split()}
    if not words_a or not words_b:
        return 0.0
    union = len(words_a | words_b)
    if union == 0:
        return 0.0
    return len(words_a & words_b) / union
