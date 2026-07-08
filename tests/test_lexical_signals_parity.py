# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ADR-0001 lexical-signals bit-exactness parity
"""Bit-exactness parity for the pure-Python lexical-signal ports (ADR-0001).

``director_ai.core.scoring.lexical_signals`` reproduces the ``backfire_kernel``
``rust_entity_overlap`` / ``rust_numerical_consistency`` / ``rust_negation_flip`` /
``rust_word_overlap`` kernels so the rule scorer can run kernel-absent. These tests

* lock the ports' contract everywhere (whitespace tokenisation, multi-word
  entities, shared-number boolean, the three-content-word negation gate), and
* compare each port to the real compiled kernel over a randomised corpus with no
  mocks — the empirical proof ADR-0001 requires (skipped only kernel-absent).
"""

from __future__ import annotations

import random

import pytest

from director_ai.core.scoring import lexical_signals as lex

try:
    from backfire_kernel import (
        rust_entity_overlap as _rust_entity_overlap,
    )
    from backfire_kernel import (
        rust_negation_flip as _rust_negation_flip,
    )
    from backfire_kernel import (
        rust_numerical_consistency as _rust_numerical_consistency,
    )
    from backfire_kernel import (
        rust_word_overlap as _rust_word_overlap,
    )

    _HAS_RUST = True
except ImportError:  # pragma: no cover - exercised only in a no-kernel install
    _HAS_RUST = False

_needs_rust = pytest.mark.skipif(
    not _HAS_RUST, reason="backfire_kernel (compiled) not installed"
)


# ── Contract unit tests (always run) ──────────────────────────────────────────


def test_entity_overlap_empty_both_is_one() -> None:
    assert lex.entity_overlap("lower only", "still lower") == 1.0


def test_entity_overlap_multiword_entities_group() -> None:
    # "New York" is one entity; disjoint from "Paris" → Jaccard 0/2.
    assert lex.entity_overlap("New York rocks", "Paris rocks") == 0.0
    assert lex.entity_overlap("New York", "New York today") == 1.0


def test_extract_entities_keeps_trailing_punctuation() -> None:
    # Whitespace tokenisation retains the period: "Dr. Smith" is one entity.
    assert lex.extract_entities("Dr. Smith arrived.") == {"Dr. Smith"}


def test_extract_numbers_absorbs_commas_and_trims() -> None:
    assert lex.extract_numbers("Total 1,000 and 2.5, done.") == {"1,000", "2.5"}
    assert lex.extract_numbers("no digits here") == set()


def test_numerical_consistency_shared_number_boolean() -> None:
    assert lex.numerical_consistency("has 10 and 20", "has 10 only") is True
    assert lex.numerical_consistency("has 10", "has 30") is False
    assert lex.numerical_consistency("no numbers", "also none") is None


def test_negation_flip_needs_three_shared_content_words() -> None:
    # Polarity differs but < 3 shared content words → not a flip.
    assert lex.negation_flip("not blue", "blue") is False
    # Polarity differs and ≥ 3 shared content words → flip.
    assert lex.negation_flip("the sky is not clear today", "the sky is clear today")


def test_word_overlap_jaccard_empty_is_zero() -> None:
    assert lex.word_overlap_jaccard("", "anything") == 0.0
    assert lex.word_overlap_jaccard("a b c", "b c d") == 0.5


# ── Parity against the real compiled kernel ───────────────────────────────────

_TOKENS = [
    "the",
    "New",
    "York",
    "Paris",
    "Dr.",
    "Smith",
    "cat",
    "dog",
    "not",
    "no",
    "never",
    "is",
    "happy",
    "sad",
    "10",
    "10.5",
    "100%",
    "3,000",
    "2026",
    "York.",
    "Alpha",
    "Beta",
    "test",
    "without",
    "London",
    "Big",
    "Ben",
    "5",
    "5.",
    "café",
    "École",
    "naïve",
    "and",
    "but",
    "It",
    "He",
    "they",
    "today",
    "clear",
    "friend",
]


def _gen(rng: random.Random) -> str:
    return " ".join(rng.choice(_TOKENS) for _ in range(rng.randint(0, 10)))


@_needs_rust
def test_parity_randomised_all_signals() -> None:
    rng = random.Random(20260708)
    for _ in range(20_000):
        a, b = _gen(rng), _gen(rng)
        assert lex.entity_overlap(a, b) == _rust_entity_overlap(a, b), (a, b)
        assert lex.numerical_consistency(a, b) == _rust_numerical_consistency(a, b), (
            a,
            b,
        )
        assert lex.negation_flip(a, b) == _rust_negation_flip(a, b), (a, b)
        assert lex.word_overlap_jaccard(a, b) == _rust_word_overlap(a, b), (a, b)


@_needs_rust
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Ñoño Ángel visited.", "Ángel Ñoño left."),
        ("ABC DEF GHI.", "abc def."),
        ("A B C is here.", "A B C"),
        ("1,000,000 and 1.5.6 numbers", "1,000,000 here"),
        ("The.Cat.Ran", "The Cat"),
        ("István Kovács", "Kovács István"),
        ("not no never here friend today", "here friend today never"),
        ("MixedCase Word", "Word MixedCase"),
    ],
)
def test_parity_adversarial_edges(a: str, b: str) -> None:
    assert lex.entity_overlap(a, b) == _rust_entity_overlap(a, b)
    assert lex.numerical_consistency(a, b) == _rust_numerical_consistency(a, b)
    assert lex.negation_flip(a, b) == _rust_negation_flip(a, b)
    assert lex.word_overlap_jaccard(a, b) == _rust_word_overlap(a, b)
