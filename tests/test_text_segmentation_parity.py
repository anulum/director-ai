# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ADR-0001 text-segmentation bit-exactness parity
"""Bit-exactness parity for the pure-Python text-segmentation ports (ADR-0001).

``director_ai.core.text_segmentation`` reproduces the ``backfire_kernel``
``rust_extract_reasoning_steps`` and ``rust_split_sentences`` kernels so the
reasoning verifier can run kernel-absent. These tests

* lock the ports' contract everywhere (numbered/bullet/NL step strategies, the
  sentence fallback's trailing-punctuation stripping and UTF-8 byte-length filter,
  the abbreviation-aware sentence boundary), and
* compare each port to the real compiled kernel over a randomised corpus with no
  mocks — the empirical proof ADR-0001 requires (skipped only kernel-absent).
"""

from __future__ import annotations

import random

import pytest

from director_ai.core import text_segmentation as seg

try:
    from backfire_kernel import (
        rust_extract_reasoning_steps as _rust_extract_reasoning_steps,
    )
    from backfire_kernel import (
        rust_split_sentences as _rust_split_sentences,
    )

    _HAS_RUST = True
except ImportError:  # pragma: no cover - exercised only in a no-kernel install
    _HAS_RUST = False

_needs_rust = pytest.mark.skipif(
    not _HAS_RUST, reason="backfire_kernel (compiled) not installed"
)


# ── Contract unit tests (always run) ──────────────────────────────────────────


def test_extract_reasoning_steps_numbered() -> None:
    # The numbered-step regex anchors on start-of-string or a newline, so numbered
    # markers must be newline-separated to trip the numbered branch (matching the
    # kernel); an inline "1. .. 2. .." falls through to the sentence strategy.
    steps = seg.extract_reasoning_steps("1. first here\n2. second here\n3. third here")
    assert steps == ["first here", "second here", "third here"]


def test_extract_reasoning_steps_sentence_fallback_strips_terminal_dot() -> None:
    # The sentence fallback splits on every . ! ? and strips it — the last piece
    # loses its trailing period (the drift the reasoning verifier used to carry).
    steps = seg.extract_reasoning_steps("The sky is blue here. Shorter waves win now.")
    assert steps == ["The sky is blue here", "Shorter waves win now"]


def test_extract_reasoning_steps_below_two_is_empty() -> None:
    assert seg.extract_reasoning_steps("just one short bit") == []


def test_split_sentences_keeps_terminal_punctuation() -> None:
    assert seg.split_sentences("Hello world. How are you? Fine!") == [
        "Hello world.",
        "How are you?",
        "Fine!",
    ]


def test_split_sentences_respects_abbreviations() -> None:
    # "Dr." is an abbreviation, so it does not end the sentence.
    assert seg.split_sentences("Dr. Smith arrived. We started.") == [
        "Dr. Smith arrived.",
        "We started.",
    ]


def test_split_sentences_empty_is_empty() -> None:
    assert seg.split_sentences("   ") == []


# ── Parity against the real compiled kernel ───────────────────────────────────

_FRAGMENTS = [
    "Step 1: All men are mortal.",
    "Step 2: Socrates is a man.",
    "First, we assume X.",
    "Then Y follows.",
    "Therefore Z holds.",
    "1. Premise one here.",
    "2. Premise two here.",
    "3. Conclusion drawn.",
    "- Bullet one point.",
    "* Bullet two here.",
    "• Bullet three now.",
    "The sky is blue because scattering.",
    "This means shorter waves dominate.",
    "Hence we see blue.",
    "A is true.",
    "So C must hold.",
    "Dr. Smith agreed.",
    "It costs 5 dollars e.g. cheap.",
    "What now?",
    "Really!",
    "café résumé here.",
    "Next we compute the sum.",
    "Finally, done.",
    "no marker sentence here at all.",
    "U.S. and U.K. differ i.e. in law.",
    "Тест здесь длинное предложение тоже.",
]


def _gen(rng: random.Random) -> str:
    sep = rng.choice([" ", "\n", "  ", "\n\n"])
    return sep.join(rng.choice(_FRAGMENTS) for _ in range(rng.randint(0, 6)))


@_needs_rust
def test_parity_randomised() -> None:
    rng = random.Random(20260708)
    for _ in range(20_000):
        text = _gen(rng)
        assert seg.extract_reasoning_steps(text) == list(
            _rust_extract_reasoning_steps(text)
        ), text
        assert seg.split_sentences(text) == list(_rust_split_sentences(text)), text


@_needs_rust
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Dr. Smith and Prof. Jones met vs. the others etc. here today now.",
        "Step 1: do X. Step 2: do Y.\nStep 3: do Z.",
        "1) first item here 2) second item here 3) third one here",
        "- one\n- two\n* three\n• four",
        "First do this. Then do that. Finally stop.",
        "U.S. and U.K. differ e.g. in law i.e. clearly.",
        "café résumé naïve señor über. Zweite Sätze hier auch mit.",
        "Тест. Второе предложение здесь тоже длинное.",
        "(e.g.) parenthetical abbrev here. Next sentence follows now.",
        "Multiple!!! Punctuation??? Marks... here now really.",
        "\n\nStep 1: leading newlines.\nStep 2: second.\n",
    ],
)
def test_parity_adversarial_edges(text: str) -> None:
    assert seg.extract_reasoning_steps(text) == list(
        _rust_extract_reasoning_steps(text)
    )
    assert seg.split_sentences(text) == list(_rust_split_sentences(text))
