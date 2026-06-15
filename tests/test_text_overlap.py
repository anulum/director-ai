# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — shared word-overlap helper tests

from __future__ import annotations

import pytest

from director_ai.core import text_overlap as to
from director_ai.core.text_overlap import word_overlap


def test_identical_and_disjoint():
    assert word_overlap("alpha beta gamma", "alpha beta gamma") == 1.0
    assert word_overlap("alpha beta", "gamma delta") == 0.0


def test_empty_inputs():
    assert word_overlap("", "") == 0.0
    assert word_overlap("word", "") == 0.0


def test_partial_overlap_value():
    # {the,cat,sat} vs {the,dog,sat}: |∩|=2 (the,sat), |∪|=4 -> 0.5
    assert word_overlap("the cat sat", "the dog sat") == pytest.approx(0.5)


def test_case_folded_and_punctuation_retained():
    # case-folded
    assert word_overlap("Alpha BETA", "alpha beta") == 1.0
    # punctuation retained on the token (matches the Rust kernel), so "cat." and
    # "cat" are distinct
    assert word_overlap("cat.", "cat") == 0.0


@pytest.mark.parametrize(
    "a,b",
    [
        ("the cat sat on the mat", "the dog sat on the mat"),
        ("one two three four", "three four five six"),
        ("repeated repeated word", "word repeated"),
    ],
)
def test_python_and_rust_paths_are_bit_exact(a, b, monkeypatch):
    """Forcing each path yields the same value (dispatch is a speed choice)."""
    # Python path (default, small input)
    py = word_overlap(a, b)
    # Force the Rust path by setting the threshold to 0 words
    rust = word_overlap(a, b, large_input_words=0)
    assert rust == pytest.approx(py)


def test_large_input_uses_rust_when_available(monkeypatch):
    calls = {"n": 0}

    def _fake_rust(a, b):
        calls["n"] += 1
        return 0.5

    monkeypatch.setattr(to, "_RUST_WORD_OVERLAP", True)
    monkeypatch.setattr(to, "rust_word_overlap", _fake_rust)
    big = " ".join(["w"] * 1200)
    word_overlap(big, big, large_input_words=1000)
    assert calls["n"] == 1  # dispatched to Rust above the threshold


def test_small_input_skips_rust(monkeypatch):
    calls = {"n": 0}

    def _fake_rust(a, b):  # pragma: no cover - must not be called
        calls["n"] += 1
        return 0.5

    monkeypatch.setattr(to, "_RUST_WORD_OVERLAP", True)
    monkeypatch.setattr(to, "rust_word_overlap", _fake_rust)
    word_overlap("a small input", "another small input", large_input_words=1000)
    assert calls["n"] == 0  # stayed on Python below the threshold


def test_python_fallback_when_kernel_absent(monkeypatch):
    monkeypatch.setattr(to, "_RUST_WORD_OVERLAP", False)
    monkeypatch.setattr(to, "rust_word_overlap", None)
    big = " ".join(["w"] * 1200)
    # even above the threshold, with no kernel it must still return the Python value
    assert word_overlap(big, big, large_input_words=1000) == 1.0
