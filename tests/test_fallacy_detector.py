# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — informal-fallacy detector tests

"""Detection, polyglot-parity and reasoning-integration coverage for the
informal-fallacy detector."""

from __future__ import annotations

import pytest

from director_ai.core.verification import fallacy_detector as fd
from director_ai.core.verification.fallacy_detector import (
    FallacyMatch,
    FallacyResult,
    detect_fallacies,
)
from director_ai.core.verification.reasoning_verifier import verify_reasoning_chain


@pytest.fixture
def python_only(monkeypatch):
    monkeypatch.setattr(fd, "_RUST_FALLACY", False)
    monkeypatch.setattr(fd, "rust_detect_fallacies", None)


_EXPECT = [
    ("You're just biased.", "ad_hominem"),
    ("Because experts say so it holds.", "appeal_to_authority"),
    ("Everyone knows this works.", "bandwagon"),
    ("You're either with us or against us.", "false_dichotomy"),
    ("This proves that all of them lie.", "hasty_generalization"),
    ("It will inevitably lead to ruin.", "slippery_slope"),
    ("Think of the children here.", "appeal_to_emotion"),
    ("Sales rose after the launch, therefore the launch did it.", "post_hoc"),
]


@pytest.mark.parametrize(("text", "expected"), _EXPECT)
def test_each_fallacy_family_detected(python_only, text, expected):
    result = detect_fallacies(text)
    assert expected in result.types


def test_clean_text_has_no_fallacies(python_only):
    result = detect_fallacies("The capital of France is Paris and 2 + 2 = 4.")
    assert result.clean is True
    assert result.matches == []


def test_match_carries_explanation(python_only):
    match = detect_fallacies("Everyone knows this.").matches[0]
    assert isinstance(match, FallacyMatch)
    assert match.fallacy_type == "bandwagon"
    assert match.explanation == fd.FALLACY_EXPLANATIONS["bandwagon"]


def test_types_property_deduplicates_in_order(python_only):
    result = detect_fallacies("Everyone knows it and everybody agrees with it.")
    # Two bandwagon hits collapse to a single type entry.
    assert result.types == ["bandwagon"]
    assert len(result.matches) == 2


def test_fallacy_result_dataclass_default():
    assert FallacyResult().clean is True


def test_rust_and_python_parity():
    corpus = [t for t, _ in _EXPECT] + [
        "He is incompetent and nobody agrees with him.",
        "The capital of France is Paris.",
        "Correlation here clearly implies a causal link.",
    ]
    rust = {t: fd._scan(t) for t in corpus}
    python = {t: fd._scan_python(t) for t in corpus}
    assert rust == python


def test_rust_scan_path_is_used_by_default():
    # With the kernel installed, _scan must route through it and still parse.
    assert fd._RUST_FALLACY is True
    assert detect_fallacies("Everyone knows this.").types == ["bandwagon"]


# --------------------------------------------------------------------------- #
# integration into verify_reasoning_chain                                      #
# --------------------------------------------------------------------------- #


def test_reasoning_chain_reports_fallacies_without_affecting_validity():
    text = (
        "Step 1: the plan is sound because the numbers add up. "
        "Step 2: besides, everyone knows this approach works. "
        "Therefore we proceed with the plan as outlined."
    )
    with_fallacies = verify_reasoning_chain(text)
    without = verify_reasoning_chain(text, check_fallacies=False)
    assert "bandwagon" in [m.fallacy_type for m in with_fallacies.fallacies]
    # Fallacies are heuristic and must not change the structural validity signal.
    assert with_fallacies.chain_valid == without.chain_valid
    assert with_fallacies.issues_found == without.issues_found


def test_reasoning_chain_fallacies_can_be_disabled():
    result = verify_reasoning_chain(
        "Step 1: everyone knows this. Step 2: so it is true.",
        check_fallacies=False,
    )
    assert result.fallacies == []


def test_reasoning_chain_single_line_still_scans_fallacies():
    result = verify_reasoning_chain("Quick note: everyone knows this is best.")
    assert result.steps_found < 2
    assert [m.fallacy_type for m in result.fallacies] == ["bandwagon"]
