# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Heuristic coherence orchestration tests
"""Focused tests for the split heuristic-coherence orchestration helpers."""

from __future__ import annotations

import pytest

from director_ai.core.scoring.heuristic_coherence import (
    HeuristicCoherenceInputs,
    HeuristicCoherenceRoute,
    combine_weighted_coherence,
    select_heuristic_coherence_route,
)
from director_ai.core.scoring.scorer import DIVERGENCE_NEUTRAL


def test_route_selection_keeps_dialogue_before_summarisation():
    inputs = HeuristicCoherenceInputs(
        auto_dialogue_profile=True,
        use_prompt_as_premise=False,
        nli_available=True,
        task_type="dialogue",
        w_logic=0.0,
    )

    assert select_heuristic_coherence_route(inputs) is HeuristicCoherenceRoute.DIALOGUE


def test_route_selection_keeps_zero_logic_short_circuit_without_nli():
    inputs = HeuristicCoherenceInputs(
        auto_dialogue_profile=True,
        use_prompt_as_premise=True,
        nli_available=False,
        task_type="default",
        w_logic=0.0,
    )

    assert (
        select_heuristic_coherence_route(inputs) is HeuristicCoherenceRoute.FACTUAL_ONLY
    )


@pytest.mark.parametrize(
    ("h_logic", "expected"),
    [
        (0.0, 1.0),
        (0.25, 0.75),
        (0.5, 0.5),
        (1.0, 0.0),
    ],
)
def test_combine_weighted_coherence_calibrates_no_kb_neutral_fact(
    h_logic: float,
    expected: float,
):
    result = combine_weighted_coherence(
        h_logic=h_logic,
        h_factual=DIVERGENCE_NEUTRAL,
        w_logic=0.6,
        w_fact=0.4,
        nli_available=True,
        evidence_present=False,
        dialogue_route=False,
    )

    assert result == pytest.approx(expected)


def test_combine_weighted_coherence_skips_no_kb_calibration_for_dialogue():
    result = combine_weighted_coherence(
        h_logic=0.0,
        h_factual=DIVERGENCE_NEUTRAL,
        w_logic=0.6,
        w_fact=0.4,
        nli_available=True,
        evidence_present=False,
        dialogue_route=True,
    )

    assert result == pytest.approx(0.8)
