# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for meta-confidence estimation pipeline."""

from __future__ import annotations

from director_ai.core.scoring.meta_confidence import (
    _margin_confidence,
    _signal_agreement,
    compute_meta_confidence,
)


class TestMarginConfidence:
    def test_zero_margin(self):
        assert _margin_confidence(0.50, 0.50) == 0.0

    def test_small_margin(self):
        c = _margin_confidence(0.51, 0.50)
        assert 0.0 < c < 0.15

    def test_large_margin(self):
        c = _margin_confidence(0.80, 0.50)
        assert c > 0.99

    def test_margin_saturates_at_020(self):
        assert _margin_confidence(0.70, 0.50) > 0.99

    def test_below_threshold(self):
        c = _margin_confidence(0.30, 0.50)
        assert c > 0.99

    def test_symmetric(self):
        above = _margin_confidence(0.55, 0.50)
        below = _margin_confidence(0.45, 0.50)
        assert abs(above - below) < 1e-10

    def test_exact_020_boundary(self):
        assert _margin_confidence(0.70, 0.50) > 0.99
        assert _margin_confidence(0.69, 0.50) < 1.0


class TestSignalAgreement:
    def test_perfect_agreement_both_zero(self):
        assert _signal_agreement(0.0, 0.0) == 1.0

    def test_perfect_agreement_both_one(self):
        assert _signal_agreement(1.0, 1.0) == 1.0

    def test_maximum_disagreement(self):
        assert _signal_agreement(0.0, 1.0) == 0.0

    def test_partial_agreement(self):
        a = _signal_agreement(0.3, 0.5)
        assert 0.7 < a < 0.9

    def test_symmetric(self):
        assert _signal_agreement(0.3, 0.7) == _signal_agreement(0.7, 0.3)


class TestComputeMetaConfidence:
    def test_high_confidence_all_signals(self):
        vc, mc, sa = compute_meta_confidence(
            score=0.90,
            threshold=0.50,
            h_logical=0.1,
            h_factual=0.1,
            nli_confidence=0.95,
        )
        assert vc > 0.8
        assert mc == 1.0
        assert sa == 1.0

    def test_low_confidence_close_to_threshold(self):
        vc, mc, sa = compute_meta_confidence(
            score=0.505,
            threshold=0.50,
            h_logical=0.1,
            h_factual=0.1,
            nli_confidence=0.95,
        )
        assert vc < 0.1

    def test_low_confidence_signal_disagreement(self):
        vc, mc, sa = compute_meta_confidence(
            score=0.80,
            threshold=0.50,
            h_logical=0.1,
            h_factual=0.9,
            nli_confidence=0.95,
        )
        assert vc < 0.3

    def test_low_confidence_uncertain_nli(self):
        vc, mc, sa = compute_meta_confidence(
            score=0.80,
            threshold=0.50,
            h_logical=0.1,
            h_factual=0.1,
            nli_confidence=0.1,
        )
        assert vc == 0.1

    def test_without_nli_confidence(self):
        vc, mc, sa = compute_meta_confidence(
            score=0.80,
            threshold=0.50,
            h_logical=0.1,
            h_factual=0.1,
            nli_confidence=None,
        )
        assert vc == min(mc, sa)

    def test_all_signals_zero(self):
        vc, mc, sa = compute_meta_confidence(
            score=0.50,
            threshold=0.50,
            h_logical=0.0,
            h_factual=1.0,
            nli_confidence=0.0,
        )
        assert vc == 0.0

    def test_verdict_is_min_of_signals(self):
        vc, mc, sa = compute_meta_confidence(
            score=0.60,
            threshold=0.50,
            h_logical=0.3,
            h_factual=0.5,
            nli_confidence=0.7,
        )
        assert vc == min(0.7, mc, sa)

    def test_returns_three_floats(self):
        result = compute_meta_confidence(0.8, 0.5, 0.2, 0.2, 0.9)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_boundary_score_zero(self):
        vc, _, _ = compute_meta_confidence(0.0, 0.5, 0.5, 0.5, 0.5)
        assert vc > 0.0  # far from threshold

    def test_boundary_score_one(self):
        vc, _, _ = compute_meta_confidence(1.0, 0.5, 0.0, 0.0, 1.0)
        assert vc == 1.0
