# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — No-KB Calibration & Cross-Turn Blending Tests

"""Tests for the no-KB coherence rescaling and cross-turn blending formulas.

The no-KB calibration (scorer.py _heuristic_coherence) rescales coherence
from [lo, hi] to [0, 1] when NLI is available but no KB context exists.
The cross-turn blending merges session context with per-turn h_logic.
"""

from __future__ import annotations

import pytest

from director_ai.core.scoring.scorer import DIVERGENCE_NEUTRAL


class TestNoKBCalibration:
    """Verify rescaling math when NLI is active but no ground truth store."""

    @staticmethod
    def _calibrate(h_logic, w_logic=0.6, w_fact=0.4):
        """Replicate the no-KB calibration formula from scorer.py:1279-1288."""
        h_fact = DIVERGENCE_NEUTRAL
        total_div = w_logic * h_logic + w_fact * h_fact
        coherence = 1.0 - total_div
        lo = 1.0 - w_logic - w_fact * DIVERGENCE_NEUTRAL
        hi = 1.0 - w_fact * DIVERGENCE_NEUTRAL
        span = hi - lo
        if span > 1e-9:
            coherence = max(0.0, min(1.0, (coherence - lo) / span))
        return coherence

    def test_zero_divergence_maps_to_one(self):
        result = self._calibrate(h_logic=0.0)
        assert result == pytest.approx(1.0)

    def test_full_divergence_maps_to_zero(self):
        result = self._calibrate(h_logic=1.0)
        assert result == pytest.approx(0.0)

    def test_half_divergence_maps_to_half(self):
        result = self._calibrate(h_logic=0.5)
        assert result == pytest.approx(0.5)

    def test_quarter_divergence(self):
        result = self._calibrate(h_logic=0.25)
        assert result == pytest.approx(0.75)

    def test_custom_weights_medical(self):
        result = self._calibrate(h_logic=0.0, w_logic=0.5, w_fact=0.5)
        assert result == pytest.approx(1.0)

    def test_custom_weights_medical_full(self):
        result = self._calibrate(h_logic=1.0, w_logic=0.5, w_fact=0.5)
        assert result == pytest.approx(0.0)

    def test_result_clamped_to_unit(self):
        for h in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            result = self._calibrate(h)
            assert 0.0 <= result <= 1.0, f"h_logic={h} produced {result}"

    def test_range_with_default_weights(self):
        lo = 1.0 - 0.6 - 0.4 * 0.5  # 0.2
        hi = 1.0 - 0.4 * 0.5  # 0.8
        assert lo == pytest.approx(0.2)
        assert hi == pytest.approx(0.8)

    def test_monotonic_decrease(self):
        """Higher h_logic → lower calibrated coherence."""
        prev = 1.0
        for h in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            result = self._calibrate(h)
            assert result <= prev + 1e-9, f"Non-monotonic at h_logic={h}"
            prev = result


class TestCrossTurnBlending:
    """Verify the 0.7 * h_logic + 0.3 * cross_turn blending formula."""

    @staticmethod
    def _blend(h_logic, cross_turn, w_logic=0.6, w_fact=0.4):
        """Replicate cross-turn blending from scorer.py:1425-1441."""
        h_fact = DIVERGENCE_NEUTRAL
        blended = 0.7 * h_logic + 0.3 * cross_turn
        total_div = w_logic * blended + w_fact * h_fact
        coherence = 1.0 - total_div
        lo = 1.0 - w_logic - w_fact * DIVERGENCE_NEUTRAL
        hi = 1.0 - w_fact * DIVERGENCE_NEUTRAL
        span = hi - lo
        if span > 1e-9:
            coherence = max(0.0, min(1.0, (coherence - lo) / span))
        return blended, coherence

    def test_identical_scores_no_change(self):
        blended, _ = self._blend(h_logic=0.3, cross_turn=0.3)
        assert blended == pytest.approx(0.3)

    def test_contradictory_cross_turn_increases_divergence(self):
        _, coh_base = self._blend(h_logic=0.2, cross_turn=0.2)
        _, coh_contra = self._blend(h_logic=0.2, cross_turn=0.8)
        assert coh_contra < coh_base

    def test_consistent_cross_turn_decreases_divergence(self):
        _, coh_base = self._blend(h_logic=0.5, cross_turn=0.5)
        _, coh_good = self._blend(h_logic=0.5, cross_turn=0.1)
        assert coh_good > coh_base

    def test_blending_weights_sum(self):
        blended, _ = self._blend(h_logic=0.0, cross_turn=1.0)
        assert blended == pytest.approx(0.3)  # 0.7*0 + 0.3*1

    def test_blending_preserves_range(self):
        for hl in [0.0, 0.5, 1.0]:
            for ct in [0.0, 0.5, 1.0]:
                blended, coherence = self._blend(hl, ct)
                assert 0.0 <= blended <= 1.0, f"blended OOB: hl={hl}, ct={ct}"
                assert 0.0 <= coherence <= 1.0, f"coherence OOB: hl={hl}, ct={ct}"
