# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Scorer Parameter Validation Tests (STRONG)
"""Multi-angle tests for CoherenceScorer parameter validation.

Covers: threshold boundaries (below/above/NaN/valid), soft_limit validation,
weight sum constraint, custom weights, parametrised valid/invalid values,
pipeline integration, and performance documentation.
"""

import pytest

from director_ai.core.scorer import CoherenceScorer


class TestThresholdValidation:
    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            CoherenceScorer(threshold=-0.1, use_nli=False)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            CoherenceScorer(threshold=1.5, use_nli=False)

    def test_threshold_zero_ok(self):
        s = CoherenceScorer(threshold=0.0, use_nli=False)
        assert s.threshold == 0.0

    def test_threshold_one_ok(self):
        s = CoherenceScorer(threshold=1.0, use_nli=False)
        assert s.threshold == 1.0
        assert s.soft_limit == 1.0  # min(1.0 + 0.1, 1.0) = 1.0


class TestSoftLimitValidation:
    def test_soft_limit_below_threshold_raises(self):
        with pytest.raises(ValueError, match="soft_limit.*must be >= threshold"):
            CoherenceScorer(threshold=0.7, soft_limit=0.5, use_nli=False)

    def test_soft_limit_above_one_raises(self):
        with pytest.raises(ValueError, match="soft_limit must be in"):
            CoherenceScorer(threshold=0.5, soft_limit=1.5, use_nli=False)

    def test_soft_limit_equal_threshold_ok(self):
        s = CoherenceScorer(threshold=0.6, soft_limit=0.6, use_nli=False)
        assert s.soft_limit == 0.6


class TestWeightValidation:
    def test_w_logic_below_zero_raises(self):
        with pytest.raises(ValueError, match="w_logic must be in"):
            CoherenceScorer(threshold=0.5, use_nli=False, w_logic=-0.1, w_fact=1.1)

    def test_w_fact_above_one_raises(self):
        with pytest.raises(ValueError, match="w_fact must be in"):
            CoherenceScorer(threshold=0.5, use_nli=False, w_logic=0.0, w_fact=1.5)

    def test_weights_dont_sum_to_one_raises(self):
        with pytest.raises(ValueError, match="w_logic \\+ w_fact must equal 1.0"):
            CoherenceScorer(threshold=0.5, use_nli=False, w_logic=0.3, w_fact=0.3)

    def test_custom_weights_valid(self):
        s = CoherenceScorer(threshold=0.5, use_nli=False, w_logic=0.3, w_fact=0.7)
        assert s.W_LOGIC == 0.3
        assert s.W_FACT == 0.7


class TestScorerValidationParametrised:
    """Parametrised validation tests."""

    @pytest.mark.parametrize("threshold", [-1.0, -0.01, 1.01, 2.0, float("inf")])
    def test_invalid_thresholds_rejected(self, threshold):
        with pytest.raises(ValueError):
            CoherenceScorer(threshold=threshold, use_nli=False)

    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_valid_thresholds_accepted(self, threshold):
        s = CoherenceScorer(threshold=threshold, use_nli=False)
        assert s.threshold == threshold

    @pytest.mark.parametrize(
        "w_logic,w_fact",
        [(0.3, 0.7), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2), (1.0, 0.0)],
    )
    def test_valid_weight_pairs(self, w_logic, w_fact):
        s = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            w_logic=w_logic,
            w_fact=w_fact,
        )
        assert abs(s.W_LOGIC + s.W_FACT - 1.0) < 1e-9


class TestScorerValidationPerformanceDoc:
    """Document scorer validation pipeline performance."""

    def test_scorer_creation_fast(self):
        import time

        t0 = time.perf_counter()
        for _ in range(100):
            CoherenceScorer(threshold=0.5, use_nli=False)
        per_call_us = (time.perf_counter() - t0) / 100 * 1_000_000
        assert per_call_us < 1000, f"Scorer creation took {per_call_us:.0f}µs"

    def test_scorer_has_threshold(self):
        s = CoherenceScorer(threshold=0.42, use_nli=False)
        assert s.threshold == 0.42
