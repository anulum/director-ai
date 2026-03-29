# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Scorer Parameter Validation Tests

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
