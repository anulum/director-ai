# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — probability calibration tests

from __future__ import annotations

import warnings

import numpy as np
import pytest

from director_ai.core.calibration.probability_calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)


class TestInputValidation:
    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            expected_calibration_error([0.1, 0.2], [1])

    def test_two_dimensional_rejected(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            brier_score([[0.1, 0.2]], [[1, 0]])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            brier_score([], [])

    def test_non_finite_probability_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            brier_score([float("nan")], [1])

    def test_out_of_range_probability_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score([1.5], [1])

    def test_non_binary_label_rejected(self):
        with pytest.raises(ValueError, match="binary"):
            brier_score([0.5], [2])


class TestReliabilityAndMetrics:
    def test_reliability_bins_omit_empty_and_close_top(self):
        # p == 1.0 must land in the last bin, not overflow.
        bins = reliability_bins([0.05, 1.0], [0, 1], n_bins=10)
        assert len(bins) == 2
        assert bins[0][2] == 1  # count
        assert bins[-1][0] == pytest.approx(1.0)  # mean confidence of top bin

    def test_reliability_bins_rejects_zero_bins(self):
        with pytest.raises(ValueError, match="n_bins"):
            reliability_bins([0.5], [1], n_bins=0)

    def test_perfectly_calibrated_has_low_ece(self):
        rng = np.random.default_rng(0)
        p = np.linspace(0.02, 0.98, 5000)
        y = (rng.random(5000) < p).astype(int)
        assert expected_calibration_error(p, y, n_bins=10) < 0.03

    def test_constant_wrong_confidence_has_high_ece(self):
        # Always predict 0.9 on all-negative data: gap is ~0.9.
        ece = expected_calibration_error([0.9] * 100, [0] * 100)
        assert ece == pytest.approx(0.9)

    def test_brier_matches_closed_form(self):
        # (0.8-1)^2 + (0.3-0)^2 over 2 = (0.04 + 0.09)/2.
        assert brier_score([0.8, 0.3], [1, 0]) == pytest.approx(0.065)

    def test_bool_labels_accepted(self):
        assert brier_score([0.5, 0.5], [True, False]) == pytest.approx(0.25)


class TestIsotonicCalibrator:
    def test_pav_is_monotone_non_decreasing(self):
        cal = IsotonicCalibrator.fit(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0, 0, 1, 0, 1, 1],
        )
        assert list(cal.y_values) == sorted(cal.y_values)

    def test_pav_pools_violators_to_weighted_mean(self):
        # Labels [0,0,1,0,1,1]: the 1@0.3 then 0@0.4 violate monotonicity, so
        # PAV pools indices {2,3} to their mean 0.5; the least-squares isotonic
        # fit is exactly [0, 0, 0.5, 0.5, 1, 1].
        cal = IsotonicCalibrator.fit(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0, 0, 1, 0, 1, 1],
        )
        assert list(cal.y_values) == pytest.approx([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])

    def test_duplicate_scores_are_merged(self):
        cal = IsotonicCalibrator.fit([0.5, 0.5, 0.5], [1, 0, 1])
        # One knot at 0.5 with the averaged label 2/3.
        assert cal.x_thresholds == (0.5,)
        assert cal.y_values[0] == pytest.approx(2.0 / 3.0)

    def test_transform_interpolates_and_clamps(self):
        cal = IsotonicCalibrator.fit([0.2, 0.8], [0, 1])
        # Midpoint interpolates; outside the range clamps to the endpoints.
        assert cal.transform([0.5]) == [pytest.approx(0.5)]
        assert cal.transform([0.0, 1.0]) == [pytest.approx(0.0), pytest.approx(1.0)]

    def test_transform_rejects_non_finite(self):
        cal = IsotonicCalibrator.fit([0.2, 0.8], [0, 1])
        with pytest.raises(ValueError, match="finite"):
            cal.transform([float("inf")])

    def test_calibration_reduces_ece_on_miscalibrated_scores(self):
        rng = np.random.default_rng(2)
        p = np.linspace(0.02, 0.98, 4000)
        y = (rng.random(4000) < p).astype(int)
        overconfident = np.clip(p**2, 0.0, 1.0)
        cal = IsotonicCalibrator.fit(overconfident, y)
        recalibrated = cal.transform(overconfident)
        assert expected_calibration_error(recalibrated, y) < expected_calibration_error(
            overconfident, y
        )


class TestPlattCalibrator:
    def test_separable_data_stays_finite(self):
        # Perfectly separable data would push a naive sigmoid to infinite slope;
        # the regularised targets + line search keep the fit finite.
        s = np.concatenate([np.full(200, 0.1), np.full(200, 0.9)])
        y = np.concatenate([np.zeros(200, int), np.ones(200, int)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cal = PlattCalibrator.fit(s, y)
        assert np.isfinite(cal.a) and np.isfinite(cal.b)
        low, _mid, high = cal.transform([0.1, 0.5, 0.9])
        assert low < 0.01 and high > 0.99

    def test_transform_handles_extreme_scores_without_overflow(self):
        cal = PlattCalibrator.fit([0.2, 0.8], [0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = cal.transform([1e6, -1e6])
        assert out[0] in (0.0, 1.0) or 0.0 <= out[0] <= 1.0
        assert 0.0 <= out[1] <= 1.0

    def test_transform_rejects_non_finite(self):
        cal = PlattCalibrator.fit([0.2, 0.8], [0, 1])
        with pytest.raises(ValueError, match="finite"):
            cal.transform([float("nan")])

    def test_monotone_decreasing_in_negative_slope(self):
        # Higher raw score -> higher grounded probability -> negative a.
        rng = np.random.default_rng(3)
        good = np.clip(rng.normal(0.85, 0.1, 1000), 0, 1)
        bad = np.clip(rng.normal(0.15, 0.1, 1000), 0, 1)
        s = np.concatenate([good, bad])
        y = np.concatenate([np.ones(1000, int), np.zeros(1000, int)])
        cal = PlattCalibrator.fit(s, y)
        assert cal.a < 0.0
        probs = cal.transform([0.1, 0.5, 0.9])
        assert probs[0] < probs[1] < probs[2]

    def test_calibration_reduces_ece(self):
        rng = np.random.default_rng(4)
        good = np.clip(rng.normal(0.8, 0.15, 2000), 0, 1)
        bad = np.clip(rng.normal(0.2, 0.15, 2000), 0, 1)
        s = np.concatenate([good, bad])
        y = np.concatenate([np.ones(2000, int), np.zeros(2000, int)])
        cal = PlattCalibrator.fit(s, y)
        recalibrated = cal.transform(s)
        assert expected_calibration_error(recalibrated, y) < expected_calibration_error(
            s, y
        )
