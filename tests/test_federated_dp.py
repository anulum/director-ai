# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Federated DP Calibration Tests
"""Multi-angle tests for federated, differentially private calibration."""

from __future__ import annotations

import pytest

from director_ai.core.federated_dp import (
    CohortTooSmallError,
    FederatedCalibrationRound,
    RoundResult,
)


def _round(**kw) -> FederatedCalibrationRound:
    base = {
        "initial_value": 0.5,
        "clip_norm": 0.2,
        "noise_multiplier": 0.0,
        "min_cohort": 3,
        "learning_rate": 1.0,
        "seed": 1,
    }
    base.update(kw)
    initial = base.pop("initial_value")
    return FederatedCalibrationRound(initial, **base)


class TestValidation:
    def test_clip_norm_positive(self):
        with pytest.raises(ValueError, match="clip_norm"):
            _round(clip_norm=0.0)

    def test_noise_multiplier_non_negative(self):
        with pytest.raises(ValueError, match="noise_multiplier"):
            _round(noise_multiplier=-1.0)

    def test_min_cohort_positive(self):
        with pytest.raises(ValueError, match="min_cohort"):
            _round(min_cohort=0)

    def test_initial_value_in_bounds(self):
        with pytest.raises(ValueError, match="value_bounds"):
            _round(initial_value=2.0)

    def test_bad_tenant_rejected(self):
        r = _round(min_cohort=1)
        with pytest.raises(ValueError, match="tenant_id"):
            r.submit_update(tenant_id="bad tenant!", update=0.1)


class TestAggregation:
    def test_no_noise_mean_update(self):
        r = _round()
        for t in ("a", "b", "c"):
            r.submit_update(tenant_id=t, update=0.1)
        res = r.aggregate()
        assert res.clipped_mean == pytest.approx(0.1)
        assert res.new_value == pytest.approx(0.6)
        assert res.cohort_size == 3
        assert r.rounds_applied == 1
        assert r.cohort_size == 0  # pending updates cleared

    def test_per_tenant_clipping(self):
        r = _round(min_cohort=1, clip_norm=0.2)
        r.submit_update(tenant_id="a", update=5.0)
        assert r.aggregate().clipped_mean == pytest.approx(0.2)

    def test_negative_clipping(self):
        r = _round(min_cohort=1, clip_norm=0.2)
        r.submit_update(tenant_id="a", update=-5.0)
        assert r.aggregate().clipped_mean == pytest.approx(-0.2)

    def test_one_vote_per_tenant(self):
        r = _round(min_cohort=1, clip_norm=1.0)
        r.submit_update(tenant_id="a", update=0.1)
        r.submit_update(tenant_id="a", update=0.3)  # overwrites a's vote
        assert r.cohort_size == 1
        assert r.aggregate().clipped_mean == pytest.approx(0.3)

    def test_value_is_clamped_to_bounds(self):
        r = _round(min_cohort=1, clip_norm=1.0, value_bounds=(0.0, 0.55))
        r.submit_update(tenant_id="a", update=1.0)
        assert r.aggregate().new_value == 0.55

    def test_learning_rate_scales_update(self):
        r = _round(min_cohort=1, clip_norm=0.2, learning_rate=0.5)
        r.submit_update(tenant_id="a", update=0.2)
        # 0.5 + 0.5 * 0.2 = 0.6
        assert r.aggregate().new_value == pytest.approx(0.6)


class TestCohortGate:
    def test_too_small_cohort_refused(self):
        r = _round(min_cohort=3)
        r.submit_update(tenant_id="a", update=0.1)
        r.submit_update(tenant_id="b", update=0.1)
        with pytest.raises(CohortTooSmallError):
            r.aggregate()

    def test_refused_round_keeps_value_and_updates(self):
        r = _round(min_cohort=3)
        r.submit_update(tenant_id="a", update=0.1)
        with pytest.raises(CohortTooSmallError):
            r.aggregate()
        assert r.value == 0.5
        assert r.cohort_size == 1  # updates not cleared on refusal


class TestNoiseAndConvergence:
    def test_noise_is_deterministic_with_seed(self):
        def run(seed):
            r = _round(noise_multiplier=1.0, min_cohort=2, seed=seed)
            r.submit_update(tenant_id="a", update=0.1)
            r.submit_update(tenant_id="b", update=0.1)
            return r.aggregate().new_value

        assert run(7) == run(7)

    def test_noise_perturbs_the_update(self):
        noised = _round(noise_multiplier=2.0, min_cohort=2, seed=3)
        noised.submit_update(tenant_id="a", update=0.0)
        noised.submit_update(tenant_id="b", update=0.0)
        # With zero mean update, any movement is the DP noise.
        assert noised.aggregate().new_value != 0.5

    def test_repeated_rounds_converge_without_noise(self):
        # Every tenant pushes toward 0.8; no noise → value climbs to the bound.
        r = _round(noise_multiplier=0.0, min_cohort=2, clip_norm=0.1, learning_rate=1.0)
        for _ in range(10):
            r.submit_update(tenant_id="a", update=0.1)
            r.submit_update(tenant_id="b", update=0.1)
            r.aggregate()
        assert r.value > 0.7

    def test_round_result_to_dict(self):
        r = _round(min_cohort=1)
        r.submit_update(tenant_id="a", update=0.1)
        d = r.aggregate().to_dict()
        assert set(d) == {
            "previous_value",
            "new_value",
            "cohort_size",
            "clipped_mean",
            "noise_scale",
        }


class TestGuardWiring:
    def test_guard_federated_calibration(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        rnd = guard.federated_calibration(min_cohort=1, noise_multiplier=0.0)
        assert isinstance(rnd, FederatedCalibrationRound)
        # Seeded at the configured coherence threshold by default.
        assert rnd.value == DirectorConfig().coherence_threshold
        rnd.submit_update(tenant_id="t1", update=0.05)
        assert isinstance(rnd.aggregate(), RoundResult)
