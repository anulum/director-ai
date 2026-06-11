# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — UncertaintyRouter tests

"""Tests for uncertainty-aware routing of conformal intervals.

Covers each action band (confidently-low allow, confidently-high reject,
wide-uncertain human escalation, narrow-uncertain model escalation), the
unreliable-calibration override to human review, width computation, the
audit reason text, and threshold validation."""

from __future__ import annotations

import pytest

from director_ai.core.calibration.conformal import PredictionInterval
from director_ai.core.routing import UncertaintyRouter


def _interval(lower, upper, *, is_reliable=True, n=50):
    return PredictionInterval(
        point_estimate=(lower + upper) / 2,
        lower=lower,
        upper=upper,
        coverage=0.9,
        calibration_size=n,
        is_reliable=is_reliable,
    )


class TestBands:
    def test_confidently_low_allows(self):
        decision = UncertaintyRouter().route(_interval(0.02, 0.15))
        assert decision.action == "allow"
        assert "allow_upper" in decision.reason

    def test_confidently_high_rejects(self):
        decision = UncertaintyRouter().route(_interval(0.85, 0.97))
        assert decision.action == "reject"
        assert "reject_lower" in decision.reason

    def test_wide_uncertain_escalates_to_human(self):
        decision = UncertaintyRouter().route(_interval(0.1, 0.9))
        assert decision.action == "escalate_human"
        assert decision.width == pytest.approx(0.8)

    def test_narrow_uncertain_escalates_to_model(self):
        decision = UncertaintyRouter().route(_interval(0.4, 0.6))
        assert decision.action == "escalate_model"
        assert decision.width == pytest.approx(0.2)

    def test_unreliable_calibration_defers_to_human(self):
        decision = UncertaintyRouter().route(
            _interval(0.02, 0.05, is_reliable=False, n=3)
        )
        assert decision.action == "escalate_human"
        assert "unreliable" in decision.reason

    def test_decision_carries_interval_fields(self):
        decision = UncertaintyRouter().route(_interval(0.3, 0.7))
        assert decision.lower == 0.3
        assert decision.upper == 0.7
        assert decision.point_estimate == pytest.approx(0.5)
        assert decision.is_reliable is True


class TestThresholds:
    def test_custom_thresholds_shift_bands(self):
        router = UncertaintyRouter(
            allow_upper=0.3, reject_lower=0.6, escalate_human_width=0.1
        )
        assert router.route(_interval(0.1, 0.25)).action == "allow"
        assert router.route(_interval(0.65, 0.7)).action == "reject"
        # 0.35–0.5 uncertain, width 0.15 >= 0.1 → human
        assert router.route(_interval(0.35, 0.5)).action == "escalate_human"

    def test_allow_upper_must_be_below_reject_lower(self):
        with pytest.raises(ValueError, match="allow_upper must be < reject_lower"):
            UncertaintyRouter(allow_upper=0.8, reject_lower=0.5)

    def test_allow_upper_out_of_range(self):
        with pytest.raises(ValueError, match="allow_upper"):
            UncertaintyRouter(allow_upper=1.5)

    def test_escalate_width_out_of_range(self):
        with pytest.raises(ValueError, match="escalate_human_width"):
            UncertaintyRouter(escalate_human_width=0.0)


class TestProductionGuardWiring:
    """ProductionGuard.check populates uncertainty_action when routing is on."""

    def test_uncertainty_action_none_without_router(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard.from_profile("fast")
        result = guard.check("What is 2+2?", "2+2 equals 4.")
        assert result.uncertainty_action is None

    def test_uncalibrated_conformal_routes_to_human(self):
        from director_ai.core.calibration.conformal import ConformalPredictor
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard.from_profile("fast")
        # Uncalibrated predictor → interval is unreliable → defer to human.
        guard._conformal = ConformalPredictor(coverage=0.9)
        guard.enable_uncertainty_routing()
        result = guard.check("What is 2+2?", "2+2 equals 4.")
        assert result.uncertainty_action == "escalate_human"
