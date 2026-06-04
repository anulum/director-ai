# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Dedicated tests for conformal uncertainty routing."""

from __future__ import annotations

import pytest

import director_ai
from director_ai.core import ConformalRoutingPolicy as CoreConformalRoutingPolicy
from director_ai.core.calibration.conformal import (
    ConformalPredictor,
    ConformalRoutingDecision,
    ConformalRoutingPolicy,
    PredictionInterval,
)


def _interval(
    *,
    lower: float,
    upper: float,
    reliable: bool = True,
    calibration_size: int = 40,
) -> PredictionInterval:
    return PredictionInterval(
        point_estimate=(lower + upper) / 2.0,
        lower=lower,
        upper=upper,
        coverage=0.95,
        calibration_size=calibration_size,
        is_reliable=reliable,
    )


class TestConformalRoutingPolicy:
    def test_invalid_threshold_order_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="allow_max_risk"):
            ConformalRoutingPolicy(
                allow_max_risk=0.2,
                escalate_min_risk=0.1,
                reject_min_risk=0.7,
            )

    def test_min_samples_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="min_samples"):
            ConformalRoutingPolicy(min_samples=0)

    def test_unreliable_interval_routes_to_human_review(self) -> None:
        decision = ConformalRoutingPolicy().decide(
            _interval(lower=0.0, upper=0.01, reliable=False, calibration_size=0)
        )

        assert decision.action == "human_review"
        assert decision.reason == "unreliable_calibration"
        assert decision.route_to == "human_review"

    def test_policy_min_samples_overrides_interval_reliability(self) -> None:
        decision = ConformalRoutingPolicy(min_samples=50).decide(
            _interval(lower=0.0, upper=0.01, reliable=True, calibration_size=30)
        )

        assert decision.action == "human_review"
        assert decision.reason == "unreliable_calibration"

    def test_low_upper_bound_allows_current_path(self) -> None:
        decision = ConformalRoutingPolicy(allow_max_risk=0.05).decide(
            _interval(lower=0.0, upper=0.04)
        )

        assert decision.action == "allow"
        assert decision.reason == "upper_risk_within_allowance"
        assert decision.route_to == "current_path"

    def test_high_lower_bound_rejects(self) -> None:
        decision = ConformalRoutingPolicy(reject_min_risk=0.7).decide(
            _interval(lower=0.72, upper=0.94)
        )

        assert decision.action == "reject"
        assert decision.reason == "lower_risk_exceeds_reject_threshold"
        assert decision.route_to == "reject"

    def test_high_upper_bound_escalates_to_stronger_model(self) -> None:
        decision = ConformalRoutingPolicy(escalate_min_risk=0.2).decide(
            _interval(lower=0.06, upper=0.21)
        )

        assert decision.action == "escalate"
        assert decision.reason == "upper_risk_exceeds_escalation_threshold"
        assert decision.route_to == "stronger_model"

    def test_ambiguous_mid_interval_routes_to_human_review(self) -> None:
        decision = ConformalRoutingPolicy(
            allow_max_risk=0.05,
            escalate_min_risk=0.2,
        ).decide(_interval(lower=0.04, upper=0.12))

        assert decision.action == "human_review"
        assert decision.reason == "ambiguous_calibrated_interval"

    def test_decision_carries_interval_evidence(self) -> None:
        interval = _interval(lower=0.03, upper=0.2, calibration_size=64)
        decision = ConformalRoutingPolicy().decide(interval)

        assert isinstance(decision, ConformalRoutingDecision)
        assert decision.interval is interval
        assert decision.risk_lower == interval.lower
        assert decision.risk_upper == interval.upper
        assert decision.calibration_size == 64


class TestConformalPredictorRouting:
    def test_predict_interval_returns_production_guard_tuple(self) -> None:
        predictor = ConformalPredictor(coverage=0.9, min_samples=2)
        predictor.calibrate([0.95, 0.9], [False, False])

        lower, upper = predictor.predict_interval(0.95)

        assert 0.0 <= lower <= upper <= 1.0

    def test_add_observation_updates_calibration_window(self) -> None:
        predictor = ConformalPredictor(coverage=0.9, min_samples=2)
        predictor.add_observation(0.95, correct_label=True)
        predictor.add_observation(0.9, correct_label=True)

        interval = predictor.predict(0.95)

        assert interval.calibration_size == 2
        assert interval.is_reliable
        assert interval.upper < 0.2

    def test_predictor_route_uses_supplied_policy(self) -> None:
        predictor = ConformalPredictor(coverage=0.8, min_samples=3)
        predictor.calibrate([0.99, 0.98, 0.97], [False, False, False])
        policy = ConformalRoutingPolicy(allow_max_risk=0.05, min_samples=3)

        decision = predictor.route(0.99, policy)

        assert decision.action == "allow"
        assert decision.coverage == pytest.approx(0.8)
        assert decision.is_reliable

    def test_predictor_route_defaults_to_predictor_min_samples(self) -> None:
        predictor = ConformalPredictor(coverage=0.8, min_samples=3)
        predictor.calibrate([0.4, 0.35, 0.3], [True, True, True])

        decision = predictor.route(0.2)

        assert decision.action in {"escalate", "reject"}
        assert decision.calibration_size == 3


class TestConformalRoutingExports:
    def test_top_level_exports_include_routing_policy(self) -> None:
        assert director_ai.ConformalRoutingPolicy is ConformalRoutingPolicy
        assert director_ai.ConformalRoutingDecision is ConformalRoutingDecision

    def test_core_exports_include_routing_policy(self) -> None:
        assert CoreConformalRoutingPolicy is ConformalRoutingPolicy
