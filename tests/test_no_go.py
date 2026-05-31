# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — no-go policy tests

from __future__ import annotations

import pytest

from director_ai.core.guard_control.decision import GuardDecision, RiskEnvelope
from director_ai.core.guard_control.no_go import (
    NoGoPolicy,
    ReviewedIrreversibilityThreshold,
)
from director_ai.core.irreversibility import Forecast


def _decision(
    *,
    risk_score: float = 0.7,
    action_category: str = "text",
    reversibility: str = "reversible",
    domain: str = "general",
    calibrated_threshold: float = 0.5,
    no_go_threshold: float = 0.95,
    attributes: dict[str, str] | None = None,
) -> GuardDecision:
    return GuardDecision(
        decision="warn",
        risk_score=risk_score,
        confidence_low=0.61,
        confidence_high=0.79,
        policy_id="policy.no-go",
        reason="operator_review",
        tenant_safe_explanation="The action needs operator review.",
        evidence_refs=("risk:forecast",),
        verifier_signals=(),
        risk_envelope=RiskEnvelope(
            action_category=action_category,
            reversibility=reversibility,
            domain=domain,
            calibrated_threshold=calibrated_threshold,
            no_go_threshold=no_go_threshold,
        ),
        attributes=attributes or {},
    )


class RecordingForecaster:
    def __init__(self, forecast: Forecast) -> None:
        self.forecast_result = forecast
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def forecast(self, actions, *, seed=0):
        self.calls.append((tuple(actions), seed))
        return self.forecast_result


def test_reviewed_irreversibility_threshold_validates_required_provenance() -> None:
    with pytest.raises(ValueError, match="source_ref is required"):
        ReviewedIrreversibilityThreshold(
            threshold=0.5,
            source_ref=" ",
            reviewer_id="reviewer-a",
            calibration_size=10,
            coverage=0.95,
        )
    with pytest.raises(ValueError, match="reviewer_id is required"):
        ReviewedIrreversibilityThreshold(
            threshold=0.5,
            source_ref="calibration://run",
            reviewer_id=" ",
            calibration_size=10,
            coverage=0.95,
        )
    with pytest.raises(ValueError, match="calibration_size must be positive"):
        ReviewedIrreversibilityThreshold(
            threshold=0.5,
            source_ref="calibration://run",
            reviewer_id="reviewer-a",
            calibration_size=0,
            coverage=0.95,
        )
    with pytest.raises(ValueError, match="must be approved"):
        ReviewedIrreversibilityThreshold(
            threshold=0.5,
            source_ref="calibration://run",
            reviewer_id="reviewer-a",
            calibration_size=10,
            coverage=0.95,
            approved=False,
        )


def test_reviewed_irreversibility_threshold_formats_telemetry_attributes() -> None:
    threshold = ReviewedIrreversibilityThreshold(
        threshold=0.6123459,
        source_ref="calibration://irreversibility/2026-05-31",
        reviewer_id="reviewer-a",
        calibration_size=123,
        coverage=0.951111,
    )

    assert threshold.to_attributes() == {
        "reviewed_threshold": "0.612346",
        "reviewed_threshold_source": "calibration://irreversibility/2026-05-31",
        "reviewed_threshold_reviewer": "reviewer-a",
        "reviewed_threshold_calibration_size": "123",
        "reviewed_threshold_coverage": "0.951111",
    }


def test_no_go_policy_rejects_invalid_forecast_configuration() -> None:
    with pytest.raises(ValueError, match="forecast_seed must be an int"):
        NoGoPolicy(forecast_seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="forecast_action_keys"):
        NoGoPolicy(forecast_action_keys=())
    with pytest.raises(ValueError, match="forecast_action_keys"):
        NoGoPolicy(forecast_action_keys=("action_sequence", " "))


def test_no_go_policy_respects_disabled_irreversibility_forecast() -> None:
    forecaster = RecordingForecaster(
        Forecast(
            p_irreversible=1.0,
            ci_low=1.0,
            ci_high=1.0,
            crossed=10,
            samples=10,
        )
    )
    policy = NoGoPolicy(
        default_threshold=0.9,
        irreversibility_forecaster=forecaster,
        enable_irreversibility_forecast=False,
    )

    verdict = policy.evaluate(
        _decision(
            risk_score=0.7,
            attributes={"action_sequence": "destroy audited state"},
        )
    )

    assert verdict.decision == "warn"
    assert verdict.forecast is None
    assert forecaster.calls == []


def test_no_go_policy_uses_first_non_empty_configured_action_key() -> None:
    forecaster = RecordingForecaster(
        Forecast(
            p_irreversible=0.1,
            ci_low=0.1,
            ci_high=0.2,
            crossed=1,
            samples=10,
        )
    )
    policy = NoGoPolicy(
        default_threshold=0.9,
        irreversibility_forecaster=forecaster,
        forecast_seed=17,
        forecast_action_keys=("missing", "action_description", "tool_action"),
    )

    verdict = policy.evaluate(
        _decision(
            risk_score=0.7,
            attributes={
                "action_description": "  prepare rollback  \n\n run migration ",
                "tool_action": "ignored once action_description is present",
            },
        )
    )

    assert verdict.decision == "warn"
    assert forecaster.calls == [(("prepare rollback", "run migration"), 17)]


def test_no_go_policy_skips_forecast_when_actions_are_absent() -> None:
    forecaster = RecordingForecaster(
        Forecast(
            p_irreversible=1.0,
            ci_low=1.0,
            ci_high=1.0,
            crossed=10,
            samples=10,
        )
    )
    policy = NoGoPolicy(
        default_threshold=0.9,
        irreversibility_forecaster=forecaster,
    )

    verdict = policy.evaluate(_decision(risk_score=0.7, attributes={"tool_action": "  "}))

    assert verdict.decision == "warn"
    assert verdict.forecast is None
    assert forecaster.calls == []


def test_no_go_policy_blocks_on_lower_of_default_and_envelope_thresholds() -> None:
    policy = NoGoPolicy(default_threshold=0.9, enable_irreversibility_forecast=False)

    verdict = policy.evaluate(
        _decision(
            risk_score=0.72,
            no_go_threshold=0.7,
        )
    )

    assert verdict.decision == "block"
    assert verdict.reason == "no_go_threshold_exceeded"
    assert verdict.requires_human_review is True
