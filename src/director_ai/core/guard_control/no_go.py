# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — no-go policy

"""Deterministic no-go policy for high-risk guard decisions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from director_ai.core.irreversibility import Forecast, IrreversibilityForecaster

from .decision import GuardDecision


class _IrreversibilityForecasterLike(Protocol):
    def forecast(
        self,
        actions: Sequence[str],
        *,
        seed: int = 0,
    ) -> Forecast: ...


@dataclass(frozen=True)
class NoGoVerdict:
    """Result of applying :class:`NoGoPolicy` to a guard decision."""

    decision: str
    reason: str
    requires_human_review: bool
    original_decision: GuardDecision
    forecast: Forecast | None = None


class NoGoPolicy:
    """Block irreversible or threshold-exceeding decisions deterministically.

    The default path also evaluates tenant-safe action labels through the
    irreversibility forecaster once the upstream decision crosses the calibrated
    risk threshold. The policy blocks only on the conservative lower confidence
    bound, so uncertain forecasts escalate through review instead of being
    treated as proof.
    """

    def __init__(
        self,
        *,
        default_threshold: float = 0.9,
        irreversible_threshold: float = 0.6,
        require_human_review_for_irreversible: bool = True,
        irreversibility_forecaster: _IrreversibilityForecasterLike | None = None,
        forecast_seed: int = 0,
        forecast_action_keys: Sequence[str] = (
            "action_sequence",
            "action_description",
            "proposed_action",
            "tool_action",
            "physical_action",
        ),
        enable_irreversibility_forecast: bool = True,
    ) -> None:
        _validate_threshold("default_threshold", default_threshold)
        _validate_threshold("irreversible_threshold", irreversible_threshold)
        if not isinstance(forecast_seed, int):
            raise ValueError("forecast_seed must be an int")
        action_keys = tuple(key.strip() for key in forecast_action_keys)
        if not action_keys or any(not key for key in action_keys):
            raise ValueError("forecast_action_keys must contain non-empty keys")
        self._default_threshold = default_threshold
        self._irreversible_threshold = irreversible_threshold
        self._review_irreversible = require_human_review_for_irreversible
        self._forecast_seed = forecast_seed
        self._forecast_action_keys = action_keys
        self._forecaster: _IrreversibilityForecasterLike | None
        if enable_irreversibility_forecast:
            self._forecaster = irreversibility_forecaster or IrreversibilityForecaster()
        else:
            self._forecaster = None

    def evaluate(self, decision: GuardDecision) -> NoGoVerdict:
        """Return the final deterministic no-go verdict."""
        envelope = decision.risk_envelope
        if (
            envelope.reversibility == "irreversible"
            and decision.risk_score >= self._irreversible_threshold
        ):
            return NoGoVerdict(
                decision="block",
                reason="no_go_irreversible_risk",
                requires_human_review=self._review_irreversible,
                original_decision=decision,
            )
        forecast = self._forecast_irreversibility(decision)
        if forecast is not None and forecast.ci_low >= self._irreversible_threshold:
            return NoGoVerdict(
                decision="block",
                reason="no_go_irreversibility_forecast",
                requires_human_review=True,
                original_decision=decision,
                forecast=forecast,
            )
        threshold = min(self._default_threshold, envelope.no_go_threshold)
        if decision.risk_score >= threshold:
            return NoGoVerdict(
                decision="block",
                reason="no_go_threshold_exceeded",
                requires_human_review=True,
                original_decision=decision,
            )
        return NoGoVerdict(
            decision=decision.decision,
            reason=decision.reason,
            requires_human_review=False,
            original_decision=decision,
            forecast=forecast,
        )

    def _forecast_irreversibility(self, decision: GuardDecision) -> Forecast | None:
        if self._forecaster is None:
            return None
        if decision.risk_score < decision.risk_envelope.calibrated_threshold:
            return None
        actions = _tenant_safe_action_sequence(
            decision,
            action_keys=self._forecast_action_keys,
        )
        if not actions:
            return None
        return self._forecaster.forecast(actions, seed=self._forecast_seed)


def _tenant_safe_action_sequence(
    decision: GuardDecision,
    *,
    action_keys: Sequence[str],
) -> tuple[str, ...]:
    for key in action_keys:
        value = decision.attributes.get(key)
        if value is None:
            continue
        actions = tuple(
            line.strip() for line in str(value).splitlines() if line.strip()
        )
        if actions:
            return actions
    return ()


def _validate_threshold(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
