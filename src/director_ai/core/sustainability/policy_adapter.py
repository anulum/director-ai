# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — sustainability policy adapter

"""Token, energy, cost, and carbon signals for guard-control policy."""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from director_ai.core.guard_control import GuardDecision, RiskEnvelope, VerifierSignal

EstimateProvenance = Literal["measured", "configured", "projected"]

_VALID_PROVENANCE = frozenset({"measured", "configured", "projected"})
_HIGH_RISK_ACTIONS = frozenset({"physical", "tool", "code", "training"})
_HIGH_RISK_DOMAINS = frozenset({"physical", "medical", "legal", "security"})


@dataclass(frozen=True)
class HardwareProfile:
    """Deployment hardware profile with explicit measurement provenance."""

    profile_id: str
    energy_kwh_per_1k_tokens: float
    carbon_kg_per_kwh: float
    provenance: EstimateProvenance

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must be non-empty")
        _validate_non_negative(
            "energy_kwh_per_1k_tokens", self.energy_kwh_per_1k_tokens
        )
        _validate_non_negative("carbon_kg_per_kwh", self.carbon_kg_per_kwh)
        if self.provenance not in _VALID_PROVENANCE:
            raise ValueError("provenance must be measured, configured, or projected")

    def to_dict(self) -> dict[str, str | float]:
        """Return tenant-safe profile metadata."""
        return {
            "profile_id": self.profile_id,
            "energy_kwh_per_1k_tokens": self.energy_kwh_per_1k_tokens,
            "carbon_kg_per_kwh": self.carbon_kg_per_kwh,
            "provenance": self.provenance,
        }


class HardwareProfileRegistry:
    """Thread-safe registry for deployment hardware profiles."""

    def __init__(self, profiles: tuple[HardwareProfile, ...] = ()) -> None:
        self._lock = threading.Lock()
        self._profiles: dict[str, HardwareProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: HardwareProfile, *, replace: bool = False) -> None:
        """Register a hardware profile by id."""
        with self._lock:
            if not replace and profile.profile_id in self._profiles:
                raise ValueError(
                    f"hardware profile {profile.profile_id!r} already registered"
                )
            self._profiles[profile.profile_id] = profile

    def get(self, profile_id: str) -> HardwareProfile:
        """Return a profile by id."""
        if not profile_id.strip():
            raise ValueError("profile_id must be non-empty")
        with self._lock:
            try:
                return self._profiles[profile_id]
            except KeyError as exc:
                raise KeyError(
                    f"hardware profile {profile_id!r} is not registered"
                ) from exc

    def snapshot(self) -> dict[str, dict[str, str | float]]:
        """Return tenant-safe profile metadata for all registered profiles."""
        with self._lock:
            return {
                profile_id: profile.to_dict()
                for profile_id, profile in sorted(self._profiles.items())
            }


@dataclass(frozen=True)
class SustainabilityEstimate:
    """Tenant-safe token/cost/energy/carbon estimate for one request."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    energy_kwh: float
    carbon_kg: float
    cost: float
    provenance: EstimateProvenance
    hardware_profile_id: str

    def __post_init__(self) -> None:
        _validate_token_count("input_tokens", self.input_tokens)
        _validate_token_count("output_tokens", self.output_tokens)
        _validate_token_count("total_tokens", self.total_tokens)
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        _validate_non_negative("energy_kwh", self.energy_kwh)
        _validate_non_negative("carbon_kg", self.carbon_kg)
        _validate_non_negative("cost", self.cost)
        if self.provenance not in _VALID_PROVENANCE:
            raise ValueError("provenance must be measured, configured, or projected")
        if not self.hardware_profile_id.strip():
            raise ValueError("hardware_profile_id must be non-empty")

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a serialisable estimate without prompt or completion payloads."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "energy_kwh": self.energy_kwh,
            "carbon_kg": self.carbon_kg,
            "cost": self.cost,
            "provenance": self.provenance,
            "hardware_profile_id": self.hardware_profile_id,
        }


class TokenEnergyCostEstimator:
    """Deterministic token-cost-energy estimator for a hardware profile."""

    def __init__(
        self,
        *,
        hardware_profile: HardwareProfile,
        cost_per_1k_tokens: float,
    ) -> None:
        _validate_non_negative("cost_per_1k_tokens", cost_per_1k_tokens)
        self._hardware_profile = hardware_profile
        self._cost_per_1k_tokens = cost_per_1k_tokens

    @property
    def hardware_profile(self) -> HardwareProfile:
        return self._hardware_profile

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._cost_per_1k_tokens

    def estimate(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> SustainabilityEstimate:
        """Estimate request resource usage from token counts."""
        _validate_token_count("input_tokens", input_tokens)
        _validate_token_count("output_tokens", output_tokens)
        total_tokens = input_tokens + output_tokens
        token_units = total_tokens / 1000.0
        energy_kwh = token_units * self._hardware_profile.energy_kwh_per_1k_tokens
        carbon_kg = energy_kwh * self._hardware_profile.carbon_kg_per_kwh
        cost = token_units * self._cost_per_1k_tokens
        return SustainabilityEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            energy_kwh=energy_kwh,
            carbon_kg=carbon_kg,
            cost=cost,
            provenance=self._hardware_profile.provenance,
            hardware_profile_id=self._hardware_profile.profile_id,
        )


@dataclass(frozen=True)
class SustainabilityTelemetrySummary:
    """Per-tenant aggregate telemetry with threshold alert names."""

    tenant_id: str
    request_count: int
    total_tokens: int
    energy_kwh: float
    carbon_kg: float
    cost: float
    alerts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if self.request_count < 0:
            raise ValueError("request_count must be non-negative")
        _validate_token_count("total_tokens", self.total_tokens)
        _validate_non_negative("energy_kwh", self.energy_kwh)
        _validate_non_negative("carbon_kg", self.carbon_kg)
        _validate_non_negative("cost", self.cost)

    def to_dict(self) -> dict[str, str | int | float | list[str]]:
        """Return a tenant-safe aggregate summary."""
        return {
            "tenant_id": self.tenant_id,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "energy_kwh": self.energy_kwh,
            "carbon_kg": self.carbon_kg,
            "cost": self.cost,
            "alerts": list(self.alerts),
        }


class SustainabilityTelemetry:
    """Thread-safe in-memory per-tenant sustainability telemetry."""

    def __init__(
        self,
        *,
        token_alert_threshold: int,
        cost_alert_threshold: float,
        carbon_alert_threshold: float,
    ) -> None:
        _validate_token_count("token_alert_threshold", token_alert_threshold)
        _validate_non_negative("cost_alert_threshold", cost_alert_threshold)
        _validate_non_negative("carbon_alert_threshold", carbon_alert_threshold)
        self._token_threshold = token_alert_threshold
        self._cost_threshold = cost_alert_threshold
        self._carbon_threshold = carbon_alert_threshold
        self._lock = threading.Lock()
        self._records: dict[str, list[SustainabilityEstimate]] = defaultdict(list)

    def record(self, tenant_id: str, estimate: SustainabilityEstimate) -> None:
        """Append one estimate for a tenant."""
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        with self._lock:
            self._records[tenant_id].append(estimate)

    def summary(self, tenant_id: str) -> SustainabilityTelemetrySummary:
        """Return aggregate telemetry and threshold alerts for one tenant."""
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        with self._lock:
            records = tuple(self._records.get(tenant_id, ()))
        total_tokens = sum(estimate.total_tokens for estimate in records)
        energy_kwh = sum(estimate.energy_kwh for estimate in records)
        carbon_kg = sum(estimate.carbon_kg for estimate in records)
        cost = sum(estimate.cost for estimate in records)
        alerts = _alerts(
            total_tokens=total_tokens,
            token_threshold=self._token_threshold,
            cost=cost,
            cost_threshold=self._cost_threshold,
            carbon_kg=carbon_kg,
            carbon_threshold=self._carbon_threshold,
        )
        return SustainabilityTelemetrySummary(
            tenant_id=tenant_id,
            request_count=len(records),
            total_tokens=total_tokens,
            energy_kwh=energy_kwh,
            carbon_kg=carbon_kg,
            cost=cost,
            alerts=alerts,
        )


class SustainabilityPolicyAdapter:
    """Convert sustainability estimates into guard-control decisions."""

    def __init__(
        self,
        *,
        estimator: TokenEnergyCostEstimator,
        policy_id: str,
        carbon_defer_kg: float,
        forecast_headroom_ratio: float = 0.1,
    ) -> None:
        if not policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        _validate_non_negative("carbon_defer_kg", carbon_defer_kg)
        if not math.isfinite(forecast_headroom_ratio) or not (
            0.0 <= forecast_headroom_ratio <= 1.0
        ):
            raise ValueError("forecast_headroom_ratio must be finite and in [0, 1]")
        self.estimator = estimator
        self._policy_id = policy_id
        self._carbon_defer_kg = carbon_defer_kg
        self._forecast_headroom_ratio = forecast_headroom_ratio

    def evaluate(
        self,
        *,
        tenant_id: str,
        input_tokens: int,
        output_tokens: int,
        quota_remaining_tokens: int,
        forecast_next_tokens: int,
        risk_envelope: RiskEnvelope,
        evidence_ref: str = "sustainability://estimate",
    ) -> GuardDecision:
        """Return a guard decision for the sustainability policy state."""
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        _validate_token_count("quota_remaining_tokens", quota_remaining_tokens)
        _validate_token_count("forecast_next_tokens", forecast_next_tokens)
        estimate = self.estimator.estimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        basis = _decision_basis(
            estimate=estimate,
            quota_remaining_tokens=quota_remaining_tokens,
            forecast_next_tokens=forecast_next_tokens,
            carbon_defer_kg=self._carbon_defer_kg,
            forecast_headroom_ratio=self._forecast_headroom_ratio,
            risk_envelope=risk_envelope,
        )
        signal = VerifierSignal(
            verifier="sustainability.policy",
            modality="sustainability",
            score=basis.signal_score,
            verdict=basis.signal_verdict,
            confidence_low=basis.confidence_low,
            confidence_high=basis.confidence_high,
            evidence_refs=(evidence_ref,),
            failure_mode=basis.reason if basis.decision != "allow" else "",
        )
        return GuardDecision(
            decision=basis.decision,
            risk_score=basis.risk_score,
            confidence_low=basis.confidence_low,
            confidence_high=basis.confidence_high,
            policy_id=self._policy_id,
            reason=basis.reason,
            tenant_safe_explanation=basis.explanation,
            evidence_refs=(evidence_ref,),
            verifier_signals=(signal,),
            risk_envelope=risk_envelope,
            attributes=_decision_attributes(
                estimate=estimate,
                quota_remaining_tokens=quota_remaining_tokens,
                forecast_next_tokens=forecast_next_tokens,
                basis=basis,
            ),
        )


@dataclass(frozen=True)
class _DecisionBasis:
    decision: str
    reason: str
    explanation: str
    risk_score: float
    signal_score: float
    signal_verdict: str
    confidence_low: float
    confidence_high: float
    recommended_action: str = ""
    override_basis: str = ""


def _decision_basis(
    *,
    estimate: SustainabilityEstimate,
    quota_remaining_tokens: int,
    forecast_next_tokens: int,
    carbon_defer_kg: float,
    forecast_headroom_ratio: float,
    risk_envelope: RiskEnvelope,
) -> _DecisionBasis:
    quota_exceeded = estimate.total_tokens > quota_remaining_tokens
    high_carbon = estimate.carbon_kg > carbon_defer_kg
    reserve_floor = quota_remaining_tokens * forecast_headroom_ratio
    forecast_headroom = estimate.total_tokens + forecast_next_tokens > max(
        0.0, quota_remaining_tokens - reserve_floor
    )
    if _is_high_risk_safety_action(risk_envelope) and (
        quota_exceeded or high_carbon or forecast_headroom
    ):
        return _DecisionBasis(
            decision="warn",
            reason="sustainability_high_risk_not_blocked",
            explanation=(
                "Sustainability limits were exceeded, but the action is high-risk "
                "and cannot be blocked solely for cost, quota, or carbon policy."
            ),
            risk_score=min(risk_envelope.calibrated_threshold, 0.49),
            signal_score=0.7,
            signal_verdict="policy_warn",
            confidence_low=0.6,
            confidence_high=0.9,
            recommended_action="continue_with_review",
            override_basis="high_risk_safety_action",
        )
    if quota_exceeded:
        return _DecisionBasis(
            decision="halt",
            reason="sustainability_quota_exceeded",
            explanation="The request would exceed the tenant sustainability quota.",
            risk_score=0.95,
            signal_score=0.95,
            signal_verdict="policy_halt",
            confidence_low=0.9,
            confidence_high=1.0,
        )
    if forecast_headroom:
        return _DecisionBasis(
            decision="warn",
            reason="sustainability_forecast_headroom",
            explanation=(
                "Forecast demand would leave insufficient sustainability quota "
                "headroom."
            ),
            risk_score=0.65,
            signal_score=0.65,
            signal_verdict="policy_warn",
            confidence_low=0.55,
            confidence_high=0.85,
            recommended_action="reduce_or_reserve_capacity",
        )
    if high_carbon:
        return _DecisionBasis(
            decision="warn",
            reason="sustainability_high_carbon_defer",
            explanation="The request falls above the configured carbon deferral budget.",
            risk_score=0.6,
            signal_score=0.6,
            signal_verdict="policy_warn",
            confidence_low=0.5,
            confidence_high=0.85,
            recommended_action="defer",
        )
    return _DecisionBasis(
        decision="allow",
        reason="sustainability_within_budget",
        explanation="The request is within configured sustainability policy limits.",
        risk_score=0.0,
        signal_score=0.0,
        signal_verdict="policy_allow",
        confidence_low=0.8,
        confidence_high=1.0,
    )


def _decision_attributes(
    *,
    estimate: SustainabilityEstimate,
    quota_remaining_tokens: int,
    forecast_next_tokens: int,
    basis: _DecisionBasis,
) -> dict[str, str]:
    attributes = {
        "input_units": str(estimate.input_tokens),
        "output_units": str(estimate.output_tokens),
        "total_units": str(estimate.total_tokens),
        "quota_remaining_units": str(quota_remaining_tokens),
        "forecast_next_units": str(forecast_next_tokens),
        "energy_kwh": f"{estimate.energy_kwh:.12g}",
        "carbon_kg": f"{estimate.carbon_kg:.12g}",
        "cost": f"{estimate.cost:.12g}",
        "provenance": estimate.provenance,
        "hardware_profile_id": estimate.hardware_profile_id,
    }
    if basis.recommended_action:
        attributes["recommended_action"] = basis.recommended_action
    if basis.override_basis:
        attributes["override_basis"] = basis.override_basis
    return attributes


def _alerts(
    *,
    total_tokens: int,
    token_threshold: int,
    cost: float,
    cost_threshold: float,
    carbon_kg: float,
    carbon_threshold: float,
) -> tuple[str, ...]:
    alerts: list[str] = []
    if total_tokens >= token_threshold:
        alerts.append("token_threshold")
    if cost >= cost_threshold:
        alerts.append("cost_threshold")
    if carbon_kg >= carbon_threshold:
        alerts.append("carbon_threshold")
    return tuple(alerts)


def _is_high_risk_safety_action(risk_envelope: RiskEnvelope) -> bool:
    return (
        risk_envelope.action_category in _HIGH_RISK_ACTIONS
        or risk_envelope.domain in _HIGH_RISK_DOMAINS
        or risk_envelope.reversibility == "irreversible"
    )


def _validate_token_count(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
