# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — sustainability policy adapter tests

from __future__ import annotations

import pytest

from director_ai.core.guard_control import GuardDecision, RiskEnvelope
from director_ai.core.sustainability import (
    HardwareProfile,
    HardwareProfileRegistry,
    SustainabilityEstimate,
    SustainabilityPolicyAdapter,
    SustainabilityTelemetry,
    SustainabilityTelemetrySummary,
    TokenEnergyCostEstimator,
)


def _risk_envelope(
    *,
    action_category: str = "text",
    domain: str = "general",
    reversibility: str = "reversible",
) -> RiskEnvelope:
    return RiskEnvelope(
        action_category=action_category,
        reversibility=reversibility,
        domain=domain,
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    )


def _adapter(
    *,
    energy_kwh_per_1k_tokens: float = 0.001,
    carbon_kg_per_kwh: float = 0.2,
    carbon_defer_kg: float = 1.0,
) -> SustainabilityPolicyAdapter:
    profile = HardwareProfile(
        profile_id="edge-gpu-a",
        energy_kwh_per_1k_tokens=energy_kwh_per_1k_tokens,
        carbon_kg_per_kwh=carbon_kg_per_kwh,
        provenance="measured",
    )
    estimator = TokenEnergyCostEstimator(
        hardware_profile=profile,
        cost_per_1k_tokens=0.004,
    )
    return SustainabilityPolicyAdapter(
        estimator=estimator,
        policy_id="policy.sustainability.production",
        carbon_defer_kg=carbon_defer_kg,
        forecast_headroom_ratio=0.2,
    )


def test_estimator_discloses_profile_and_estimate_provenance() -> None:
    for provenance in ("measured", "configured", "projected"):
        profile = HardwareProfile(
            profile_id=f"gpu-{provenance}",
            energy_kwh_per_1k_tokens=0.002,
            carbon_kg_per_kwh=0.1,
            provenance=provenance,
        )
        estimator = TokenEnergyCostEstimator(
            hardware_profile=profile,
            cost_per_1k_tokens=0.01,
        )

        estimate = estimator.estimate(input_tokens=700, output_tokens=300)

        assert estimate.provenance == provenance
        assert estimate.total_tokens == 1000
        assert estimate.energy_kwh == pytest.approx(0.002)
        assert estimate.carbon_kg == pytest.approx(0.0002)
        assert estimate.cost == pytest.approx(0.01)
        assert estimate.to_dict()["hardware_profile_id"] == f"gpu-{provenance}"
        assert estimator.hardware_profile == profile
        assert estimator.cost_per_1k_tokens == pytest.approx(0.01)


def test_estimator_rejects_invalid_profiles_and_token_counts() -> None:
    with pytest.raises(ValueError, match="provenance"):
        HardwareProfile(
            profile_id="gpu",
            energy_kwh_per_1k_tokens=0.001,
            carbon_kg_per_kwh=0.2,
            provenance="guessed",
        )
    with pytest.raises(ValueError, match="energy_kwh_per_1k_tokens"):
        HardwareProfile(
            profile_id="gpu",
            energy_kwh_per_1k_tokens=-0.001,
            carbon_kg_per_kwh=0.2,
            provenance="measured",
        )
    estimator = _adapter().estimator
    with pytest.raises(ValueError, match="input_tokens"):
        estimator.estimate(input_tokens=-1, output_tokens=1)


def test_hardware_profile_registry_tracks_profiles_without_payloads() -> None:
    registry = HardwareProfileRegistry()
    profile = HardwareProfile(
        profile_id="gpu-a",
        energy_kwh_per_1k_tokens=0.002,
        carbon_kg_per_kwh=0.1,
        provenance="configured",
    )

    registry.register(profile)

    assert registry.get("gpu-a") == profile
    assert HardwareProfileRegistry((profile,)).get("gpu-a") == profile
    assert registry.snapshot()["gpu-a"]["provenance"] == "configured"
    assert "prompt" not in repr(registry.snapshot()).lower()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(profile)
    with pytest.raises(KeyError, match="missing"):
        registry.get("missing")


def test_quota_exhaustion_halts_normal_action_as_policy_signal() -> None:
    decision = _adapter().evaluate(
        tenant_id="tenant-a",
        input_tokens=800,
        output_tokens=300,
        quota_remaining_tokens=1000,
        forecast_next_tokens=100,
        risk_envelope=_risk_envelope(),
    )

    assert isinstance(decision, GuardDecision)
    assert decision.decision == "halt"
    assert decision.reason == "sustainability_quota_exceeded"
    assert decision.verifier_signals[0].modality == "sustainability"
    assert decision.verifier_signals[0].verdict == "policy_halt"
    assert "prompt" not in repr(decision.to_dict()).lower()


def test_forecast_headroom_warns_before_quota_exhaustion() -> None:
    decision = _adapter().evaluate(
        tenant_id="tenant-a",
        input_tokens=250,
        output_tokens=250,
        quota_remaining_tokens=1000,
        forecast_next_tokens=900,
        risk_envelope=_risk_envelope(),
    )

    assert decision.decision == "warn"
    assert decision.reason == "sustainability_forecast_headroom"
    assert decision.attributes["forecast_next_units"] == "900"


def test_high_carbon_deferral_is_warn_not_factual_allow() -> None:
    decision = _adapter(energy_kwh_per_1k_tokens=1.0, carbon_defer_kg=0.3).evaluate(
        tenant_id="tenant-a",
        input_tokens=1000,
        output_tokens=1000,
        quota_remaining_tokens=10_000,
        forecast_next_tokens=0,
        risk_envelope=_risk_envelope(),
    )

    assert decision.decision == "warn"
    assert decision.reason == "sustainability_high_carbon_defer"
    assert decision.attributes["recommended_action"] == "defer"
    assert decision.verifier_signals[0].modality == "sustainability"


def test_high_risk_actions_are_not_blocked_solely_for_cost_or_carbon() -> None:
    decision = _adapter(energy_kwh_per_1k_tokens=10.0, carbon_defer_kg=0.1).evaluate(
        tenant_id="tenant-a",
        input_tokens=6000,
        output_tokens=6000,
        quota_remaining_tokens=1000,
        forecast_next_tokens=0,
        risk_envelope=_risk_envelope(action_category="physical", domain="physical"),
    )

    assert decision.decision == "warn"
    assert decision.reason == "sustainability_high_risk_not_blocked"
    assert decision.attributes["override_basis"] == "high_risk_safety_action"
    assert (
        decision.to_safety_event(
            hook_id="sustainability.policy",
            hook_scope="agent",
            tenant_id="tenant-a",
        ).policy_decision
        == "warn"
    )


def test_per_tenant_telemetry_summary_alerts_without_payloads() -> None:
    telemetry = SustainabilityTelemetry(
        token_alert_threshold=1000,
        cost_alert_threshold=1.0,
        carbon_alert_threshold=0.5,
    )
    telemetry.record(
        "tenant-a",
        SustainabilityEstimate(
            input_tokens=700,
            output_tokens=500,
            total_tokens=1200,
            energy_kwh=0.1,
            carbon_kg=0.6,
            cost=1.2,
            provenance="configured",
            hardware_profile_id="gpu-a",
        ),
    )
    telemetry.record(
        "tenant-b",
        SustainabilityEstimate(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            energy_kwh=0.01,
            carbon_kg=0.02,
            cost=0.03,
            provenance="projected",
            hardware_profile_id="gpu-b",
        ),
    )

    summary = telemetry.summary("tenant-a")
    other = telemetry.summary("tenant-b")
    rendered = repr(summary.to_dict())

    assert summary.total_tokens == 1200
    assert summary.alerts == ("token_threshold", "cost_threshold", "carbon_threshold")
    assert other.alerts == ()
    assert "tenant-b" not in rendered
    assert "prompt" not in rendered.lower()
    assert "raw" not in rendered.lower()


def test_allow_decision_exposes_numeric_sustainability_accounting() -> None:
    decision = _adapter().evaluate(
        tenant_id="tenant-a",
        input_tokens=100,
        output_tokens=50,
        quota_remaining_tokens=10_000,
        forecast_next_tokens=100,
        risk_envelope=_risk_envelope(),
        evidence_ref="sustainability://tenant-a/request-1",
    )

    assert decision.decision == "allow"
    assert decision.reason == "sustainability_within_budget"
    assert decision.risk_score == 0.0
    assert decision.evidence_refs == ("sustainability://tenant-a/request-1",)
    assert decision.attributes["input_units"] == "100"
    assert decision.attributes["output_units"] == "50"
    assert decision.attributes["total_units"] == "150"
    assert "recommended_action" not in decision.attributes
    assert decision.verifier_signals[0].verdict == "policy_allow"


def test_high_risk_domain_and_irreversible_actions_warn_instead_of_halting() -> None:
    for envelope in (
        _risk_envelope(domain="medical"),
        _risk_envelope(action_category="training"),
        _risk_envelope(reversibility="irreversible"),
    ):
        decision = _adapter().evaluate(
            tenant_id="tenant-a",
            input_tokens=2000,
            output_tokens=2000,
            quota_remaining_tokens=1000,
            forecast_next_tokens=0,
            risk_envelope=envelope,
        )

        assert decision.decision == "warn"
        assert decision.attributes["override_basis"] == "high_risk_safety_action"


@pytest.mark.parametrize(
    "estimate_kwargs",
    [
        {"input_tokens": -1},
        {"output_tokens": -1},
        {"total_tokens": 99},
        {"energy_kwh": -0.1},
        {"carbon_kg": -0.1},
        {"cost": -0.1},
        {"provenance": "untracked"},
        {"hardware_profile_id": " "},
    ],
)
def test_sustainability_estimate_rejects_invalid_accounting_fields(
    estimate_kwargs: dict[str, object],
) -> None:
    kwargs = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "energy_kwh": 0.01,
        "carbon_kg": 0.02,
        "cost": 0.03,
        "provenance": "measured",
        "hardware_profile_id": "gpu-a",
    }
    kwargs.update(estimate_kwargs)

    with pytest.raises(ValueError):
        SustainabilityEstimate(**kwargs)


def test_registry_replacement_and_empty_ids_are_guarded() -> None:
    registry = HardwareProfileRegistry()
    first = HardwareProfile(
        profile_id="gpu-a",
        energy_kwh_per_1k_tokens=0.001,
        carbon_kg_per_kwh=0.2,
        provenance="measured",
    )
    replacement = HardwareProfile(
        profile_id="gpu-a",
        energy_kwh_per_1k_tokens=0.002,
        carbon_kg_per_kwh=0.1,
        provenance="configured",
    )

    registry.register(first)
    registry.register(replacement, replace=True)

    assert registry.get("gpu-a") == replacement
    with pytest.raises(ValueError, match="profile_id"):
        registry.get(" ")
    with pytest.raises(ValueError, match="profile_id"):
        HardwareProfile(
            profile_id=" ",
            energy_kwh_per_1k_tokens=0.001,
            carbon_kg_per_kwh=0.2,
            provenance="measured",
        )


def test_telemetry_rejects_invalid_tenants_and_thresholds() -> None:
    estimate = SustainabilityEstimate(
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        energy_kwh=0.01,
        carbon_kg=0.01,
        cost=0.01,
        provenance="configured",
        hardware_profile_id="gpu-a",
    )

    with pytest.raises(ValueError, match="token_alert_threshold"):
        SustainabilityTelemetry(
            token_alert_threshold=-1,
            cost_alert_threshold=1.0,
            carbon_alert_threshold=1.0,
        )
    telemetry = SustainabilityTelemetry(
        token_alert_threshold=10,
        cost_alert_threshold=1.0,
        carbon_alert_threshold=1.0,
    )
    with pytest.raises(ValueError, match="tenant_id"):
        telemetry.record(" ", estimate)
    with pytest.raises(ValueError, match="tenant_id"):
        telemetry.summary("")
    empty = telemetry.summary("tenant-c")
    assert empty.request_count == 0
    assert empty.to_dict()["alerts"] == []

    with pytest.raises(ValueError, match="tenant_id"):
        SustainabilityTelemetrySummary(
            tenant_id=" ",
            request_count=0,
            total_tokens=0,
            energy_kwh=0.0,
            carbon_kg=0.0,
            cost=0.0,
            alerts=(),
        )
    with pytest.raises(ValueError, match="request_count"):
        SustainabilityTelemetrySummary(
            tenant_id="tenant-a",
            request_count=-1,
            total_tokens=0,
            energy_kwh=0.0,
            carbon_kg=0.0,
            cost=0.0,
            alerts=(),
        )


def test_policy_adapter_rejects_invalid_operational_inputs() -> None:
    with pytest.raises(ValueError, match="policy_id"):
        SustainabilityPolicyAdapter(
            estimator=_adapter().estimator,
            policy_id=" ",
            carbon_defer_kg=0.1,
        )
    with pytest.raises(ValueError, match="forecast_headroom_ratio"):
        SustainabilityPolicyAdapter(
            estimator=_adapter().estimator,
            policy_id="policy",
            carbon_defer_kg=0.1,
            forecast_headroom_ratio=1.1,
        )
    adapter = _adapter()
    with pytest.raises(ValueError, match="tenant_id"):
        adapter.evaluate(
            tenant_id=" ",
            input_tokens=1,
            output_tokens=1,
            quota_remaining_tokens=10,
            forecast_next_tokens=0,
            risk_envelope=_risk_envelope(),
        )
    with pytest.raises(ValueError, match="quota_remaining_tokens"):
        adapter.evaluate(
            tenant_id="tenant-a",
            input_tokens=1,
            output_tokens=1,
            quota_remaining_tokens=-1,
            forecast_next_tokens=0,
            risk_envelope=_risk_envelope(),
        )


def test_sum_helpers_use_python_fallback_when_accelerator_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from director_ai.core.sustainability import policy_adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "_RUST_SUSTAINABILITY_POLICY", False)

    assert adapter_mod._sum_int([1, 2, 3]) == 6
    assert adapter_mod._sum_float([1.0, 2.5, 3.5]) == 7.0
