# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — federated safety signal sharing tests

from __future__ import annotations

import pytest

from director_ai.core.federated_privacy import PrivacyAccountant
from director_ai.core.federated_privacy.signal_sharing import (
    FederatedSafetySignalAggregator,
    FederatedSafetySignalRelease,
)
from director_ai.core.safety_event import SafetyEvent
from director_ai.core.safety_protocol import director_safety_signal_from_event


def _signal(
    *,
    tenant_id: str,
    decision: str = "halt",
    scope: str = "streaming",
    signal_id: str | None = None,
):
    event = SafetyEvent.from_policy_decision(
        hook_id=f"{scope}.guard",
        hook_scope=scope,
        policy_decision=decision,
        halt_reason="policy_threshold",
        tenant_safe_explanation="Tenant-safe safety signal.",
        tenant_id=tenant_id,
        observed_score=0.8,
        threshold=0.5,
        evidence_refs=("policy:threshold",),
    )
    return director_safety_signal_from_event(
        event,
        producer_id="producer-a",
        framework="test",
        signal_id=signal_id,
    )


def test_signal_release_includes_raw_counts_only_when_requested() -> None:
    release = FederatedSafetySignalRelease(
        noisy_counts={"decision:halt": 1.25},
        epsilon_spent=0.7,
        categories=("decision:halt",),
        signal_count=2,
        distinct_tenants=2,
        raw_counts={"decision:halt": 2},
    )

    public_payload = release.to_dict()
    audit_payload = release.to_dict(include_raw=True)

    assert "raw_counts" not in public_payload
    assert public_payload["privacy"]["raw_payload_included"] is False
    assert audit_payload["raw_counts"] == {"decision:halt": 2}
    assert audit_payload["privacy"]["raw_payload_included"] is True


def test_signal_aggregator_validates_categories_and_minimum_tenants() -> None:
    with pytest.raises(ValueError, match="min_tenants"):
        FederatedSafetySignalAggregator(epsilon=0.5, min_tenants=0)
    with pytest.raises(ValueError, match="categories must be non-empty"):
        FederatedSafetySignalAggregator(epsilon=0.5, categories=())
    with pytest.raises(ValueError, match="categories must be unique"):
        FederatedSafetySignalAggregator(
            epsilon=0.5,
            categories=("decision:halt", "decision:halt"),
        )
    with pytest.raises(ValueError, match="invalid category"):
        FederatedSafetySignalAggregator(epsilon=0.5, categories=("Bad Category",))


def test_signal_aggregator_rejects_blank_tenant_id() -> None:
    aggregator = FederatedSafetySignalAggregator(
        epsilon=0.5,
        min_tenants=1,
        seed=0,
        allow_insecure_seed=True,
    )

    with pytest.raises(ValueError, match="tenant_id is required"):
        aggregator.submit_signal(_signal(tenant_id=" "))


def test_signal_aggregator_deduplicates_signal_ids_before_release() -> None:
    aggregator = FederatedSafetySignalAggregator(
        epsilon=0.5,
        min_tenants=1,
        seed=0,
        allow_insecure_seed=True,
    )
    signal = _signal(tenant_id="tenant-a", signal_id="dsp_fixed")

    assert aggregator.submit_signal(signal) == ("decision:halt", "scope:streaming")
    assert aggregator.submit_signal(signal) == ()
    release = aggregator.release()

    assert release.signal_count == 1
    assert release.distinct_tenants == 1
    assert release.raw_counts["decision:halt"] == 1
    assert release.raw_counts["scope:streaming"] == 1


def test_signal_aggregator_ignores_categories_outside_release_contract() -> None:
    aggregator = FederatedSafetySignalAggregator(
        epsilon=0.5,
        categories=("decision:halt",),
        min_tenants=1,
        seed=0,
        allow_insecure_seed=True,
    )

    accepted = aggregator.submit_signal(_signal(tenant_id="tenant-a", decision="halt"))
    release = aggregator.release()

    assert accepted == ("decision:halt",)
    assert release.categories == ("decision:halt",)
    assert "scope:streaming" not in release.raw_counts


def test_signal_aggregator_validates_transport_payloads() -> None:
    aggregator = FederatedSafetySignalAggregator(
        epsilon=0.5,
        min_tenants=1,
        seed=0,
        allow_insecure_seed=True,
    )
    payload = _signal(tenant_id="tenant-a", decision="warn").to_transport_dict()

    assert aggregator.submit_transport(payload) == ("decision:warn", "scope:streaming")
    release = aggregator.release()
    assert release.signal_count == 1
    assert release.raw_counts["decision:warn"] == 1


def test_signal_aggregator_reset_drops_pending_window_without_budget_charge() -> None:
    accountant = PrivacyAccountant(max_epsilon=2.0)
    aggregator = FederatedSafetySignalAggregator(
        epsilon=0.5,
        accountant=accountant,
        min_tenants=1,
        seed=0,
        allow_insecure_seed=True,
    )
    aggregator.submit_signal(_signal(tenant_id="tenant-a", decision="halt"))

    aggregator.reset()

    with pytest.raises(ValueError, match="min_tenants"):
        aggregator.release()
    assert accountant.cumulative_epsilon() == 0.0
