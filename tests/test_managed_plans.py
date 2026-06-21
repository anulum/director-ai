# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the managed-service plan registry and quota decisions."""

from __future__ import annotations

from director_ai.managed import Plan, PlanRegistry, QuotaDecision


def test_default_registry_exposes_three_tiers() -> None:
    reg = PlanRegistry()
    assert set(reg.names()) == {"free", "pro", "scale"}


def test_get_known_plan_returns_its_limits() -> None:
    reg = PlanRegistry()
    free = reg.get("free")
    assert isinstance(free, Plan)
    assert free.monthly_request_cap == 1_000
    assert reg.get("scale").monthly_request_cap is None


def test_unknown_plan_degrades_to_free_not_unlimited() -> None:
    reg = PlanRegistry()
    fallback = reg.get("enterprise-typo")
    assert fallback.name == "free"
    assert fallback.monthly_request_cap == 1_000


def test_decide_within_cap_allows_with_remaining() -> None:
    reg = PlanRegistry()
    decision = reg.decide("free", used_this_period=10)
    assert isinstance(decision, QuotaDecision)
    assert decision.allowed
    assert decision.limit == 1_000
    assert decision.used == 10
    assert decision.remaining == 990


def test_decide_at_cap_denies() -> None:
    reg = PlanRegistry()
    decision = reg.decide("free", used_this_period=1_000)
    assert not decision.allowed
    assert decision.remaining == 0
    assert "cap" in decision.reason


def test_decide_over_cap_denies() -> None:
    reg = PlanRegistry()
    assert not reg.decide("free", used_this_period=5_000).allowed


def test_decide_unlimited_plan_always_allows() -> None:
    reg = PlanRegistry()
    decision = reg.decide("scale", used_this_period=10_000_000)
    assert decision.allowed
    assert decision.limit is None
    assert decision.remaining is None


def test_custom_registry_overrides_caps() -> None:
    reg = PlanRegistry(
        {"trial": Plan("trial", monthly_request_cap=5, requests_per_minute=1)}
    )
    assert reg.names() == ["trial"]
    assert reg.decide("trial", used_this_period=5).allowed is False
    # unknown name with no free tier present still falls back to the free default
    assert reg.get("missing").name == "free"
