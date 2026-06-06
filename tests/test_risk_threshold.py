# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for risk-adaptive thresholding.

Covers factor and policy validation, the per-factor contribution of each risk
input, clamping, determinism, and the ProductionGuard integration.
"""

from __future__ import annotations

import pytest

from director_ai.core.risk_threshold import (
    RiskAdaptiveThreshold,
    RiskFactors,
    RiskThresholdPolicy,
)


def _adapter() -> RiskAdaptiveThreshold:
    return RiskAdaptiveThreshold(RiskThresholdPolicy())


# ── RiskFactors ─────────────────────────────────────────────────────────


class TestRiskFactors:
    @pytest.mark.parametrize(
        "field",
        [
            "tenant_risk",
            "retrieval_confidence",
            "action_reversibility",
            "freshness",
            "historical_fpr",
        ],
    )
    def test_unit_validation(self, field):
        with pytest.raises(ValueError, match=field):
            RiskFactors(**{field: 1.5})

    def test_defaults_are_safe_end(self):
        f = RiskFactors()
        assert f.retrieval_confidence == 1.0
        assert f.action_reversibility == 1.0
        assert f.freshness == 1.0
        assert f.tenant_risk == 0.0


# ── RiskThresholdPolicy ─────────────────────────────────────────────────


class TestPolicy:
    def test_base_threshold_range(self):
        with pytest.raises(ValueError, match="base_threshold"):
            RiskThresholdPolicy(base_threshold=1.5)

    def test_min_max_order(self):
        with pytest.raises(ValueError, match="min_threshold must not exceed"):
            RiskThresholdPolicy(min_threshold=0.9, max_threshold=0.5)


# ── adapter: per-factor contributions ───────────────────────────────────


class TestContributions:
    def test_baseline_is_base_threshold(self):
        decision = _adapter().evaluate(RiskFactors())
        assert decision.threshold == 0.6
        assert decision.contributions == {}

    def test_admin_role_relaxes(self):
        decision = _adapter().evaluate(RiskFactors(user_role="admin"))
        assert decision.contributions["user_role"] == -0.05
        assert decision.threshold == 0.55

    def test_anonymous_role_tightens(self):
        decision = _adapter().evaluate(RiskFactors(user_role="anonymous"))
        assert decision.contributions["user_role"] == 0.10

    def test_unknown_role_no_contribution(self):
        decision = _adapter().evaluate(RiskFactors(user_role="ceo"))
        assert "user_role" not in decision.contributions

    def test_medical_domain_tightens(self):
        decision = _adapter().evaluate(RiskFactors(domain="medical"))
        assert decision.contributions["domain"] == 0.10

    def test_unknown_domain_no_contribution(self):
        decision = _adapter().evaluate(RiskFactors(domain="astrology"))
        assert "domain" not in decision.contributions

    def test_tenant_risk_scales(self):
        decision = _adapter().evaluate(RiskFactors(tenant_risk=1.0))
        assert decision.contributions["tenant_risk"] == 0.15

    def test_low_retrieval_confidence_tightens(self):
        decision = _adapter().evaluate(RiskFactors(retrieval_confidence=0.0))
        assert decision.contributions["retrieval_confidence"] == 0.15

    def test_full_retrieval_confidence_no_contribution(self):
        decision = _adapter().evaluate(RiskFactors(retrieval_confidence=1.0))
        assert "retrieval_confidence" not in decision.contributions

    def test_low_reversibility_tightens(self):
        decision = _adapter().evaluate(RiskFactors(action_reversibility=0.0))
        assert decision.contributions["action_reversibility"] == 0.15

    def test_external_exposure(self):
        decision = _adapter().evaluate(RiskFactors(external_exposure=True))
        assert decision.contributions["external_exposure"] == 0.05

    def test_pii_present(self):
        decision = _adapter().evaluate(RiskFactors(pii_present=True))
        assert decision.contributions["pii_present"] == 0.08

    def test_stale_freshness_tightens(self):
        decision = _adapter().evaluate(RiskFactors(freshness=0.0))
        assert decision.contributions["freshness"] == 0.10

    def test_historical_fpr_relaxes(self):
        decision = _adapter().evaluate(RiskFactors(historical_fpr=1.0))
        assert decision.contributions["historical_fpr"] == -0.15


# ── adapter: aggregation, clamping, determinism ─────────────────────────


class TestAggregation:
    def test_total_delta(self):
        decision = _adapter().evaluate(RiskFactors(domain="medical", pii_present=True))
        assert decision.total_delta == pytest.approx(0.18)

    def test_clamps_at_max(self):
        decision = _adapter().evaluate(
            RiskFactors(
                domain="medical",
                pii_present=True,
                external_exposure=True,
                tenant_risk=1.0,
                retrieval_confidence=0.0,
                action_reversibility=0.0,
                freshness=0.0,
            )
        )
        assert decision.threshold == 0.95

    def test_clamps_at_min(self):
        policy = RiskThresholdPolicy(
            base_threshold=0.4, min_threshold=0.35, historical_fpr_weight=0.5
        )
        decision = RiskAdaptiveThreshold(policy).evaluate(
            RiskFactors(user_role="admin", historical_fpr=1.0)
        )
        assert decision.threshold == 0.35

    def test_deterministic(self):
        adapter = _adapter()
        factors = RiskFactors(domain="finance", pii_present=True, tenant_risk=0.5)
        assert (
            adapter.evaluate(factors).threshold == adapter.evaluate(factors).threshold
        )

    def test_to_dict(self):
        payload = _adapter().evaluate(RiskFactors(domain="medical")).to_dict()
        assert payload["base_threshold"] == 0.6
        assert payload["contributions"]["domain"] == 0.10
        assert "total_delta" in payload


# ── guard integration ───────────────────────────────────────────────────


class TestGuardIntegration:
    def test_guard_uses_config_base_threshold(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        decision = guard.risk_threshold(RiskFactors())
        assert decision.base_threshold == guard.config.coherence_threshold

    def test_guard_risk_tightens_for_high_risk(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        base = guard.risk_threshold(RiskFactors()).threshold
        risky = guard.risk_threshold(
            RiskFactors(domain="medical", pii_present=True)
        ).threshold
        assert risky > base
