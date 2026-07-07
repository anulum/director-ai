# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — License Tier Enforcement Tests
"""Multi-angle tests for runtime license-tier enforcement (SEC-3)."""

from __future__ import annotations

import pytest

from director_ai.core.license import (
    LicenseError,
    LicenseInfo,
    enforce_capability_tier,
    require_tier,
    tier_enforcement_enabled,
    tier_rank,
)
from director_ai.guard import ProductionGuard

_ENV = "DIRECTOR_ENFORCE_LICENSE_TIER"


def _license(tier: str, *, valid: bool = True) -> LicenseInfo:
    return LicenseInfo(tier=tier, valid=valid)


class TestTierRank:
    @pytest.mark.parametrize(
        ("tier", "expected"),
        [("community", 0), ("indie", 1), ("trial", 2), ("pro", 2), ("enterprise", 3)],
    )
    def test_known_tiers_rank_in_order(self, tier, expected):
        assert tier_rank(tier) == expected

    def test_unknown_tier_ranks_as_community(self):
        assert tier_rank("platinum") == 0

    def test_rank_is_case_insensitive(self):
        assert tier_rank("PRO") == tier_rank("pro")


class TestRequireTier:
    def test_higher_tier_satisfies_lower_minimum(self):
        require_tier(_license("enterprise"), "pro")

    def test_equal_tier_passes(self):
        require_tier(_license("pro"), "pro")

    def test_below_minimum_raises_naming_required_and_active_tiers(self):
        with pytest.raises(LicenseError, match="pro") as exc:
            require_tier(_license("indie"), "pro", capability="sector_policy")
        message = str(exc.value)
        assert "indie" in message
        assert "sector_policy" in message

    def test_community_minimum_always_passes(self):
        require_tier(_license("community", valid=False), "community")

    def test_invalid_license_is_treated_as_community(self):
        with pytest.raises(LicenseError):
            require_tier(_license("enterprise", valid=False), "pro")

    def test_trial_satisfies_pro_but_not_enterprise(self):
        require_tier(_license("trial"), "pro")
        with pytest.raises(LicenseError):
            require_tier(_license("trial"), "enterprise")


class TestEnforcementToggle:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv(_ENV, value)
        assert tier_enforcement_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_falsy_values_disable(self, monkeypatch, value):
        monkeypatch.setenv(_ENV, value)
        assert tier_enforcement_enabled() is False

    def test_absent_env_disables(self, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)
        assert tier_enforcement_enabled() is False


class TestEnforceCapabilityTier:
    def test_noop_when_disabled_does_not_load_license(self, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)

        def _boom() -> LicenseInfo:
            raise AssertionError("load_license must not run when enforcement is off")

        monkeypatch.setattr("director_ai.core.license.load_license", _boom)
        enforce_capability_tier("sector_policy")

    def test_raises_when_enabled_and_license_insufficient(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1")
        monkeypatch.setattr(
            "director_ai.core.license.load_license", lambda: _license("community")
        )
        with pytest.raises(LicenseError, match="sector_policy"):
            enforce_capability_tier("sector_policy")

    def test_passes_when_enabled_and_license_sufficient(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1")
        monkeypatch.setattr(
            "director_ai.core.license.load_license", lambda: _license("enterprise")
        )
        enforce_capability_tier("sector_policy")

    def test_custom_minimum_is_honoured(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1")
        monkeypatch.setattr(
            "director_ai.core.license.load_license", lambda: _license("pro")
        )
        with pytest.raises(LicenseError, match="enterprise"):
            enforce_capability_tier("federated_dp", minimum="enterprise")


def _gate_raises(capability: str, *, minimum: str = "pro") -> None:
    """enforce_capability_tier stand-in that always gates the given capability."""
    raise LicenseError(f"gated: {capability}")


class TestGuardTierWiring:
    def test_repair_stream_gates_on_tier(self, monkeypatch):
        guard = ProductionGuard()
        monkeypatch.setattr("director_ai.guard.enforce_capability_tier", _gate_raises)
        with pytest.raises(LicenseError, match="repair_stream"):
            guard.repair_stream("prompt", "response")

    def test_sector_policy_gates_on_tier(self, monkeypatch):
        guard = ProductionGuard()
        monkeypatch.setattr("director_ai.guard.enforce_capability_tier", _gate_raises)
        with pytest.raises(LicenseError, match="sector_policy"):
            guard._evaluate_sector_policy(
                sector_policy="banking",
                prompt="p",
                response="r",
                evidence_refs=[],
                numeric_evidence_refs=[],
                policy_refs=[],
                jurisdiction="",
                product_line="",
                human_review_acknowledged=False,
            )
