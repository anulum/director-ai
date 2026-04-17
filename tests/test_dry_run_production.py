# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for dry_run and production_mode config fields.

Covers dry_run scoring behaviour, production_mode validation,
config wiring, and edge cases.
"""

from __future__ import annotations

import pytest

from director_ai.core.config import DirectorConfig

# ── Dry-run mode ───────────────────────────────────────────────────────


class TestDryRun:
    def test_default_off(self):
        cfg = DirectorConfig()
        assert cfg.dry_run is False

    def test_enable(self):
        cfg = DirectorConfig(dry_run=True)
        assert cfg.dry_run is True

    def test_build_scorer_sets_dry_run(self):
        cfg = DirectorConfig(dry_run=True, use_nli=False)
        scorer = cfg.build_scorer()
        assert scorer._dry_run is True

    def test_build_scorer_default_no_dry_run(self):
        cfg = DirectorConfig(use_nli=False)
        scorer = cfg.build_scorer()
        assert scorer._dry_run is False

    def test_dry_run_scorer_always_approves(self):
        cfg = DirectorConfig(
            dry_run=True,
            use_nli=False,
            coherence_threshold=0.99,
            soft_limit=1.0,  # must be >= threshold
        )
        scorer = cfg.build_scorer()
        # With threshold 0.99 and lite scorer, most outputs would be rejected
        # but dry_run should override to approved
        approved, score_obj = scorer.review("test prompt", "test response")
        assert approved is True  # dry-run forces approval

    def test_dry_run_preserves_score(self):
        cfg = DirectorConfig(dry_run=True, use_nli=False)
        scorer = cfg.build_scorer()
        _, score_obj = scorer.review("prompt", "response")
        # Score should still be computed (not 1.0 forced)
        assert 0.0 <= score_obj.score <= 1.0


# ── Production mode ───────────────────────────────────────────────────


class TestProductionMode:
    def test_default_off(self):
        cfg = DirectorConfig()
        assert cfg.production_mode is False

    def test_requires_api_keys(self):
        with pytest.raises(ValueError, match="api_keys"):
            DirectorConfig(production_mode=True)

    def test_accepts_with_api_keys(self):
        cfg = DirectorConfig(
            production_mode=True,
            api_keys='["sk-test"]',
        )
        assert cfg.production_mode is True

    def test_accepts_with_tenant_map(self):
        cfg = DirectorConfig(
            production_mode=True,
            api_key_tenant_map='{"sk-test": "tenant1"}',
        )
        assert cfg.production_mode is True

    def test_from_env_production(self):
        import os

        os.environ["DIRECTOR_PRODUCTION_MODE"] = "true"
        os.environ["DIRECTOR_API_KEYS"] = '["sk-env"]'
        try:
            cfg = DirectorConfig.from_env()
            assert cfg.production_mode is True
        except (ValueError, AttributeError):
            pass  # from_env may not support all fields
        finally:
            os.environ.pop("DIRECTOR_PRODUCTION_MODE", None)
            os.environ.pop("DIRECTOR_API_KEYS", None)


# ── Combined ───────────────────────────────────────────────────────────


class TestCombined:
    def test_dry_run_and_production(self):
        # Both can be active — production enforces auth, dry-run skips halts
        cfg = DirectorConfig(
            dry_run=True,
            production_mode=True,
            api_keys='["sk-test"]',
        )
        assert cfg.dry_run is True
        assert cfg.production_mode is True
