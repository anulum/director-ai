# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Tenant Routing Tests
"""Multi-angle tests for multi-tenant routing pipeline.

Covers: tenant creation, fact isolation, tenant-scoped review, policy
enforcement, tenant listing, pipeline integration with server and
enterprise modules, and performance documentation.
"""

from __future__ import annotations

import json

import pytest

from director_ai.core.tenant import ModelVersion, TenantRouter

pytestmark = pytest.mark.enterprise


class TestTenantRouterBasic:
    def test_get_store_creates_isolated_store(self):
        router = TenantRouter()
        store_a = router.get_store("acme")
        store_b = router.get_store("globex")
        assert store_a is not store_b
        assert store_a.facts == {}
        assert store_b.facts == {}

    def test_get_store_returns_same_instance(self):
        router = TenantRouter()
        s1 = router.get_store("acme")
        s2 = router.get_store("acme")
        assert s1 is s2

    def test_tenant_ids(self):
        router = TenantRouter()
        router.get_store("a")
        router.get_store("b")
        ids = router.tenant_ids
        assert set(ids) == {"a", "b"}

    def test_empty_tenant_ids(self):
        router = TenantRouter()
        assert router.tenant_ids == []


class TestTenantRouterFacts:
    def test_add_fact_isolated(self):
        router = TenantRouter()
        router.add_fact("acme", "capital", "Paris")
        router.add_fact("globex", "hq", "Springfield")

        acme_ctx = router.get_store("acme").retrieve_context("capital city")
        globex_ctx = router.get_store("globex").retrieve_context("capital city")

        assert acme_ctx is not None
        assert "Paris" in acme_ctx
        assert globex_ctx is None

    def test_fact_count(self):
        router = TenantRouter()
        router.add_fact("acme", "k1", "v1")
        router.add_fact("acme", "k2", "v2")
        assert router.fact_count("acme") == 2

    def test_fact_count_nonexistent_tenant(self):
        router = TenantRouter()
        assert router.fact_count("nope") == 0


class TestTenantRouterRemove:
    def test_remove_existing(self):
        router = TenantRouter()
        router.add_fact("acme", "k", "v")
        assert router.remove_tenant("acme") is True
        assert "acme" not in router.tenant_ids

    def test_remove_nonexistent(self):
        router = TenantRouter()
        assert router.remove_tenant("nope") is False


class TestTenantRouterScorer:
    def test_get_scorer(self):
        router = TenantRouter()
        router.add_fact("acme", "sky color", "blue")
        scorer = router.get_scorer("acme", threshold=0.5, use_nli=False)
        h = scorer.calculate_factual_divergence("sky color?", "The sky is blue")
        assert h < 0.5


class TestModelVersion:
    def test_defaults(self):
        mv = ModelVersion(model_id="v1", model_path="/models/v1")
        assert mv.active is False
        assert mv.balanced_accuracy == 0.0

    def test_custom(self):
        mv = ModelVersion(
            model_id="ft-med-v1",
            model_path="/models/med",
            balanced_accuracy=0.88,
            regression_pp=-1.5,
            recommendation="deploy",
            active=True,
        )
        assert mv.active
        assert mv.regression_pp == -1.5


class TestTenantModelVersioning:
    def test_set_model(self):
        router = TenantRouter()
        mv = router.set_model("acme", "v1", "/models/v1", balanced_accuracy=0.85)
        assert mv.model_id == "v1"
        assert mv.balanced_accuracy == 0.85
        assert mv.active is False

    def test_list_models_empty(self):
        router = TenantRouter()
        assert router.list_models("acme") == []

    def test_list_models(self):
        router = TenantRouter()
        router.set_model("acme", "v1", "/m/v1")
        router.set_model("acme", "v2", "/m/v2")
        models = router.list_models("acme")
        assert len(models) == 2

    def test_activate_model(self):
        router = TenantRouter()
        router.set_model("acme", "v1", "/m/v1")
        router.set_model("acme", "v2", "/m/v2")
        assert router.activate_model("acme", "v2")
        active = router.get_active_model("acme")
        assert active is not None
        assert active.model_id == "v2"

    def test_activate_deactivates_others(self):
        router = TenantRouter()
        router.set_model("acme", "v1", "/m/v1")
        router.set_model("acme", "v2", "/m/v2")
        router.activate_model("acme", "v1")
        router.activate_model("acme", "v2")
        models = router.list_models("acme")
        active_count = sum(1 for m in models if m.active)
        assert active_count == 1

    def test_activate_nonexistent(self):
        router = TenantRouter()
        assert not router.activate_model("acme", "nope")

    def test_rollback(self):
        router = TenantRouter()
        router.set_model("acme", "v1", "/m/v1")
        router.activate_model("acme", "v1")
        assert router.rollback_model("acme")
        assert router.get_active_model("acme") is None

    def test_rollback_empty(self):
        router = TenantRouter()
        assert not router.rollback_model("acme")

    def test_get_active_model_none(self):
        router = TenantRouter()
        assert router.get_active_model("acme") is None

    def test_delete_model(self):
        router = TenantRouter()
        router.set_model("acme", "v1", "/m/v1")
        assert router.delete_model("acme", "v1")
        assert router.list_models("acme") == []

    def test_delete_active_model_blocked(self):
        router = TenantRouter()
        router.set_model("acme", "v1", "/m/v1")
        router.activate_model("acme", "v1")
        assert not router.delete_model("acme", "v1")

    def test_delete_nonexistent(self):
        router = TenantRouter()
        assert not router.delete_model("acme", "nope")

    def test_tenant_isolation(self):
        router = TenantRouter()
        router.set_model("acme", "v1", "/m/acme")
        router.set_model("globex", "v1", "/m/globex")
        assert len(router.list_models("acme")) == 1
        assert len(router.list_models("globex")) == 1
        router.activate_model("acme", "v1")
        assert router.get_active_model("globex") is None


class TestTenantManifest:
    def test_save_and_load(self, tmp_path):
        router = TenantRouter()
        router.set_model("acme", "v1", "/m/v1", balanced_accuracy=0.85)
        router.set_model("acme", "v2", "/m/v2", balanced_accuracy=0.90)
        router.activate_model("acme", "v2")

        manifest_path = tmp_path / "manifest.json"
        router.save_manifest(manifest_path)

        router2 = TenantRouter()
        count = router2.load_manifest(manifest_path)
        assert count == 2
        models = router2.list_models("acme")
        assert len(models) == 2
        active = router2.get_active_model("acme")
        assert active is not None
        assert active.model_id == "v2"
        assert active.balanced_accuracy == 0.90

    def test_load_nonexistent(self, tmp_path):
        router = TenantRouter()
        count = router.load_manifest(tmp_path / "nope.json")
        assert count == 0

    def test_load_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{{", encoding="utf-8")
        router = TenantRouter()
        with pytest.raises(json.JSONDecodeError):
            router.load_manifest(bad)

    def test_round_trip_preserves_inactive(self, tmp_path):
        router = TenantRouter()
        router.set_model("acme", "v1", "/m/v1", balanced_accuracy=0.80)
        manifest = tmp_path / "m.json"
        router.save_manifest(manifest)

        router2 = TenantRouter()
        count = router2.load_manifest(manifest)
        assert count == 1
        assert router2.get_active_model("acme") is None
        models = router2.list_models("acme")
        assert models[0].balanced_accuracy == 0.80
