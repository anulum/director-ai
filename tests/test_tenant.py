from __future__ import annotations

from director_ai.core.tenant import TenantRouter


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
