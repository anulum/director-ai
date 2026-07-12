# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real-surface tests for TenantScopedBackend — no mocks, no patching.

Exercises the guard over real InMemory and Hybrid backends and a real
VectorGroundTruthStore end to end: cross-tenant documents never leave
the bound scope, adds stamp real metadata, and the retrieval facade
round-trips through the enforced stack.
"""

from __future__ import annotations

import pytest

from director_ai.core.vector_store import (
    HybridBackend,
    InMemoryBackend,
    TenantIsolationError,
    TenantScopedBackend,
    VectorGroundTruthStore,
)


class TestRealBackendIsolation:
    def _shared_index(self) -> HybridBackend:
        """One shared hybrid index holding two tenants' documents."""
        hybrid = HybridBackend(InMemoryBackend())
        hybrid.add(
            "acme-1",
            "ACME warranty period is 24 months",
            {"tenant_id": "acme"},
        )
        hybrid.add(
            "globex-1",
            "Globex warranty period is 6 months",
            {"tenant_id": "globex"},
        )
        return hybrid

    def test_bound_scope_survives_shared_index(self):
        shared = self._shared_index()
        acme = TenantScopedBackend(shared, "acme")
        globex = TenantScopedBackend(shared, "globex")

        acme_rows = acme.query("warranty period", n_results=5)
        globex_rows = globex.query("warranty period", n_results=5)

        assert [r["id"] for r in acme_rows] == ["acme-1"]
        assert [r["id"] for r in globex_rows] == ["globex-1"]

    def test_add_through_guard_lands_in_bound_scope(self):
        shared = self._shared_index()
        acme = TenantScopedBackend(shared, "acme")

        acme.add("acme-2", "ACME support hours are 9 to 5")

        rows = acme.query("support hours", n_results=5)
        assert [r["id"] for r in rows] == ["acme-2"]
        globex = TenantScopedBackend(shared, "globex")
        assert globex.query("support hours", n_results=5) == []

    def test_cross_tenant_query_attempt_raises(self):
        acme = TenantScopedBackend(self._shared_index(), "acme")

        with pytest.raises(TenantIsolationError):
            acme.query("warranty", tenant_id="globex")


class TestRealStoreRoundTrip:
    def test_enforced_store_retrieves_only_bound_tenant(self):
        shared = HybridBackend(InMemoryBackend())
        shared.add(
            "other",
            "Foreign tenant refund policy is 14 days",
            {"tenant_id": "globex"},
        )
        store = VectorGroundTruthStore(
            backend=TenantScopedBackend(shared, "acme"),
            tenant_id="acme",
        )
        store.ingest(["The refund policy is 30 days."])

        context = store.retrieve_context("refund policy")

        assert context is not None
        assert "30 days" in context
        assert "14 days" not in context
