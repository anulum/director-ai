# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Adversarial tests for TenantScopedBackend isolation enforcement.

Covers: constructor contracts, add-time tenant stamping and conflict
rejection, query-scope forcing (an empty caller tenant can never widen
the scope), defence-in-depth verification against a lying backend
(drop + metric in default mode, raise in strict mode), composition
with HybridBackend, and grounded() wiring.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from director_ai.core.metrics import metrics
from director_ai.core.vector_store import (
    HybridBackend,
    InMemoryBackend,
    TenantIsolationError,
    TenantScopedBackend,
)


class _LyingBackend(InMemoryBackend):
    """Returns rows regardless of the tenant filter — a broken filter."""

    def query(self, text, n_results=3, tenant_id=""):
        with self._lock:
            snapshot = list(self._docs)
        return [{**doc, "distance": 0.5} for doc in snapshot[:n_results]]


class _RecordingBackend(InMemoryBackend):
    """Records the tenant_id each query was called with."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_tenants: list[str] = []

    def query(self, text, n_results=3, tenant_id=""):
        self.seen_tenants.append(tenant_id)
        return super().query(text, n_results, tenant_id)


class TestConstructorContracts:
    def test_rejects_missing_base(self):
        with pytest.raises(ValueError, match="base backend is required"):
            TenantScopedBackend(None, "t1")

    @pytest.mark.parametrize("tenant", ["", None, 42])
    def test_rejects_empty_or_non_string_tenant(self, tenant):
        with pytest.raises(TenantIsolationError, match="non-empty tenant_id"):
            TenantScopedBackend(InMemoryBackend(), tenant)

    @pytest.mark.parametrize(
        "tenant",
        ["-leading-dash", "has space", "a" * 129, "tab\tchar", "semi;colon"],
    )
    def test_rejects_malformed_tenant_shape(self, tenant):
        with pytest.raises(TenantIsolationError, match="must match"):
            TenantScopedBackend(InMemoryBackend(), tenant)

    def test_rejects_non_bool_strict(self):
        with pytest.raises(ValueError, match="strict must be a boolean"):
            TenantScopedBackend(InMemoryBackend(), "t1", strict="yes")

    def test_exposes_bound_tenant(self):
        guard = TenantScopedBackend(InMemoryBackend(), "acme")

        assert guard.tenant_id == "acme"


class TestAddStamping:
    def setup_method(self):
        self.base = InMemoryBackend()
        self.guard = TenantScopedBackend(self.base, "t1")

    def test_stamps_tenant_when_metadata_missing(self):
        self.guard.add("d1", "text one")

        assert self.base._docs[0]["metadata"]["tenant_id"] == "t1"

    def test_stamps_tenant_into_existing_metadata(self):
        self.guard.add("d1", "text one", metadata={"source": "manual"})

        stored = self.base._docs[0]["metadata"]
        assert stored["tenant_id"] == "t1"
        assert stored["source"] == "manual"

    def test_accepts_matching_tenant_metadata(self):
        self.guard.add("d1", "text", metadata={"tenant_id": "t1"})

        assert self.base._docs[0]["metadata"]["tenant_id"] == "t1"

    def test_rejects_conflicting_tenant_metadata(self):
        with pytest.raises(TenantIsolationError, match="conflicts with the bound"):
            self.guard.add("d1", "text", metadata={"tenant_id": "t2"})

    def test_does_not_mutate_caller_metadata(self):
        caller_meta: dict = {"source": "manual"}

        self.guard.add("d1", "text", metadata=caller_meta)

        assert caller_meta == {"source": "manual"}


class TestQueryScopeForcing:
    def test_empty_caller_tenant_cannot_widen_scope(self):
        base = _RecordingBackend()
        guard = TenantScopedBackend(base, "t1")
        guard.add("d1", "shared corpus document")

        guard.query("document")

        assert base.seen_tenants == ["t1"]

    def test_matching_caller_tenant_allowed(self):
        base = _RecordingBackend()
        guard = TenantScopedBackend(base, "t1")
        guard.add("d1", "shared corpus document")

        rows = guard.query("document", tenant_id="t1")

        assert [r["id"] for r in rows] == ["d1"]

    def test_foreign_caller_tenant_raises(self):
        guard = TenantScopedBackend(InMemoryBackend(), "t1")

        with pytest.raises(TenantIsolationError, match="conflicts with the bound"):
            guard.query("anything", tenant_id="t2")


class TestDefenceInDepth:
    def _leaky_guard(self, strict: bool) -> TenantScopedBackend:
        base = _LyingBackend()
        base.add("mine", "tenant document", metadata={"tenant_id": "t1"})
        base.add("theirs", "foreign document", metadata={"tenant_id": "t2"})
        base.add("unlabelled", "no tenant metadata at all")
        return TenantScopedBackend(base, "t1", strict=strict)

    def test_drops_foreign_and_unlabelled_rows(self):
        guard = self._leaky_guard(strict=False)

        rows = guard.query("document", n_results=3)

        assert [r["id"] for r in rows] == ["mine"]

    def test_counts_violations_in_metrics(self):
        def _violations() -> float:
            counters = metrics.get_metrics()["counters"]
            entry = counters.get("tenant_isolation_violations")
            return entry["total"] if entry else 0.0

        guard = self._leaky_guard(strict=False)
        before = _violations()

        guard.query("document", n_results=3)

        assert _violations() - before == 2.0

    def test_logs_violation_warning(self, caplog):
        guard = self._leaky_guard(strict=False)

        with caplog.at_level("WARNING", logger="DirectorAI.VectorStore"):
            guard.query("document", n_results=3)

        assert any("outside tenant" in rec.getMessage() for rec in caplog.records)

    def test_strict_mode_raises_on_leak(self):
        guard = self._leaky_guard(strict=True)

        with pytest.raises(TenantIsolationError, match="outside tenant"):
            guard.query("document", n_results=3)

    def test_clean_backend_passes_untouched(self):
        base = InMemoryBackend()
        guard = TenantScopedBackend(base, "t1", strict=True)
        guard.add("d1", "tenant document")

        rows = guard.query("tenant document")

        assert [r["id"] for r in rows] == ["d1"]


class TestComposition:
    def test_guard_over_hybrid_isolates_both_runs(self):
        """BM25 and dense runs both stay inside the bound tenant."""
        hybrid = HybridBackend(InMemoryBackend())
        hybrid.add("other", "machine learning basics", {"tenant_id": "t2"})
        guard = TenantScopedBackend(hybrid, "t1")
        guard.add("mine", "machine learning advanced topics")

        rows = guard.query("machine learning", n_results=5)

        assert [r["id"] for r in rows] == ["mine"]

    def test_count_is_index_wide(self):
        hybrid = HybridBackend(InMemoryBackend())
        hybrid.add("other", "doc", {"tenant_id": "t2"})
        guard = TenantScopedBackend(hybrid, "t1")
        guard.add("mine", "doc")

        assert guard.count() == 2


class TestGroundedWiring:
    def _patched_grounded(self, **kwargs):
        from director_ai.core.vector_store import VectorGroundTruthStore

        with (
            patch(
                "director_ai.core.retrieval.vector_store.store."
                "_build_ann_dense_backend",
                return_value=None,
            ),
            patch(
                "director_ai.core.retrieval.vector_store.store."
                "SentenceTransformerBackend",
                side_effect=RuntimeError("missing sentence-transformers"),
            ),
        ):
            return VectorGroundTruthStore.grounded(use_reranker=False, **kwargs)

    def test_enforcement_wraps_outermost_backend(self):
        store = self._patched_grounded(
            tenant_id="acme",
            enforce_tenant_isolation=True,
        )

        assert isinstance(store.backend, TenantScopedBackend)
        assert store.backend.tenant_id == "acme"
        assert isinstance(store.backend._base, HybridBackend)

    def test_enforcement_requires_tenant(self):
        with pytest.raises(TenantIsolationError, match="non-empty tenant_id"):
            self._patched_grounded(enforce_tenant_isolation=True)

    def test_default_remains_unwrapped(self):
        store = self._patched_grounded(tenant_id="acme")

        assert not isinstance(store.backend, TenantScopedBackend)

    def test_enforced_store_roundtrip(self):
        store = self._patched_grounded(
            tenant_id="acme",
            enforce_tenant_isolation=True,
        )
        store.ingest(["The warranty period is 24 months."])

        context = store.retrieve_context("warranty period")

        assert context is not None
        assert "24 months" in context
