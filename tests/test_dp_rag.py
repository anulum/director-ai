# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Differentially Private RAG Tests
"""Multi-angle tests for DP retrieval ranking and the per-tenant budget."""

from __future__ import annotations

import pytest

from director_ai.core.dp_rag import (
    DifferentiallyPrivateRetrieval,
    DPBudgetExceededError,
    PrivateRanking,
    ScoredItem,
)

_ITEMS = [ScoredItem("a", 0.9), ScoredItem("b", 0.5), ScoredItem("c", 0.1)]


class TestValidation:
    def test_max_epsilon_must_be_positive(self):
        with pytest.raises(ValueError, match="max_epsilon"):
            DifferentiallyPrivateRetrieval(0.0)

    def test_sensitivity_non_negative(self):
        with pytest.raises(ValueError, match="sensitivity"):
            DifferentiallyPrivateRetrieval(1.0, sensitivity=-1.0)

    def test_empty_items_rejected(self):
        dp = DifferentiallyPrivateRetrieval(1.0, seed=1)
        with pytest.raises(ValueError, match="non-empty"):
            dp.rank([], tenant_id="t", epsilon=0.5)

    def test_non_positive_epsilon_rejected(self):
        dp = DifferentiallyPrivateRetrieval(1.0, seed=1)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            dp.rank(_ITEMS, tenant_id="t", epsilon=0.0)

    def test_bad_tenant_rejected(self):
        dp = DifferentiallyPrivateRetrieval(1.0, seed=1)
        with pytest.raises(ValueError, match="tenant_id"):
            dp.rank(_ITEMS, tenant_id="bad tenant!", epsilon=0.5)


class TestBudget:
    def test_spend_and_remaining(self):
        dp = DifferentiallyPrivateRetrieval(max_epsilon=2.0, seed=1)
        assert dp.remaining("t1") == 2.0
        r = dp.rank(_ITEMS, tenant_id="t1", epsilon=0.5)
        assert r.epsilon_spent == 0.5
        assert dp.spent("t1") == pytest.approx(0.5)
        assert dp.remaining("t1") == pytest.approx(1.5)
        assert r.epsilon_remaining == pytest.approx(1.5)

    def test_exhaustion_refuses_without_spending(self):
        dp = DifferentiallyPrivateRetrieval(max_epsilon=1.0, seed=1)
        dp.rank(_ITEMS, tenant_id="t1", epsilon=0.8)
        before = dp.spent("t1")
        with pytest.raises(DPBudgetExceededError):
            dp.rank(_ITEMS, tenant_id="t1", epsilon=0.5)
        # The refused query did not consume any budget.
        assert dp.spent("t1") == pytest.approx(before)

    def test_tenant_isolation(self):
        dp = DifferentiallyPrivateRetrieval(max_epsilon=2.0, seed=1)
        dp.rank(_ITEMS, tenant_id="t1", epsilon=1.0)
        assert dp.remaining("t1") == pytest.approx(1.0)
        # A different tenant has its own full budget.
        assert dp.remaining("t2") == 2.0

    def test_adaptive_per_query_epsilon(self):
        dp = DifferentiallyPrivateRetrieval(max_epsilon=3.0, seed=1)
        dp.rank(_ITEMS, tenant_id="t1", epsilon=0.5)
        dp.rank(_ITEMS, tenant_id="t1", epsilon=1.5)
        assert dp.spent("t1") == pytest.approx(2.0)


class TestRanking:
    def test_returns_permutation_of_inputs(self):
        dp = DifferentiallyPrivateRetrieval(max_epsilon=5.0, seed=3)
        r = dp.rank(_ITEMS, tenant_id="t1", epsilon=0.5)
        assert {it.item_id for it in r.items} == {"a", "b", "c"}
        assert len(r.items) == 3

    def test_high_epsilon_preserves_well_separated_order(self):
        # Low noise (large epsilon) keeps the order of well-separated scores.
        dp = DifferentiallyPrivateRetrieval(max_epsilon=200.0, seed=5)
        r = dp.rank(_ITEMS, tenant_id="t1", epsilon=100.0)
        assert [it.item_id for it in r.items] == ["a", "b", "c"]

    def test_deterministic_with_seed(self):
        a = DifferentiallyPrivateRetrieval(max_epsilon=5.0, seed=9)
        b = DifferentiallyPrivateRetrieval(max_epsilon=5.0, seed=9)
        ra = a.rank(_ITEMS, tenant_id="x", epsilon=0.7)
        rb = b.rank(_ITEMS, tenant_id="x", epsilon=0.7)
        assert [i.item_id for i in ra.items] == [i.item_id for i in rb.items]
        assert [i.score for i in ra.items] == [i.score for i in rb.items]

    def test_successive_queries_draw_independent_noise(self):
        # Same seed base, but successive calls advance the noise stream.
        dp = DifferentiallyPrivateRetrieval(max_epsilon=5.0, seed=1)
        first = dp.rank(_ITEMS, tenant_id="t1", epsilon=0.5).items
        second = dp.rank(_ITEMS, tenant_id="t1", epsilon=0.5).items
        # Noised scores differ between the two queries (independent draws).
        assert [i.score for i in first] != [i.score for i in second]

    def test_to_dict_is_tenant_safe(self):
        dp = DifferentiallyPrivateRetrieval(max_epsilon=5.0, seed=1)
        d = dp.rank(_ITEMS, tenant_id="t1", epsilon=0.5).to_dict()
        assert set(d) == {"tenant_id", "epsilon_spent", "epsilon_remaining", "items"}
        assert d["items"][0].keys() == {"item_id", "score"}


class TestGuardWiring:
    def test_guard_dp_retrieval_persists(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        dp = guard.dp_retrieval
        assert guard.dp_retrieval is dp  # persists across calls
        r = dp.rank(_ITEMS, tenant_id="t1", epsilon=1.0)
        assert isinstance(r, PrivateRanking)
        assert dp.spent("t1") == pytest.approx(1.0)
