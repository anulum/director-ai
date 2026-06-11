# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DP-RAG decoding + unified-budget pipeline tests
"""Multi-angle tests for DP token decoding and the unified DP-RAG budget.

Covers the exponential-mechanism decoder (validation, sensitivity-zero passthrough,
the high-/low-ε noise regimes and the distributional bias toward the top logit,
independent noise across calls, tenant-safe payload) and the pipeline that meters
retrieval, decoding, and score release against one per-tenant ε accountant
(shared-budget accumulation, refuse-before-spend with no state mutation, stage
logging, per-tenant isolation, and input validation).
"""

from __future__ import annotations

import pytest

from director_ai.core.dp_rag import (
    DPBudgetExceededError,
    DPRagPipeline,
    DPTokenChoice,
    DPTokenDecoder,
    PipelineRanking,
    ScoredItem,
    StageCharge,
)


class TestDPTokenDecoder:
    def test_select_returns_valid_index(self):
        decoder = DPTokenDecoder(seed=1)
        choice = decoder.select([0.1, 0.2, 0.3], epsilon=1.0)
        assert isinstance(choice, DPTokenChoice)
        assert 0 <= choice.index < 3
        assert choice.epsilon_spent == 1.0

    def test_deterministic_with_seed(self):
        a = DPTokenDecoder(seed=7).select([1.0, 2.0, 0.5], epsilon=2.0)
        b = DPTokenDecoder(seed=7).select([1.0, 2.0, 0.5], epsilon=2.0)
        assert a == b

    def test_high_epsilon_selects_true_argmax(self):
        # Very low noise: the dominant logit is selected.
        decoder = DPTokenDecoder(seed=3)
        choice = decoder.select([0.0, 0.0, 5.0, 0.0], epsilon=50.0)
        assert choice.index == 2

    def test_sensitivity_zero_is_plain_argmax(self):
        decoder = DPTokenDecoder(sensitivity=0.0, seed=99)
        # No private signal → no noise → deterministic argmax regardless of ε.
        choice = decoder.select([0.2, 0.9, 0.4], epsilon=0.01)
        assert choice.index == 1
        assert choice.noisy_logit == pytest.approx(0.9)

    def test_distribution_favours_top_logit(self):
        # Exponential mechanism: over many draws the top logit wins most often.
        decoder = DPTokenDecoder(seed=2024)
        logits = [3.0, 0.0, 0.0, 0.0]
        counts = [0, 0, 0, 0]
        for _ in range(400):
            counts[decoder.select(logits, epsilon=4.0).index] += 1
        assert counts[0] == max(counts)
        assert counts[0] > sum(counts[1:])

    def test_independent_noise_across_calls(self):
        decoder = DPTokenDecoder(seed=5)
        first = decoder.select([1.0, 1.0, 1.0], epsilon=0.5)
        second = decoder.select([1.0, 1.0, 1.0], epsilon=0.5)
        # Distinct per-call seeds → the two draws are not forced to coincide.
        assert first.noisy_logit != second.noisy_logit

    def test_empty_logits_rejected(self):
        with pytest.raises(ValueError, match="logits must be non-empty"):
            DPTokenDecoder().select([], epsilon=1.0)

    def test_non_positive_epsilon_rejected(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="epsilon must be positive"):
                DPTokenDecoder(seed=1).select([0.1, 0.2], epsilon=bad)

    def test_non_finite_epsilon_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be positive and finite"):
            DPTokenDecoder(seed=1).select([0.1, 0.2], epsilon=float("inf"))

    def test_non_finite_logits_rejected(self):
        with pytest.raises(ValueError, match="logits must be finite"):
            DPTokenDecoder(seed=1).select([0.1, float("nan")], epsilon=1.0)

    def test_negative_sensitivity_rejected(self):
        with pytest.raises(ValueError, match="sensitivity must be non-negative"):
            DPTokenDecoder(sensitivity=-1.0)

    def test_choice_to_dict_is_tenant_safe(self):
        payload = DPTokenDecoder(seed=1).select([0.1, 0.2], epsilon=1.0).to_dict()
        assert set(payload) == {"index", "noisy_logit", "epsilon_spent"}

    def test_sensitivity_property(self):
        assert DPTokenDecoder(sensitivity=2.5).sensitivity == 2.5


def _items() -> list[ScoredItem]:
    return [ScoredItem("a", 0.9), ScoredItem("b", 0.5), ScoredItem("c", 0.1)]


class TestDPRagPipelineSharedBudget:
    def test_stages_share_one_budget(self):
        pipe = DPRagPipeline(max_epsilon=10.0, seed=11)
        pipe.rank(_items(), tenant_id="t", epsilon=2.0)
        pipe.decode([1.0, 2.0, 3.0], tenant_id="t", epsilon=3.0)
        pipe.release_score(0.7, tenant_id="t", epsilon=1.5)
        assert pipe.spent("t") == pytest.approx(6.5)
        assert pipe.remaining("t") == pytest.approx(3.5)

    def test_stage_log_records_each_stage_in_order(self):
        pipe = DPRagPipeline(max_epsilon=10.0, seed=1)
        pipe.rank(_items(), tenant_id="t", epsilon=1.0)
        pipe.decode([0.1, 0.2], tenant_id="t", epsilon=2.0)
        pipe.release_score(0.4, tenant_id="t", epsilon=0.5)
        log = pipe.stage_log("t")
        assert [c.stage for c in log] == ["retrieve", "decode", "release"]
        assert [c.epsilon for c in log] == [1.0, 2.0, 0.5]
        assert all(isinstance(c, StageCharge) for c in log)

    def test_rank_returns_sorted_ranking(self):
        pipe = DPRagPipeline(max_epsilon=10.0, seed=1)
        ranking = pipe.rank(_items(), tenant_id="t", epsilon=0.5)
        assert isinstance(ranking, PipelineRanking)
        scores = [it.score for it in ranking.items]
        assert scores == sorted(scores, reverse=True)
        assert ranking.epsilon_spent == 0.5
        assert ranking.epsilon_remaining == pytest.approx(9.5)

    def test_refuse_before_spend_leaves_budget_untouched(self):
        pipe = DPRagPipeline(max_epsilon=5.0, seed=1)
        pipe.rank(_items(), tenant_id="t", epsilon=4.0)
        before_spent = pipe.spent("t")
        before_log = pipe.stage_log("t")
        with pytest.raises(DPBudgetExceededError, match="decode epsilon"):
            pipe.decode([0.1, 0.2], tenant_id="t", epsilon=2.0)
        # No noise drawn, no charge, no log entry.
        assert pipe.spent("t") == before_spent
        assert pipe.stage_log("t") == before_log

    def test_budget_exhaustion_across_mixed_stages(self):
        pipe = DPRagPipeline(max_epsilon=6.0, seed=1)
        pipe.rank(_items(), tenant_id="t", epsilon=3.0)
        pipe.decode([0.1, 0.2], tenant_id="t", epsilon=3.0)
        with pytest.raises(DPBudgetExceededError, match="release epsilon"):
            pipe.release_score(0.5, tenant_id="t", epsilon=0.5)

    def test_per_tenant_isolation(self):
        pipe = DPRagPipeline(max_epsilon=5.0, seed=1)
        pipe.rank(_items(), tenant_id="a", epsilon=4.0)
        # Tenant b has its own untouched budget.
        assert pipe.spent("b") == 0.0
        assert pipe.remaining("b") == 5.0
        pipe.decode([0.1, 0.2], tenant_id="b", epsilon=4.0)
        assert pipe.spent("a") == pytest.approx(4.0)
        assert pipe.spent("b") == pytest.approx(4.0)

    def test_deterministic_with_seed(self):
        a = DPRagPipeline(max_epsilon=10.0, seed=42).rank(
            _items(), tenant_id="t", epsilon=1.0
        )
        b = DPRagPipeline(max_epsilon=10.0, seed=42).rank(
            _items(), tenant_id="t", epsilon=1.0
        )
        assert a.items == b.items

    def test_release_score_is_noised(self):
        pipe = DPRagPipeline(max_epsilon=10.0, seed=1, score_sensitivity=1.0)
        released = pipe.release_score(0.5, tenant_id="t", epsilon=0.5)
        assert released != 0.5  # Laplace noise applied
        assert pipe.spent("t") == pytest.approx(0.5)


class TestDPRagPipelineValidation:
    def test_non_positive_max_epsilon_rejected(self):
        with pytest.raises(ValueError, match="max_epsilon must be positive"):
            DPRagPipeline(max_epsilon=0.0)

    @pytest.mark.parametrize(
        "kwarg",
        ["retrieval_sensitivity", "decode_sensitivity", "score_sensitivity"],
    )
    def test_negative_sensitivity_rejected(self, kwarg):
        with pytest.raises(ValueError, match=f"{kwarg} must be non-negative"):
            DPRagPipeline(max_epsilon=5.0, **{kwarg: -1.0})

    def test_empty_items_rejected(self):
        with pytest.raises(ValueError, match="items must be non-empty"):
            DPRagPipeline(max_epsilon=5.0).rank([], tenant_id="t", epsilon=1.0)

    def test_empty_logits_rejected(self):
        with pytest.raises(ValueError, match="logits must be non-empty"):
            DPRagPipeline(max_epsilon=5.0).decode([], tenant_id="t", epsilon=1.0)

    def test_non_positive_epsilon_rejected(self):
        pipe = DPRagPipeline(max_epsilon=5.0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            pipe.rank(_items(), tenant_id="t", epsilon=0.0)

    def test_invalid_tenant_id_rejected(self):
        pipe = DPRagPipeline(max_epsilon=5.0)
        with pytest.raises(ValueError, match="invalid tenant_id"):
            pipe.rank(_items(), tenant_id="bad tenant!", epsilon=1.0)

    def test_ranking_and_charge_to_dict_tenant_safe(self):
        pipe = DPRagPipeline(max_epsilon=5.0, seed=1)
        ranking = pipe.rank(_items(), tenant_id="t", epsilon=1.0)
        rpayload = ranking.to_dict()
        assert set(rpayload) == {
            "tenant_id",
            "epsilon_spent",
            "epsilon_remaining",
            "items",
        }
        assert all(set(it) == {"item_id", "score"} for it in rpayload["items"])
        cpayload = pipe.stage_log("t")[0].to_dict()
        assert set(cpayload) == {"stage", "epsilon", "tenant_id"}
