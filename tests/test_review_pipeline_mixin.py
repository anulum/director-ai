# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ReviewPipelineMixin composition and gating contracts

"""Contract tests for the review-orchestration mixin behind CoherenceScorer.

The review pipeline lives in ``director_ai.core.scoring._review_pipeline``
and composes on top of the divergence mixin. These tests pin the mixin
chain, the composite-divergence weighting, cache scoping, the
verified-scorer gating predicates, and the sequential-fallback triggers of
the batch path.
"""

from __future__ import annotations

import pytest

from director_ai.core import CoherenceScorer
from director_ai.core.scoring._divergence import DivergenceMixin
from director_ai.core.scoring._review_pipeline import ReviewPipelineMixin
from director_ai.core.types import EvidenceChunk, ScoringEvidence


class TestReviewPipelineComposition:
    def test_coherence_scorer_composes_both_mixins_in_order(self):
        assert CoherenceScorer.__mro__[1] is ReviewPipelineMixin
        assert CoherenceScorer.__mro__[2] is DivergenceMixin
        for name in (
            "review",
            "review_batch",
            "areview",
            "compute_divergence",
            "_finalise_review",
            "_score_cache_scope",
            "_verified_source_from_evidence",
            "_should_run_verified_scorer",
            "_apply_verified_scorer",
            "_apply_reasoning_tier",
            "_review_batch_requires_sequential",
        ):
            assert getattr(CoherenceScorer, name) is getattr(ReviewPipelineMixin, name)

    def test_compute_divergence_applies_configured_weights(self):
        scorer = CoherenceScorer(use_nli=False, w_logic=0.7, w_fact=0.3)
        scorer.calculate_logical_divergence = lambda _p, _a: 0.8
        scorer.calculate_factual_divergence = lambda _p, _a: 0.2
        assert scorer.compute_divergence("p", "a") == pytest.approx(
            0.7 * 0.8 + 0.3 * 0.2
        )


class TestCacheScope:
    def test_scope_is_empty_without_session_and_store(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._score_cache_scope() == ""

    def test_scope_includes_store_and_session_parts(self):
        class FakeSession:
            context_text = "turn-1"

            def __len__(self):
                return 1

        class FakeStore:
            def cache_scope(self, tenant_id=""):
                return f"v7:{tenant_id}"

        scorer = CoherenceScorer(use_nli=False, ground_truth_store=FakeStore())
        scope = scorer._score_cache_scope(session=FakeSession(), tenant_id="acme")
        assert scope == "session:turn-1\x1fstore:v7:acme"


class TestVerifiedScorerGating:
    def _evidence(self, text="Paris is in France."):
        return ScoringEvidence(
            chunks=[EvidenceChunk(text=text, distance=0.1, source="keyword")],
            nli_premise=text,
            nli_hypothesis="Paris is in France.",
            nli_score=0.1,
        )

    def test_verified_source_prefers_chunks_then_premise(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._verified_source_from_evidence(None) == ""
        assert (
            scorer._verified_source_from_evidence(self._evidence("A fact."))
            == "A fact."
        )
        premise_only = ScoringEvidence(
            chunks=[EvidenceChunk(text="   ", distance=0.1, source="keyword")],
            nli_premise="  premise text  ",
            nli_hypothesis="h",
            nli_score=0.5,
        )
        assert scorer._verified_source_from_evidence(premise_only) == "premise text"

    def test_should_run_verified_scorer_requires_enable_and_source(self):
        scorer = CoherenceScorer(use_nli=False)
        evidence = self._evidence()
        assert (
            scorer._should_run_verified_scorer(
                coherence=0.5, threshold=0.5, task_type="rag", evidence=evidence
            )
            is False
        )
        scorer._verified_scorer_enabled = True
        assert (
            scorer._should_run_verified_scorer(
                coherence=0.5, threshold=0.5, task_type="rag", evidence=None
            )
            is False
        )

    def test_should_run_verified_scorer_on_low_confidence_or_task_route(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._verified_scorer_enabled = True
        evidence = self._evidence()
        assert (
            scorer._should_run_verified_scorer(
                coherence=0.55, threshold=0.5, task_type="default", evidence=evidence
            )
            is True
        )
        assert (
            scorer._should_run_verified_scorer(
                coherence=0.95, threshold=0.5, task_type="rag", evidence=evidence
            )
            is True
        )
        assert (
            scorer._should_run_verified_scorer(
                coherence=0.95, threshold=0.5, task_type="default", evidence=evidence
            )
            is False
        )


class TestBatchSequentialTriggers:
    def test_plain_scorer_batches_short_items(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._review_batch_requires_sequential([("p", "a")]) is False

    def test_long_actions_with_claim_decomposition_force_sequential(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._review_batch_requires_sequential([("p", "x" * 150)]) is True

    def test_custom_aggregation_forces_sequential(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._logic_outer_agg = "mean"
        assert scorer._review_batch_requires_sequential([("p", "a")]) is True

    def test_abstention_threshold_forces_sequential(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._retrieval_abstention_threshold = 0.4
        assert scorer._review_batch_requires_sequential([("p", "a")]) is True


class TestFinaliseReview:
    def test_dry_run_approves_but_logs_the_true_verdict(self):
        scorer = CoherenceScorer(use_nli=False, threshold=0.9)
        scorer._dry_run = True
        approved, score = scorer._finalise_review(0.2, 0.8, 0.8, "action")
        assert approved is True
        assert score.approved is True
        assert score.score == pytest.approx(0.2)

    def test_soft_limit_band_sets_warning_and_history_grows(self):
        scorer = CoherenceScorer(
            use_nli=False, threshold=0.4, soft_limit=0.6, history_window=2
        )
        approved, score = scorer._finalise_review(0.5, 0.2, 0.2, "borderline")
        assert approved is True
        assert score.warning is True
        assert scorer.history == ["borderline"]
        scorer._finalise_review(0.9, 0.1, 0.1, "second")
        scorer._finalise_review(0.9, 0.1, 0.1, "third")
        assert scorer.history == ["second", "third"]
