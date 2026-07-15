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

    def test_soft_limit_override_moves_the_warning_band(self):
        scorer = CoherenceScorer(use_nli=False, threshold=0.4, soft_limit=0.9)
        approved, score = scorer._finalise_review(
            0.5,
            0.2,
            0.2,
            "action",
            threshold_override=0.01,
            soft_limit_override=0.11,
        )
        assert approved is True
        assert score.warning is False


class _RawSupportNLI:
    """Model-backed NLI double for the raw-support review paths."""

    model_available = True

    def __init__(self, divs):
        self.divs = list(divs)
        self.score_calls = []

    def score_claim_coverage(self, premise, response, support_threshold=0.6):
        return 0.5, list(self.divs), [f"claim-{i}" for i in range(len(self.divs))]

    def score(self, premise, hypothesis):
        self.score_calls.append((premise, hypothesis))
        return 0.1

    def _ensure_model(self):
        return True


class _RecordingSession:
    """Session double recording turns without contradiction scoring."""

    context_text = "User: earlier question\nAssistant: earlier answer"
    intent_drift = None

    def __init__(self):
        self.turns = []

    def __len__(self):
        return 1

    def update_contradictions(self, action, scorer_fn):
        del action, scorer_fn
        import types

        return types.SimpleNamespace(contradiction_index=0.0, trend=0.0)

    def add_turn(self, prompt, action, score):
        self.turns.append((prompt, action, score))


_DIALOGUE_PROMPT = "User: hi\nAssistant: hello\nUser: how are you?"


class TestRawSupportOperatingPoints:
    """WCS-2a: matched-FPR support gates on the review path."""

    def _dialogue_scorer(self, divs, **kwargs):
        scorer = CoherenceScorer(use_nli=False, **kwargs)
        scorer._nli = _RawSupportNLI(divs)
        return scorer

    def test_effective_threshold_prefers_the_raw_operating_point(self):
        scorer = self._dialogue_scorer([0.5], threshold=0.6, soft_limit=0.7)
        threshold, soft = scorer._effective_review_threshold(
            _DIALOGUE_PROMPT, "reply", "dialogue"
        )
        assert threshold == pytest.approx(0.0091)
        # The configured soft margin (0.1) carries onto the support scale.
        assert soft == pytest.approx(0.1091)

    def test_effective_threshold_negative_soft_margin_clamps_to_zero(self):
        # The constructor enforces soft_limit >= threshold, but both are
        # mutable attributes — the margin clamp guards that invariant.
        scorer = self._dialogue_scorer([0.5], threshold=0.6, soft_limit=0.7)
        scorer.soft_limit = 0.5
        scorer.threshold = 0.8
        threshold, soft = scorer._effective_review_threshold(
            _DIALOGUE_PROMPT, "reply", "dialogue"
        )
        assert soft == pytest.approx(threshold)

    def test_effective_threshold_uses_adaptive_task_types_off_raw_routes(self):
        scorer = CoherenceScorer(use_nli=False, threshold=0.6)
        scorer._adaptive_threshold_enabled = True
        scorer._task_type_thresholds = {"qa": 0.69}
        scorer._get_meta_classifier = lambda: None
        threshold, soft = scorer._effective_review_threshold(
            "What is the capital of France?", "Paris.", "qa"
        )
        assert threshold == pytest.approx(0.69)
        assert soft is None

    def test_effective_threshold_meta_classifier_overrides_composite_scale(self):
        import types

        scorer = CoherenceScorer(use_nli=False, threshold=0.6)
        scorer._get_meta_classifier = lambda: types.SimpleNamespace(
            predict_threshold=lambda prompt, action: (0.5, 0.9)
        )
        threshold, _ = scorer._effective_review_threshold(
            "What is the capital of France?", "Paris.", "qa"
        )
        assert threshold == pytest.approx(scorer.W_FACT + scorer.W_LOGIC * 0.5)

    def test_effective_threshold_meta_classifier_never_moves_raw_gates(self):
        import types

        scorer = self._dialogue_scorer([0.5])
        scorer._get_meta_classifier = lambda: types.SimpleNamespace(
            predict_threshold=lambda prompt, action: (0.5, 0.9)
        )
        threshold, _ = scorer._effective_review_threshold(
            _DIALOGUE_PROMPT, "reply", "dialogue"
        )
        assert threshold == pytest.approx(0.0091)

    def test_dialogue_review_approves_low_support_above_the_gate(self):
        # Weakest-link support 0.10 would fail any composite threshold;
        # at the matched-FPR gate (0.0091) it is a confident approval.
        scorer = self._dialogue_scorer([0.9], threshold=0.6)
        approved, score = scorer.review(_DIALOGUE_PROMPT, "reply")
        assert approved is True
        assert score.score == pytest.approx(0.1)
        assert score.detected_task_type == "dialogue"

    def test_dialogue_review_rejects_support_below_the_gate(self):
        scorer = self._dialogue_scorer([0.9999], threshold=0.6)
        approved, score = scorer.review(_DIALOGUE_PROMPT, "reply")
        assert approved is False
        assert score.score == pytest.approx(0.0001)

    def test_cache_hit_gates_identically_to_the_fresh_path(self):
        # A decision must not depend on cache state: pre-WCS-2a the hit
        # path re-gated cached scores on the GLOBAL threshold, silently
        # bypassing per-task and raw-support gates.
        scorer = self._dialogue_scorer([0.9], threshold=0.6, cache_size=8)
        fresh_approved, fresh_score = scorer.review(_DIALOGUE_PROMPT, "reply")
        cached_approved, cached_score = scorer.review(_DIALOGUE_PROMPT, "reply")
        assert fresh_approved is True
        assert cached_approved is True
        assert cached_score.score == pytest.approx(fresh_score.score)
        assert cached_score.detected_task_type == "dialogue"

    def test_raw_summarisation_review_skips_the_cross_turn_blend(self):
        scorer = CoherenceScorer(use_nli=False, threshold=0.6)
        scorer._nli = _RawSupportNLI([0.3])
        scorer._summarization_aggregation = "weakest_link"
        session = _RecordingSession()
        approved, score = scorer.review(
            "Summarize the quarterly report.", "Revenue rose.", session=session
        )
        assert approved is True
        assert score.score == pytest.approx(0.7)
        assert score.cross_turn_divergence is None
        # The cross-turn NLI blend never ran on the calibrated support.
        assert scorer._nli.score_calls == []
        assert session.turns and session.turns[0][2] == pytest.approx(0.7)

    def test_review_with_samples_never_blends_raw_supports(self):
        import types

        scorer = self._dialogue_scorer([0.9], threshold=0.6)
        consistency = types.SimpleNamespace(
            consistency_score=0.95,
            semantic_entropy=0.1,
            entailment_backend="fake",
        )
        scorer.enable_self_consistency(
            types.SimpleNamespace(score=lambda action, samples: consistency),
            weight=0.5,
        )
        approved, score = scorer.review_with_samples(
            _DIALOGUE_PROMPT, "reply", ["alt one", "alt two"]
        )
        assert approved is True
        assert score.score == pytest.approx(0.1)
        assert score.self_consistency_score == pytest.approx(0.95)
        assert score.semantic_entropy == pytest.approx(0.1)
        assert score.self_consistency_backend == "fake"

    def test_review_with_samples_still_blends_composite_routes(self):
        import types

        scorer = CoherenceScorer(use_nli=False, threshold=0.4)
        scorer.calculate_logical_divergence = lambda *a, **k: 0.2
        consistency = types.SimpleNamespace(
            consistency_score=0.9,
            semantic_entropy=0.2,
            entailment_backend="fake",
        )
        scorer.enable_self_consistency(
            types.SimpleNamespace(score=lambda action, samples: consistency),
            weight=0.5,
        )
        approved, score = scorer.review_with_samples(
            "What is the capital of France?", "Paris.", ["Paris", "paris"]
        )
        base = scorer.review("What is the capital of France?", "Paris.")[1].score
        assert score.score == pytest.approx(round(0.5 * base + 0.5 * 0.9, 4))
        assert approved is (score.score >= 0.4 or score.approved)
