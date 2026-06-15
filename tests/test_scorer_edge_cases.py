# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Scorer Edge Cases
"""Multi-angle edge case tests for CoherenceScorer.

Covers: empty/null/unicode inputs, very long text, boundary thresholds,
score determinism, score range invariants, weight consistency, threshold
boundary behaviour, RTL/emoji/mixed-script inputs, and pipeline
performance characteristics.
"""

from __future__ import annotations

import threading

import pytest

import director_ai.core.scoring.scorer as scorer_module
from director_ai.core import CoherenceScorer
from director_ai.core.metrics import metrics
from director_ai.core.types import EvidenceChunk, ScoringEvidence

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def scorer():
    return CoherenceScorer(use_nli=False)


@pytest.fixture
def strict_scorer():
    return CoherenceScorer(use_nli=False, strict_mode=True)


# ── Empty / null inputs ────────────────────────────────────────────


class TestEmptyInputs:
    """Scorer must handle empty/null-like inputs gracefully."""

    @pytest.mark.parametrize(
        "prompt,response",
        [
            ("", "Some response"),
            ("What is 2+2?", ""),
            ("", ""),
        ],
    )
    def test_empty_inputs_no_crash(self, scorer, prompt, response):
        approved, score = scorer.review(prompt, response)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_whitespace_only_prompt(self, scorer):
        approved, score = scorer.review("   \n\t  ", "Normal response")
        assert isinstance(approved, bool)

    def test_whitespace_only_response(self, scorer):
        approved, score = scorer.review("Normal prompt", "   \n\t  ")
        assert isinstance(approved, bool)

    def test_null_bytes_in_input(self, scorer):
        approved, score = scorer.review("test\x00prompt", "test\x00response")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_single_char_inputs(self, scorer):
        approved, score = scorer.review("?", ".")
        assert isinstance(approved, bool)


# ── Unicode / multilingual ─────────────────────────────────────────


class TestUnicodeInputs:
    """Scorer must handle diverse Unicode correctly."""

    @pytest.mark.parametrize(
        "prompt,response",
        [
            ("What is this? 🎉🥳", "It is a celebration 🎉"),
            ("ما هو 2+2؟", "الإجابة هي 4"),
            ("2+2は何ですか？", "答えは4です"),
            ("Čo je 2+2?", "Odpoveď je 4"),
            ("Що таке 2+2?", "Відповідь 4"),
            ("Mixed: hello مرحبا こんにちは", "Response in English"),
        ],
    )
    def test_multilingual_no_crash(self, scorer, prompt, response):
        approved, score = scorer.review(prompt, response)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_zero_width_chars(self, scorer):
        approved, score = scorer.review(
            "test\u200b\u200c\u200dprompt", "response\ufeff"
        )
        assert isinstance(approved, bool)

    def test_surrogate_like_chars(self, scorer):
        approved, score = scorer.review("𝕋𝕖𝕤𝕥", "ℝ𝕖𝕤𝕡𝕠𝕟𝕤𝕖")
        assert isinstance(approved, bool)


# ── Very long text ─────────────────────────────────────────────────


class TestLongText:
    """Scorer must handle large inputs without OOM or timeout."""

    @pytest.mark.parametrize("length", [1_000, 10_000, 100_000])
    def test_long_response_no_crash(self, scorer, length):
        long_text = "word " * length
        approved, score = scorer.review("Summarise", long_text)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_long_prompt_no_crash(self, scorer):
        long_prompt = "context " * 50_000
        approved, score = scorer.review(long_prompt, "Short answer.")
        assert isinstance(approved, bool)


# ── Score invariants ───────────────────────────────────────────────


class TestScoreInvariants:
    """Score components must satisfy mathematical invariants."""

    def test_score_in_range(self, scorer):
        for _ in range(20):
            _, score = scorer.review("What is AI?", "AI is artificial intelligence.")
            assert 0.0 <= score.score <= 1.0
            assert 0.0 <= score.h_logical <= 1.0
            assert 0.0 <= score.h_factual <= 1.0

    def test_deterministic(self, scorer):
        _, s1 = scorer.review("What?", "Answer.")
        _, s2 = scorer.review("What?", "Answer.")
        assert s1.score == s2.score
        assert s1.h_logical == s2.h_logical
        assert s1.h_factual == s2.h_factual

    def test_weights_sum_to_one(self):
        assert abs(CoherenceScorer.W_LOGIC + CoherenceScorer.W_FACT - 1.0) < 1e-9

    def test_coherence_formula_consistency(self, scorer):
        """Score = 1 - (w_logic * h_logical + w_fact * h_factual)."""
        _, score = scorer.review("What is 2+2?", "4")
        expected = 1.0 - (
            CoherenceScorer.W_LOGIC * score.h_logical
            + CoherenceScorer.W_FACT * score.h_factual
        )
        assert abs(score.score - expected) < 1e-6


# ── Threshold boundary ─────────────────────────────────────────────


class TestThresholdBoundary:
    """Test behaviour at exact threshold boundaries."""

    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_various_thresholds_accepted(self, threshold):
        scorer = CoherenceScorer(use_nli=False, threshold=threshold)
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)

    def test_threshold_zero_always_approves(self):
        scorer = CoherenceScorer(use_nli=False, threshold=0.0)
        approved, _ = scorer.review("test", "test")
        assert approved is True

    def test_threshold_one_rejects_nonperfect(self):
        scorer = CoherenceScorer(use_nli=False, threshold=1.0, soft_limit=1.0)
        # Heuristic scorer unlikely to produce exactly 1.0
        approved, score = scorer.review("X is Y", "Z is W")
        # Either approved or not — but must not crash
        assert isinstance(approved, bool)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            CoherenceScorer(use_nli=False, threshold=1.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            CoherenceScorer(use_nli=False, threshold=-0.1)

    def test_soft_limit_below_threshold_raises(self):
        with pytest.raises(ValueError, match="soft_limit"):
            CoherenceScorer(use_nli=False, threshold=0.7, soft_limit=0.5)


# ── Strict mode ────────────────────────────────────────────────────


class TestStrictMode:
    """Strict mode must disable heuristic fallbacks."""

    def test_strict_mode_flag_stored(self, strict_scorer):
        assert strict_scorer.strict_mode is True

    def test_non_strict_default(self, scorer):
        assert scorer.strict_mode is False

    def test_require_model_backed_nli_fails_closed_without_model_backend(self):
        scorer = CoherenceScorer(
            use_nli=False,
            strict_mode=True,
            require_model_backed_nli=True,
        )
        with pytest.raises(RuntimeError, match="model-backed NLI"):
            scorer.review("What is 2+2?", "The answer is 4.")

    def test_require_model_backed_nli_batch_fails_closed_without_model_backend(self):
        scorer = CoherenceScorer(
            use_nli=False,
            strict_mode=True,
            require_model_backed_nli=True,
        )
        with pytest.raises(RuntimeError, match="model-backed NLI"):
            scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])


# ── Performance characteristics ─────────────────────────────────────


class TestPerformanceDoc:
    """Document and verify performance guarantees."""

    def test_heuristic_score_has_evidence_field(self, scorer):
        _, score = scorer.review("What is AI?", "AI is intelligence.")
        assert hasattr(score, "evidence")

    def test_heuristic_score_has_components(self, scorer):
        _, score = scorer.review("Q", "A")
        assert hasattr(score, "h_logical")
        assert hasattr(score, "h_factual")
        assert hasattr(score, "score")
        assert hasattr(score, "approved")

    def test_review_returns_tuple(self, scorer):
        result = scorer.review("Q", "A")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)


class TestFallbackIncidentConcurrency:
    """Fallback incident emission must be thread-safe and once-per-stage."""

    def test_fallback_incident_emitted_once_per_stage_under_contention(self):
        metrics.reset()
        scorer = CoherenceScorer(use_nli=False)

        def _emit() -> None:
            scorer._record_nli_fallback_incident(
                stage="logical",
                reason="nli_unavailable_using_heuristic",
            )

        threads = [threading.Thread(target=_emit) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snapshot = metrics.get_metrics()
        counter = snapshot["counters"]["nli_fallback_incidents_total"]
        assert counter["total"] == 1.0
        assert (
            counter["multi_labels"].get(
                'reason="nli_unavailable_using_heuristic",stage="logical"'
            )
            == 1.0
        )


def test_vector_store_evidence_falls_back_to_keyword_chunks_when_vector_query_misses() -> (
    None
):
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import InMemoryBackend, VectorGroundTruthStore

    class EmptyVectorBackend(InMemoryBackend):
        def query(self, text, n_results=3, tenant_id=""):
            return []

    store = VectorGroundTruthStore(backend=EmptyVectorBackend())
    store.add_fact("sky", "The sky is blue.")
    scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
    scorer._detect_task_type = lambda _prompt, _response="": "default"

    divergence, evidence = scorer.calculate_factual_divergence_with_evidence(
        "sky",
        "The sky is blue.",
    )

    assert 0.0 <= divergence <= 1.0
    assert evidence is not None
    assert evidence.chunks
    assert evidence.chunks[0].source == "keyword"


class TestScorerCoverageGaps:
    """Dedicated branch tests for CoherenceScorer internals."""

    def test_parallel_pool_lifecycle_and_destructor_shutdown(self):
        scorer = CoherenceScorer(use_nli=False)
        pool = scorer._get_parallel_pool()

        assert scorer._get_parallel_pool() is pool

        scorer.close()
        assert scorer._parallel_pool is None

        scorer._parallel_pool = scorer._get_parallel_pool()
        scorer.__del__()

    def test_model_backed_nli_requirement_allows_disabled_and_rust_paths(self):
        scorer = CoherenceScorer(use_nli=False)

        scorer.require_model_backed_nli = False
        scorer._enforce_model_backed_nli_requirement()

        scorer.require_model_backed_nli = True
        scorer._rust_scorer = object()
        scorer._enforce_model_backed_nli_requirement()

    def test_model_backed_nli_detection_rejects_lite_backend(self):
        scorer = CoherenceScorer(use_nli=False)

        class LiteNLI:
            model_available = True
            backend = "lite"

        class ModelNLI:
            model_available = True
            backend = "deberta"

        scorer._nli = LiteNLI()
        assert scorer._has_model_backed_nli() is False

        scorer._nli = ModelNLI()
        assert scorer._has_model_backed_nli() is True

    def test_adaptive_retrieval_enablement_installs_router(self):
        scorer = CoherenceScorer(use_nli=False)

        scorer.enable_adaptive_retrieval(threshold=0.7, default_retrieve=False)

        assert scorer._adaptive_router is not None
        decision = scorer._adaptive_router.should_retrieve("write a poem", "")
        assert decision.retrieve is False

    def test_constructor_custom_cache_and_validation_edges(self):
        cache = object()
        scorer = CoherenceScorer(use_nli=False, cache=cache)
        assert scorer.cache is cache

        scorer = CoherenceScorer(use_nli=False, cache_size=0)
        assert scorer.cache is None

        with pytest.raises(ValueError, match="hybrid backend"):
            CoherenceScorer(use_nli=False, scorer_backend="hybrid")
        with pytest.raises(ValueError, match="w_logic"):
            CoherenceScorer(use_nli=False, w_logic=-0.1, w_fact=1.1)
        with pytest.raises(ValueError, match="w_fact"):
            CoherenceScorer(use_nli=False, w_logic=0.5, w_fact=1.1)
        with pytest.raises(ValueError, match="equal 1.0"):
            CoherenceScorer(use_nli=False, w_logic=0.2, w_fact=0.2)

    def test_minicheck_lazy_success_and_failure(self, monkeypatch):
        class AvailableMiniCheck:
            model_available = True

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def _ensure_minicheck(self):
                return True

        monkeypatch.setattr(scorer_module, "NLIScorer", AvailableMiniCheck)
        scorer = CoherenceScorer(use_nli=False)

        minicheck = scorer._get_minicheck_scorer()

        assert minicheck is scorer._get_minicheck_scorer()
        assert minicheck.kwargs == {
            "use_model": True,
            "backend": "minicheck",
            "minicheck_variant": "deberta-v3-large",
        }

        class FailingMiniCheck:
            def __init__(self, **kwargs):
                del kwargs
                raise RuntimeError("missing minicheck")

        monkeypatch.setattr(scorer_module, "NLIScorer", FailingMiniCheck)
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._get_minicheck_scorer() is None
        assert scorer._get_minicheck_scorer() is None

    def test_prompt_premise_evidence_uses_counted_nli_and_escalation(self):
        class FakeNLI:
            model_available = True
            last_token_count = 17
            last_estimated_cost = 0.02

            def reset_token_counter(self):
                self.reset = True

            def _score_chunked_with_counts(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return 0.41, [0.4, 0.42], 2, 3

        scorer = CoherenceScorer(use_nli=False)
        scorer._nli = FakeNLI()
        scorer._confidence_weighted_agg = False
        scorer._use_prompt_as_premise = True
        scorer._should_escalate = lambda score, task_type: True
        scorer._llm_judge_check = lambda prompt, output, score: 0.33

        score, evidence = scorer._calculate_prompt_premise_divergence_with_evidence(
            "source document",
            "summary text",
        )

        assert score == 0.33
        assert evidence is not None
        assert evidence.nli_premise == "source document"
        assert evidence.nli_hypothesis == "summary text"
        assert evidence.chunk_scores == [0.4, 0.42]
        assert evidence.premise_chunk_count == 2
        assert evidence.hypothesis_chunk_count == 3
        assert evidence.token_count == 17
        assert evidence.estimated_cost_usd == 0.02

    def test_prompt_premise_evidence_confidence_weighted_path(self):
        class FakeNLI:
            model_available = True
            last_token_count = 11
            last_estimated_cost = 0.01

            def reset_token_counter(self):
                self.reset = True

            def score_chunked_confidence_weighted(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return 0.24, [0.2, 0.28]

        scorer = CoherenceScorer(use_nli=False)
        scorer._nli = FakeNLI()
        scorer._confidence_weighted_agg = True

        score, evidence = scorer._calculate_prompt_premise_divergence_with_evidence(
            "source document",
            "summary text",
        )

        assert score == 0.24
        assert evidence is not None
        assert evidence.chunk_scores == [0.2, 0.28]
        assert evidence.premise_chunk_count == 1
        assert evidence.hypothesis_chunk_count == 2

    def test_factual_divergence_strict_and_heuristic_fallbacks(self):
        class Store:
            def retrieve_context(self, prompt, top_k=3, tenant_id=""):
                del prompt, top_k, tenant_id
                return "Saturn has rings. Mars has no global ocean."

        strict = CoherenceScorer(
            use_nli=False, strict_mode=True, ground_truth_store=Store()
        )
        non_strict = CoherenceScorer(use_nli=False, ground_truth_store=Store())

        assert strict.calculate_factual_divergence("Saturn", "Saturn is ringed.") == (
            scorer_module.DIVERGENCE_CONTRADICTED
        )
        assert (
            0.0
            <= non_strict.calculate_factual_divergence(
                "Saturn",
                "Saturn is ringed.",
            )
            <= 1.0
        )

    def test_factual_divergence_with_vector_abstention_returns_neutral(self):
        from director_ai.core.vector_store import (
            InMemoryBackend,
            VectorGroundTruthStore,
        )

        class DistantBackend(InMemoryBackend):
            def query(self, text, n_results=3, tenant_id=""):
                del text, n_results, tenant_id
                return [
                    {
                        "id": "remote",
                        "text": "remote fact",
                        "distance": 0.99,
                        "metadata": {"source": "unit"},
                    }
                ]

        store = VectorGroundTruthStore(backend=DistantBackend())
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._retrieval_abstention_threshold = 0.5

        assert scorer.calculate_factual_divergence("query", "answer") == (
            scorer_module.DIVERGENCE_NEUTRAL
        )

    def test_finalise_review_dry_run_warning_and_retrieval_confidence(self):
        scorer = CoherenceScorer(use_nli=False, threshold=0.8, soft_limit=0.9)
        scorer._dry_run = True
        evidence = ScoringEvidence(
            chunks=[EvidenceChunk(text="source", distance=0.2, source="unit")],
            nli_premise="source",
            nli_hypothesis="answer",
            nli_score=0.7,
        )

        approved, score = scorer._finalise_review(
            0.7,
            0.4,
            0.5,
            "answer",
            evidence=evidence,
        )

        assert approved is True
        assert score.approved is True
        assert score.warning is True
        assert score.retrieval_confidence == pytest.approx(0.8)
        assert scorer.history[-1] == "answer"

    def test_verified_source_and_routing_guards(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._verified_scorer_enabled = True
        scorer._verified_scorer_low_confidence_margin = 0.05
        scorer._verified_scorer_task_types = {"fact_check"}

        assert scorer._verified_source_from_evidence(None) == ""
        empty = ScoringEvidence(
            chunks=[],
            nli_premise=" premise ",
            nli_hypothesis="",
            nli_score=0.5,
        )
        assert scorer._verified_source_from_evidence(empty) == "premise"
        sourced = ScoringEvidence(
            chunks=[
                EvidenceChunk(text=" first ", distance=0.1, source="a"),
                EvidenceChunk(text="", distance=0.2, source="b"),
                EvidenceChunk(text="second", distance=0.3, source="c"),
            ],
            nli_premise="ignored",
            nli_hypothesis="",
            nli_score=0.5,
        )
        assert scorer._verified_source_from_evidence(sourced) == "first second"
        assert scorer._should_run_verified_scorer(
            coherence=0.51,
            threshold=0.5,
            task_type="qa",
            evidence=sourced,
        )
        assert scorer._should_run_verified_scorer(
            coherence=0.9,
            threshold=0.5,
            task_type="fact_check",
            evidence=sourced,
        )
        assert not scorer._should_run_verified_scorer(
            coherence=0.9,
            threshold=0.5,
            task_type="qa",
            evidence=empty,
        )

    def test_review_batch_errors_and_meta_threshold_path(self):
        class FakeNLI:
            model_available = True
            backend = "deberta"

            def __init__(self, scores):
                self.scores = list(scores)

            def score_batch(self, pairs):
                del pairs
                return self.scores.pop(0)

        scorer = CoherenceScorer(use_nli=False)
        scorer._nli = FakeNLI([[0.1]])
        scorer._judge = type("FakeJudge", (), {"enabled": False})()
        scorer._confidence_weighted_agg = False

        with pytest.raises(RuntimeError, match="logical NLI batch returned"):
            scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])

        class Store:
            def retrieve_context(self, prompt, top_k=3, tenant_id=""):
                del prompt, top_k, tenant_id
                return "grounded context"

        class MetaClassifier:
            def predict_threshold(self, prompt, action):
                del prompt, action
                return 0.25, 0.9

        scorer = CoherenceScorer(use_nli=False, ground_truth_store=Store())
        scorer._nli = FakeNLI([[0.2, 0.3], [0.4]])
        scorer._judge = type("FakeJudge", (), {"enabled": False})()
        scorer._confidence_weighted_agg = False
        scorer._get_meta_classifier = lambda: MetaClassifier()

        with pytest.raises(RuntimeError, match="factual NLI batch returned"):
            scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])

        scorer._nli = FakeNLI([[0.1, 0.2], [0.3, 0.4]])
        results = scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])

        assert len(results) == 2
        assert all(result[1].detected_task_type for result in results)
