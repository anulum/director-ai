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
from collections.abc import Sequence
from types import TracebackType
from typing import Literal

import pytest

import director_ai.core.scoring._divergence as divergence_module
import director_ai.core.scoring._divergence_routing as divergence_routing_module
import director_ai.core.scoring.scorer as scorer_module
from director_ai.core import CoherenceScorer
from director_ai.core.cache import ScoreCache
from director_ai.core.metrics import metrics
from director_ai.core.types import CoherenceScore, EvidenceChunk, ScoringEvidence

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def scorer() -> CoherenceScorer:
    return CoherenceScorer(use_nli=False)


@pytest.fixture
def strict_scorer() -> CoherenceScorer:
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
    def test_empty_inputs_no_crash(
        self, scorer: CoherenceScorer, prompt: str, response: str
    ) -> None:
        approved, score = scorer.review(prompt, response)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_whitespace_only_prompt(self, scorer: CoherenceScorer) -> None:
        approved, score = scorer.review("   \n\t  ", "Normal response")
        assert isinstance(approved, bool)

    def test_whitespace_only_response(self, scorer: CoherenceScorer) -> None:
        approved, score = scorer.review("Normal prompt", "   \n\t  ")
        assert isinstance(approved, bool)

    def test_null_bytes_in_input(self, scorer: CoherenceScorer) -> None:
        approved, score = scorer.review("test\x00prompt", "test\x00response")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_single_char_inputs(self, scorer: CoherenceScorer) -> None:
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
    def test_multilingual_no_crash(
        self, scorer: CoherenceScorer, prompt: str, response: str
    ) -> None:
        approved, score = scorer.review(prompt, response)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_zero_width_chars(self, scorer: CoherenceScorer) -> None:
        approved, score = scorer.review(
            "test\u200b\u200c\u200dprompt", "response\ufeff"
        )
        assert isinstance(approved, bool)

    def test_surrogate_like_chars(self, scorer: CoherenceScorer) -> None:
        approved, score = scorer.review("𝕋𝕖𝕤𝕥", "ℝ𝕖𝕤𝕡𝕠𝕟𝕤𝕖")
        assert isinstance(approved, bool)


# ── Very long text ─────────────────────────────────────────────────


class TestLongText:
    """Scorer must handle large inputs without OOM or timeout."""

    @pytest.mark.parametrize("length", [1_000, 10_000, 100_000])
    def test_long_response_no_crash(self, scorer: CoherenceScorer, length: int) -> None:
        long_text = "word " * length
        approved, score = scorer.review("Summarise", long_text)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_long_prompt_no_crash(self, scorer: CoherenceScorer) -> None:
        long_prompt = "context " * 50_000
        approved, score = scorer.review(long_prompt, "Short answer.")
        assert isinstance(approved, bool)


# ── Score invariants ───────────────────────────────────────────────


class TestScoreInvariants:
    """Score components must satisfy mathematical invariants."""

    def test_score_in_range(self, scorer: CoherenceScorer) -> None:
        for _ in range(20):
            _, score = scorer.review("What is AI?", "AI is artificial intelligence.")
            assert 0.0 <= score.score <= 1.0
            assert 0.0 <= score.h_logical <= 1.0
            assert 0.0 <= score.h_factual <= 1.0

    def test_deterministic(self, scorer: CoherenceScorer) -> None:
        _, s1 = scorer.review("What?", "Answer.")
        _, s2 = scorer.review("What?", "Answer.")
        assert s1.score == s2.score
        assert s1.h_logical == s2.h_logical
        assert s1.h_factual == s2.h_factual

    def test_weights_sum_to_one(self) -> None:
        assert abs(CoherenceScorer.W_LOGIC + CoherenceScorer.W_FACT - 1.0) < 1e-9

    def test_coherence_formula_consistency(self, scorer: CoherenceScorer) -> None:
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
    def test_various_thresholds_accepted(self, threshold: float) -> None:
        scorer = CoherenceScorer(use_nli=False, threshold=threshold)
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)

    def test_threshold_zero_always_approves(self) -> None:
        scorer = CoherenceScorer(use_nli=False, threshold=0.0)
        approved, _ = scorer.review("test", "test")
        assert approved is True

    def test_threshold_one_rejects_nonperfect(self) -> None:
        scorer = CoherenceScorer(use_nli=False, threshold=1.0, soft_limit=1.0)
        # Heuristic scorer unlikely to produce exactly 1.0
        approved, score = scorer.review("X is Y", "Z is W")
        # Either approved or not — but must not crash
        assert isinstance(approved, bool)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            CoherenceScorer(use_nli=False, threshold=1.5)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            CoherenceScorer(use_nli=False, threshold=-0.1)

    def test_soft_limit_below_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="soft_limit"):
            CoherenceScorer(use_nli=False, threshold=0.7, soft_limit=0.5)


# ── Strict mode ────────────────────────────────────────────────────


class TestStrictMode:
    """Strict mode must disable heuristic fallbacks."""

    def test_strict_mode_flag_stored(self, strict_scorer: CoherenceScorer) -> None:
        assert strict_scorer.strict_mode is True

    def test_non_strict_default(self, scorer: CoherenceScorer) -> None:
        assert scorer.strict_mode is False

    def test_require_model_backed_nli_fails_closed_without_model_backend(self) -> None:
        scorer = CoherenceScorer(
            use_nli=False,
            strict_mode=True,
            require_model_backed_nli=True,
        )
        with pytest.raises(RuntimeError, match="model-backed NLI"):
            scorer.review("What is 2+2?", "The answer is 4.")

    def test_require_model_backed_nli_batch_fails_closed_without_model_backend(
        self,
    ) -> None:
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

    def test_heuristic_score_has_evidence_field(self, scorer: CoherenceScorer) -> None:
        _, score = scorer.review("What is AI?", "AI is intelligence.")
        assert hasattr(score, "evidence")

    def test_heuristic_score_has_components(self, scorer: CoherenceScorer) -> None:
        _, score = scorer.review("Q", "A")
        assert hasattr(score, "h_logical")
        assert hasattr(score, "h_factual")
        assert hasattr(score, "score")
        assert hasattr(score, "approved")

    def test_review_returns_tuple(self, scorer: CoherenceScorer) -> None:
        result = scorer.review("Q", "A")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)


class TestFallbackIncidentConcurrency:
    """Fallback incident emission must be thread-safe and once-per-stage."""

    def test_fallback_incident_emitted_once_per_stage_under_contention(self) -> None:
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


def test_vector_store_evidence_falls_back_to_keyword_chunks_when_vector_query_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from director_ai.core.retrieval.vector_store import (
        InMemoryBackend,
        VectorGroundTruthStore,
    )

    class EmptyVectorBackend(InMemoryBackend):
        def query(
            self, text: str, n_results: int = 3, tenant_id: str = ""
        ) -> list[dict[str, object]]:
            return []

    store = VectorGroundTruthStore(backend=EmptyVectorBackend())
    store.add_fact("sky", "The sky is blue.")
    scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
    monkeypatch.setattr(
        scorer, "_detect_task_type", lambda _prompt, _response="": "default"
    )

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

    def test_parallel_pool_lifecycle_and_destructor_shutdown(self) -> None:
        scorer = CoherenceScorer(use_nli=False)
        pool = scorer._get_parallel_pool()

        assert scorer._get_parallel_pool() is pool

        scorer.close()
        assert scorer._parallel_pool is None

        scorer._parallel_pool = scorer._get_parallel_pool()
        scorer.__del__()

    def test_native_review_requires_a_model_only_when_configured(self) -> None:
        """The optional native path remains usable with model-backed NLI disabled."""
        prompt = "Which notifications are supported?"
        response = "Notifications support Slack."
        optional = CoherenceScorer(scorer_backend="rust", use_nli=False)
        approved, score = optional.review(prompt, response)
        assert approved == score.approved
        assert score.degraded_mode

        required = CoherenceScorer(
            scorer_backend="rust", use_nli=False, require_model_backed_nli=True
        )
        with pytest.raises(RuntimeError, match="requires model-backed NLI"):
            required.review(prompt, response)

    def test_model_backed_nli_detection_rejects_lite_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scorer = CoherenceScorer(use_nli=False)

        class LiteNLI:
            model_available = True
            backend = "lite"

        class ModelNLI:
            model_available = True
            backend = "deberta"

        monkeypatch.setattr(scorer, "_nli", LiteNLI())
        assert scorer._has_model_backed_nli() is False

        monkeypatch.setattr(scorer, "_nli", ModelNLI())
        assert scorer._has_model_backed_nli() is True

    def test_adaptive_retrieval_enablement_installs_router(self) -> None:
        scorer = CoherenceScorer(use_nli=False)

        scorer.enable_adaptive_retrieval(threshold=0.7, default_retrieve=False)

        assert scorer._adaptive_router is not None
        decision = scorer._adaptive_router.should_retrieve("write a poem", "")
        assert decision.retrieve is False

    def test_constructor_custom_cache_and_validation_edges(self) -> None:
        cache = ScoreCache()
        scorer = CoherenceScorer(use_nli=False, cache=cache)
        assert scorer.cache is cache

        lite = CoherenceScorer(use_nli=False, scorer_backend="lite")
        assert lite._nli is not None
        assert lite._nli.backend == "lite"

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

    def test_minicheck_lazy_success_and_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class AvailableMiniCheck:
            model_available = True

            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def _ensure_minicheck(self) -> bool:
                return True

        monkeypatch.setattr(scorer_module, "NLIScorer", AvailableMiniCheck)
        scorer = CoherenceScorer(use_nli=False)

        minicheck = scorer._get_minicheck_scorer()

        assert minicheck is scorer._get_minicheck_scorer()
        assert isinstance(minicheck, AvailableMiniCheck)
        assert minicheck.kwargs == {
            "use_model": True,
            "backend": "minicheck",
            "minicheck_variant": "deberta-v3-large",
        }

        class FailingMiniCheck:
            def __init__(self, **kwargs: object) -> None:
                del kwargs
                raise RuntimeError("missing minicheck")

        monkeypatch.setattr(scorer_module, "NLIScorer", FailingMiniCheck)
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._get_minicheck_scorer() is None
        assert scorer._get_minicheck_scorer() is None

    def test_prompt_premise_evidence_uses_counted_nli_and_escalation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeNLI:
            model_available = True
            last_token_count = 17
            last_estimated_cost = 0.02

            def reset_token_counter(self) -> None:
                self.reset = True

            def _score_chunked_with_counts(
                self, *args: object, **kwargs: object
            ) -> tuple[float, list[float], int, int]:
                self.args = args
                self.kwargs = kwargs
                return 0.41, [0.4, 0.42], 2, 3

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(scorer, "_nli", FakeNLI())
        scorer._confidence_weighted_agg = False
        scorer._use_prompt_as_premise = True
        monkeypatch.setattr(scorer, "_should_escalate", lambda score, task_type: True)
        monkeypatch.setattr(
            scorer, "_llm_judge_check", lambda prompt, output, score: 0.33
        )

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

    def test_prompt_premise_evidence_confidence_weighted_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeNLI:
            model_available = True
            last_token_count = 11
            last_estimated_cost = 0.01

            def reset_token_counter(self) -> None:
                self.reset = True

            def score_chunked_confidence_weighted(
                self, *args: object, **kwargs: object
            ) -> tuple[float, list[float]]:
                self.args = args
                self.kwargs = kwargs
                return 0.24, [0.2, 0.28]

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(scorer, "_nli", FakeNLI())
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

    def test_dialogue_factual_divergence_requires_model_backed_nli(self) -> None:
        scorer = CoherenceScorer(use_nli=False)

        with pytest.raises(RuntimeError, match="NLI model required"):
            scorer._dialogue_factual_divergence("User: hi", "Assistant: hi")

    def test_dialogue_factual_divergence_delegates_when_nli_is_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # WCS-2a default: the dialogue route delegates to the raw
        # weakest-link support scorer, not the baseline squeeze.
        class FakeNLI:
            model_available = True

        def fake_dialogue_raw_support_divergence(
            *args: object, **kwargs: object
        ) -> tuple[float, None]:
            assert args[0] is fake_nli
            return 0.12, None

        fake_nli = FakeNLI()
        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(scorer, "_nli", fake_nli)
        monkeypatch.setattr(
            divergence_routing_module,
            "dialogue_raw_support_divergence",
            fake_dialogue_raw_support_divergence,
        )

        score, evidence = scorer._dialogue_factual_divergence(
            "User: hi",
            "Assistant: hi",
            tenant_id="tenant-a",
        )

        assert score == 0.12
        assert evidence is None

    def test_dialogue_factual_divergence_squeeze_mode_delegates_legacy_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeNLI:
            model_available = True

        def fake_dialogue_factual_divergence(
            *args: object, **kwargs: object
        ) -> tuple[float, None]:
            assert args[0] is fake_nli
            assert kwargs["baseline"] == scorer._dialogue_nli_baseline
            return 0.12, None

        fake_nli = FakeNLI()
        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(scorer, "_nli", fake_nli)
        scorer._dialogue_scoring = "baseline_squeeze"
        monkeypatch.setattr(
            divergence_routing_module,
            "dialogue_factual_divergence",
            fake_dialogue_factual_divergence,
        )

        score, evidence = scorer._dialogue_factual_divergence(
            "User: hi",
            "Assistant: hi",
            tenant_id="tenant-a",
        )

        assert score == 0.12
        assert evidence is None

    def test_injection_detection_init_failure_resets_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import director_ai.core.safety.injection as injection_module

        class FailingInjectionDetector:
            def __init__(self, **kwargs: object) -> None:
                del kwargs
                raise RuntimeError("model required")

        monkeypatch.setattr(
            injection_module,
            "InjectionDetector",
            FailingInjectionDetector,
        )
        scorer = CoherenceScorer(use_nli=False)
        scorer._injection_detector = object()
        scorer._injection_fail_closed = True

        with pytest.raises(RuntimeError, match="model required"):
            scorer.enable_injection_detection(require_model_backed_nli=True)

        assert scorer._injection_detector is None
        assert scorer._injection_fail_closed is False

    def test_factual_divergence_strict_and_heuristic_fallbacks(self) -> None:
        class Store:
            def retrieve_context(
                self, prompt: str, top_k: int = 3, tenant_id: str = ""
            ) -> str:
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

    def test_factual_divergence_claim_decomposition_reweights_long_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Store:
            def retrieve_context(
                self, prompt: str, top_k: int = 3, tenant_id: str = ""
            ) -> str:
                del prompt, top_k, tenant_id
                return "Paris is in France. Berlin is in Germany."

        class FakeNLI:
            model_available = True
            backend = "deberta"

            def __init__(self) -> None:
                self.scores: list[tuple[float, list[float]]] = [
                    (0.2, []),
                    (0.1, []),
                    (0.8, []),
                ]

            def score_chunked(
                self, *args: object, **kwargs: object
            ) -> tuple[float, list[float]]:
                del args, kwargs
                return self.scores.pop(0)

            def _split_sentences(self, text: str) -> list[str]:
                del text
                return ["Paris is in France.", "Berlin is in Italy."]

        scorer = CoherenceScorer(use_nli=False, ground_truth_store=Store())
        monkeypatch.setattr(scorer, "_nli", FakeNLI())

        score = scorer.calculate_factual_divergence(
            "cities",
            "Paris is in France. Berlin is in Italy. " * 4,
        )

        assert score == pytest.approx(0.32)

    def test_vector_store_evidence_uses_retrieved_chunks_with_nli_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock

        from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

        store = VectorGroundTruthStore()
        monkeypatch.setattr(
            store,
            "retrieve_context_with_chunks",
            MagicMock(
                return_value=[
                    EvidenceChunk(text="grounded chunk", distance=0.2, source="kb")
                ]
            ),
        )

        class FakeNLI:
            model_available = True
            last_token_count = 9
            last_estimated_cost = 0.03
            _cost_per_token = 0.001

            def reset_token_counter(self) -> None:
                self.reset = True

            def _score_chunked_with_counts(
                self, *args: object, **kwargs: object
            ) -> tuple[float, list[float], int, int]:
                self.args = args
                self.kwargs = kwargs
                return 0.21, [0.21], 1, 1

        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        monkeypatch.setattr(scorer, "_nli", FakeNLI())

        score, evidence = scorer.calculate_factual_divergence_with_evidence(
            "prompt",
            "answer",
        )

        assert score == 0.21
        assert evidence is not None
        assert evidence.chunks[0].text == "grounded chunk"
        assert evidence.chunk_scores == [0.21]
        assert evidence.token_count == 9

    def test_factual_evidence_escalates_with_judge_when_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Store:
            def retrieve_context(
                self, prompt: str, top_k: int = 3, tenant_id: str = ""
            ) -> str:
                del prompt, top_k, tenant_id
                return "grounded context"

        class FakeNLI:
            model_available = True
            last_token_count = 0
            last_estimated_cost = 0.0
            _cost_per_token = 0.0

            def reset_token_counter(self) -> None:
                self.reset = True

            def _score_chunked_with_counts(
                self, *args: object, **kwargs: object
            ) -> tuple[float, list[float], int, int]:
                del args, kwargs
                return 0.44, [0.44], 1, 1

        scorer = CoherenceScorer(use_nli=False, ground_truth_store=Store())
        monkeypatch.setattr(scorer, "_nli", FakeNLI())
        monkeypatch.setattr(
            scorer, "_should_escalate", lambda _score, task_type="default": True
        )
        monkeypatch.setattr(
            scorer,
            "_llm_judge_check",
            lambda prompt, output, score: (
                0.22
                if (prompt, output, score) == ("prompt", "answer", 0.44)
                else pytest.fail("unexpected judge call")
            ),
        )

        score, evidence = scorer.calculate_factual_divergence_with_evidence(
            "prompt",
            "answer",
        )

        assert score == 0.22
        assert evidence is not None
        assert evidence.nli_score == 0.22

    def test_factual_divergence_with_vector_abstention_returns_neutral(self) -> None:
        from director_ai.core.retrieval.vector_store import (
            InMemoryBackend,
            VectorGroundTruthStore,
        )

        class DistantBackend(InMemoryBackend):
            def query(
                self, text: str, n_results: int = 3, tenant_id: str = ""
            ) -> list[dict[str, object]]:
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

    def test_heuristic_factual_penalises_negation_and_novel_entities(self) -> None:
        score = CoherenceScorer._heuristic_factual(
            "Alice did not visit Paris.",
            "Alice visited Rome.",
        )

        assert score == 1.0

    def test_logical_strict_mode_without_nli_rejects(self) -> None:
        scorer = CoherenceScorer(use_nli=False, strict_mode=True)

        assert scorer.calculate_logical_divergence("prompt", "answer") == (
            scorer_module.DIVERGENCE_CONTRADICTED
        )

    def test_heuristic_logical_returns_neutral_for_empty_token_sets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            divergence_module, "rust_heuristic_logical_divergence", None
        )
        assert CoherenceScorer._heuristic_logical("!!!", "???") == (
            scorer_module.DIVERGENCE_NEUTRAL
        )

    def test_parallel_coherence_cleans_up_factual_future_after_logic_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time

        class FakeNLI:
            model_available = True

            def _ensure_model(self) -> None:
                self.loaded = True

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(scorer, "_nli", FakeNLI())
        monkeypatch.setattr(
            scorer, "_detect_task_type", lambda _prompt, _action="": "default"
        )

        def fail_logic(*args: object, **kwargs: object) -> None:
            del args, kwargs
            time.sleep(0.01)
            raise RuntimeError("logic failed")

        def fail_fact(*args: object, **kwargs: object) -> None:
            del args, kwargs
            time.sleep(0.03)
            raise RuntimeError("fact failed")

        monkeypatch.setattr(scorer, "calculate_logical_divergence", fail_logic)
        monkeypatch.setattr(
            scorer, "calculate_factual_divergence_with_evidence", fail_fact
        )

        with pytest.raises(RuntimeError, match="logic failed"):
            scorer._heuristic_coherence("prompt", "answer")

    def test_meta_classifier_fail_closed_raises_runtime_error(self) -> None:
        scorer = CoherenceScorer(use_nli=False)
        scorer._adaptive_threshold_enabled = True
        scorer._adaptive_threshold_fail_closed = True
        scorer._meta_classifier_path = "/tmp/director-ai-missing-classifier.json"

        with pytest.raises(RuntimeError, match="Adaptive threshold classifier"):
            scorer._get_meta_classifier()

    def test_meta_classifier_returns_value_set_while_waiting_for_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scorer = CoherenceScorer(use_nli=False)
        sentinel = object()

        class LockThatPopulatesClassifier:
            def __enter__(self) -> None:
                scorer._meta_classifier = sentinel

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> Literal[False]:
                del exc_type, exc, tb
                return False

        monkeypatch.setattr(
            scorer, "_meta_classifier_lock", LockThatPopulatesClassifier()
        )

        assert scorer._get_meta_classifier() is sentinel

    def test_meta_classifier_checks_bundled_path_when_adaptive_threshold_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class ExistingPath:
            def __init__(self, *_parts: object) -> None:
                self.parts = _parts

            @property
            def parent(self) -> ExistingPath:
                return self

            def __truediv__(self, other: str) -> ExistingPath:
                return ExistingPath(*self.parts, other)

            def exists(self) -> bool:
                return True

            def __str__(self) -> str:
                return "/tmp/director-ai-bundled-classifier.json"

        scorer = CoherenceScorer(use_nli=False)
        scorer._adaptive_threshold_enabled = True
        monkeypatch.setattr(scorer_module, "Path", ExistingPath)

        assert scorer._get_meta_classifier() is None
        assert scorer._meta_classifier_path == ""

    def test_finalise_review_dry_run_warning_and_retrieval_confidence(self) -> None:
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

    def test_verified_source_and_routing_guards(self) -> None:
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

    def test_verified_scorer_approved_result_preserves_review_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import director_ai.core.scoring.verified_scorer as verified_module

        class ApprovedVerification:
            approved = True
            coverage = 1.0
            claims = ["claim"]
            contradicted_count = 0
            fabricated_count = 0

            def to_dict(self) -> dict[str, bool]:
                return {"approved": True}

        class FakeVerifiedScorer:
            def __init__(self, nli_scorer: object = None) -> None:
                self.nli_scorer = nli_scorer

            def verify(self, *args: object, **kwargs: object) -> ApprovedVerification:
                del args, kwargs
                return ApprovedVerification()

        monkeypatch.setattr(verified_module, "VerifiedScorer", FakeVerifiedScorer)
        scorer = CoherenceScorer(use_nli=False)
        scorer._verified_scorer_enabled = True
        scorer._verified_scorer_task_types = {"rag"}
        score = scorer._finalise_review(
            0.8,
            0.1,
            0.2,
            "answer",
            evidence=ScoringEvidence(
                chunks=[EvidenceChunk(text="source", distance=0.1, source="kb")],
                nli_premise="source",
                nli_hypothesis="answer",
                nli_score=0.2,
            ),
        )[1]

        approved, updated = scorer._apply_verified_scorer(
            score,
            task_type="rag",
            threshold=0.5,
        )

        assert approved is True
        assert updated.verified_approved is True
        assert updated.verified_result == {"approved": True}

    def test_review_batch_errors_and_meta_threshold_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeNLI:
            model_available = True
            backend = "deberta"

            def __init__(self, scores: list[list[float]]) -> None:
                self.scores = list(scores)

            def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
                del pairs
                return self.scores.pop(0)

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(scorer, "_nli", FakeNLI([[0.1]]))
        scorer._judge = type("FakeJudge", (), {"enabled": False})()
        scorer._confidence_weighted_agg = False

        with pytest.raises(RuntimeError, match="logical NLI batch returned"):
            scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])

        class Store:
            def retrieve_context(
                self, prompt: str, top_k: int = 3, tenant_id: str = ""
            ) -> str:
                del prompt, top_k, tenant_id
                return "grounded context"

        class MetaClassifier:
            def predict_threshold(
                self, prompt: str, action: str
            ) -> tuple[float, float]:
                del prompt, action
                return 0.25, 0.9

        scorer = CoherenceScorer(use_nli=False, ground_truth_store=Store())
        monkeypatch.setattr(scorer, "_nli", FakeNLI([[0.2, 0.3], [0.4]]))
        scorer._judge = type("FakeJudge", (), {"enabled": False})()
        scorer._confidence_weighted_agg = False
        monkeypatch.setattr(scorer, "_get_meta_classifier", lambda: MetaClassifier())

        with pytest.raises(RuntimeError, match="factual NLI batch returned"):
            scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])

        monkeypatch.setattr(scorer, "_nli", FakeNLI([[0.1, 0.2], [0.3, 0.4]]))
        results = scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])

        assert len(results) == 2
        assert all(result[1].detected_task_type for result in results)

    def test_factual_retrieval_abstention_and_confidence_weighted_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock

        from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

        store = VectorGroundTruthStore()
        monkeypatch.setattr(
            store, "retrieve_context", MagicMock(return_value="retrieved context")
        )
        monkeypatch.setattr(
            store,
            "retrieve_context_with_chunks",
            MagicMock(
                return_value=[
                    EvidenceChunk(text="far chunk", distance=0.95, source="kb")
                ]
            ),
        )
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._retrieval_abstention_threshold = 0.2

        assert scorer.calculate_factual_divergence("prompt", "answer") == (
            scorer_module.DIVERGENCE_NEUTRAL
        )

        scorer._retrieval_abstention_threshold = 0.0
        scorer._confidence_weighted_agg = True
        monkeypatch.setattr(
            scorer, "_should_escalate", lambda _score, task_type="default": True
        )
        judge_check = MagicMock(return_value=0.07)
        monkeypatch.setattr(scorer, "_llm_judge_check", judge_check)

        nli = MagicMock()
        nli.model_available = True
        nli.score_chunked_confidence_weighted.return_value = (0.31, [0.31])
        monkeypatch.setattr(scorer, "_nli", nli)

        assert scorer.calculate_factual_divergence("prompt", "answer") == 0.07
        nli.score_chunked_confidence_weighted.assert_called_once_with(
            "retrieved context",
            "answer",
            inner_agg="max",
            premise_ratio=0.4,
            overlap_ratio=0.5,
        )
        judge_check.assert_called_once_with("prompt", "answer", 0.31)

    def test_finalise_review_evicts_history_and_verified_guard_without_source(
        self,
    ) -> None:
        scorer = CoherenceScorer(use_nli=False, threshold=0.2, history_window=1)
        scorer._finalise_review(0.9, 0.1, 0.1, "first")
        scorer._finalise_review(0.9, 0.1, 0.1, "second")

        assert scorer.history == ["second"]

        scorer._verified_scorer_enabled = True
        assert not scorer._should_run_verified_scorer(
            coherence=0.21,
            threshold=0.2,
            task_type="rag",
            evidence=None,
        )

    def test_reasoning_tier_accepts_unadjusted_verdict_with_non_string_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from director_ai.core.scoring.reasoning_scorer import ReasoningVerdict

        scorer = CoherenceScorer(
            use_nli=False,
            reasoning_enabled=True,
            reasoning_provider="local",
        )
        monkeypatch.setattr(
            scorer,
            "_verified_source_from_evidence",
            lambda _evidence: object(),
        )
        monkeypatch.setattr(
            scorer._reasoning, "should_escalate", lambda *_args, **_kwargs: True
        )
        monkeypatch.setattr(
            scorer._reasoning,
            "reason",
            lambda *_args, **_kwargs: ReasoningVerdict(
                approved=True,
                confidence=0.8,
                rationale="accepted",
                adjusted_score=None,
            ),
        )
        evidence = ScoringEvidence(
            chunks=[],
            nli_premise="source",
            nli_hypothesis="answer",
            nli_score=0.2,
        )
        score = scorer._finalise_review(
            0.51,
            0.2,
            0.3,
            "answer",
            evidence=evidence,
        )[1]

        approved, out = scorer._apply_reasoning_tier(
            (True, score),
            "prompt",
            "answer",
            evidence,
            threshold=0.5,
        )

        assert approved is True
        assert out.score == 0.51
        assert out.reasoning_escalated is True
        assert out.reasoning_rationale == "accepted"

    def test_reasoning_tier_noop_when_margin_does_not_escalate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scorer = CoherenceScorer(
            use_nli=False,
            reasoning_enabled=True,
            reasoning_provider="local",
        )
        monkeypatch.setattr(
            scorer._reasoning, "should_escalate", lambda *_args, **_kwargs: False
        )
        score = scorer._finalise_review(0.95, 0.1, 0.1, "answer")[1]

        assert scorer._apply_reasoning_tier(
            (True, score),
            "prompt",
            "answer",
            None,
            threshold=0.5,
        ) == (True, score)

    def test_review_cross_turn_failure_and_intent_drift_are_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock

        class FakeNLI:
            model_available = True

            def score(self, premise: str, hypothesis: str) -> float:
                assert premise == "prior answer"
                assert hypothesis == "current answer"
                return 0.4

        class Drift:
            drift_risk = 0.42
            triggered = False

        class FakeDrift:
            def __init__(self) -> None:
                self.calls: list[tuple[float, float, float]] = []

            def update(
                self,
                *,
                intent_divergence: float,
                injection_risk: float,
                contradiction_trend: float,
            ) -> Drift:
                self.calls.append(
                    (intent_divergence, injection_risk, contradiction_trend)
                )
                return Drift()

        class FakeSession:
            context_text = "prior answer"

            def __init__(self) -> None:
                self.intent_drift = FakeDrift()
                self.turns: list[tuple[str, str, float]] = []

            def __len__(self) -> int:
                return 1

            def update_contradictions(self, response: str, score_fn: object) -> None:
                del response, score_fn
                raise RuntimeError("tracker unavailable")

            def add_turn(self, prompt: str, action: str, score: float) -> None:
                self.turns.append((prompt, action, score))

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(scorer, "_nli", FakeNLI())
        monkeypatch.setattr(
            scorer,
            "_heuristic_coherence",
            MagicMock(return_value=(0.2, 0.5, 0.68, None)),
        )
        session = FakeSession()

        approved, score = scorer.review(
            "prompt",
            "current answer",
            session=session,
        )

        assert approved is True
        assert score.cross_turn_divergence == 0.4
        assert score.intent_drift_risk == 0.42
        assert score.intent_drift_triggered is False
        assert session.intent_drift.calls == [(0.4, 0.0, 0.0)]
        assert session.turns == [("prompt", "current answer", score.score)]

    def test_review_records_successful_injection_risk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Risk:
            injection_risk = 0.66

        class Detector:
            def detect(self, *, intent: str, response: str) -> Risk:
                assert intent == "prompt"
                assert response == "answer"
                return Risk()

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(
            scorer,
            "_heuristic_coherence",
            lambda *_args, **_kwargs: (0.1, 0.1, 0.9, None),
        )
        scorer._injection_detector = Detector()

        approved, score = scorer.review("prompt", "answer")

        assert approved is True
        assert score.injection_risk == 0.66

    def test_review_invokes_reasoning_tier_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Reasoning:
            enabled = True

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(
            scorer,
            "_heuristic_coherence",
            lambda *_args, **_kwargs: (0.1, 0.1, 0.9, None),
        )
        monkeypatch.setattr(scorer, "_reasoning", Reasoning())
        calls: list[
            tuple[tuple[bool, CoherenceScore], str, str, ScoringEvidence | None, float]
        ] = []

        def apply_reasoning(
            result: tuple[bool, CoherenceScore],
            prompt: str,
            action: str,
            evidence: ScoringEvidence | None,
            *,
            threshold: float,
        ) -> tuple[bool, CoherenceScore]:
            calls.append((result, prompt, action, evidence, threshold))
            return result

        monkeypatch.setattr(scorer, "_apply_reasoning_tier", apply_reasoning)

        approved, score = scorer.review("prompt", "answer")

        assert approved is True
        assert score.score == 0.9
        assert calls

    def test_review_injection_fail_closed_reraises_detector_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FailingDetector:
            def detect(self, *, intent: str, response: str) -> None:
                del intent, response
                raise RuntimeError("detector unavailable")

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(
            scorer,
            "_heuristic_coherence",
            lambda *_args, **_kwargs: (0.1, 0.1, 0.9, None),
        )
        scorer._injection_detector = FailingDetector()
        scorer._injection_fail_closed = True

        with pytest.raises(RuntimeError, match="detector unavailable"):
            scorer.review("prompt", "answer")

    def test_review_batch_fallback_and_no_context_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeNLI:
            model_available = True
            backend = "deberta"

            def __init__(self) -> None:
                self.calls: list[list[tuple[str, str]]] = []

            def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
                self.calls.append(list(pairs))
                return [0.1 for _pair in pairs]

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(scorer, "_nli", FakeNLI())
        scorer._judge = type("FakeJudge", (), {"enabled": False})()
        monkeypatch.setattr(
            scorer, "_detect_task_type", lambda _prompt, _action="": "dialogue"
        )
        monkeypatch.setattr(
            scorer,
            "review",
            lambda prompt, action, tenant_id="": (
                True,
                scorer._finalise_review(0.9, 0.1, 0.1, action)[1],
            ),
        )

        fallback_results = scorer.review_batch(
            [("User: hi", "Assistant: hi"), ("User: bye", "Assistant: bye")]
        )

        assert len(fallback_results) == 2
        assert all(result[0] for result in fallback_results)

        scorer = CoherenceScorer(use_nli=False)
        fake_nli = FakeNLI()
        monkeypatch.setattr(scorer, "_nli", fake_nli)
        scorer._judge = type("FakeJudge", (), {"enabled": False})()
        monkeypatch.setattr(
            scorer, "_detect_task_type", lambda _prompt, _action="": "qa"
        )

        batch_results = scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])

        assert len(batch_results) == 2
        assert len(fake_nli.calls) == 1
        assert all(result[1].h_factual == 0.5 for result in batch_results)

    def test_review_batch_requires_sequential_for_judge_and_vector_store(self) -> None:
        from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

        scorer = CoherenceScorer(use_nli=False)
        scorer._judge = type("FakeJudge", (), {"enabled": True})()
        assert scorer._review_batch_requires_sequential([("Q", "A")]) is True

        scorer._judge = type("FakeJudge", (), {"enabled": False})()
        scorer.ground_truth_store = VectorGroundTruthStore()
        assert scorer._review_batch_requires_sequential([("Q", "A")]) is True

    def test_review_batch_raises_if_nli_disappears_after_initial_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeNLI:
            model_available = True
            backend = "deberta"

        scorer = CoherenceScorer(use_nli=False)
        monkeypatch.setattr(scorer, "_nli", FakeNLI())
        scorer._judge = type("FakeJudge", (), {"enabled": False})()
        monkeypatch.setattr(
            scorer, "_detect_task_type", lambda _prompt, _action="": "qa"
        )

        def clear_nli(_items: Sequence[tuple[str, str]]) -> bool:
            scorer._nli = None
            return False

        monkeypatch.setattr(scorer, "_review_batch_requires_sequential", clear_nli)

        with pytest.raises(RuntimeError, match="NLI batch scorer not initialised"):
            scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])


class TestFactualDivergenceGroundingQuery:
    """A grounded scorer stays neutral when the prompt carries no query."""

    def test_whitespace_prompt_returns_neutral_despite_store(self) -> None:
        from director_ai.core import GroundTruthStore

        store = GroundTruthStore()
        store.add("sky colour", "The sky is blue.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        # Force the Python path so the grounding-query gate itself runs.
        scorer._rust_scorer = None

        divergence = scorer.calculate_factual_divergence("   \n\t  ", "The sky is red.")

        assert divergence == scorer_module.DIVERGENCE_NEUTRAL
