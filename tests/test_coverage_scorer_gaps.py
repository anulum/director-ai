# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.scoring.scorer import DIVERGENCE_CONTRADICTED, DIVERGENCE_NEUTRAL

# ── Line 167: cache=... path in __init__ ─────────────────────────────────────


class TestInitCachePath:
    def test_cache_object_passed_directly(self):
        from director_ai.core.cache import ScoreCache

        explicit_cache = ScoreCache(max_size=50, ttl_seconds=60)
        scorer = CoherenceScorer(use_nli=False, cache=explicit_cache)
        assert scorer.cache is explicit_cache

    def test_cache_size_creates_cache(self):
        scorer = CoherenceScorer(use_nli=False, cache_size=100, cache_ttl=120.0)
        assert scorer.cache is not None


# ── Line 283: local judge provider init branch ────────────────────────────────


class TestLocalJudgeInit:
    def test_local_judge_provider_no_model_skips(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="local",
            llm_judge_model="",
        )
        assert scorer._local_judge_model is None

    def test_local_judge_provider_with_model_tries_init(self):
        with patch.object(CoherenceScorer, "_init_local_judge") as mock_init:
            CoherenceScorer(
                use_nli=False,
                llm_judge_enabled=True,
                llm_judge_provider="local",
                llm_judge_model="some/model",
            )
            mock_init.assert_called_once_with("some/model", None)


# ── Lines 285-287: close() and __del__() ─────────────────────────────────────


class TestScorerClose:
    def test_close_shuts_down_pool(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer.close()

    def test_del_runs_without_error(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer.__del__()


# ── Line 332: _local_judge_check falls back when model is None ───────────────


class TestLocalJudgeCheck:
    def test_fallback_when_model_none(self):
        scorer = CoherenceScorer(use_nli=False)
        result = scorer._local_judge_check("prompt", "response", 0.6)
        assert result == 0.6

    def test_fallback_when_tokenizer_none(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._local_judge_model = MagicMock()
        scorer._local_judge_tokenizer = None
        result = scorer._local_judge_check("p", "r", 0.4)
        assert result == 0.4


# ── Line 406, 412, 416-424: _get_meta_classifier branches ───────────────────


class TestGetMetaClassifier:
    def test_returns_none_when_no_path(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._get_meta_classifier() is None

    def test_returns_cached_classifier(self):
        scorer = CoherenceScorer(use_nli=False)
        mock_clf = MagicMock()
        scorer._meta_classifier = mock_clf
        assert scorer._get_meta_classifier() is mock_clf

    def test_adaptive_bundled_path_does_not_exist(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._adaptive_threshold_enabled = True
        scorer._meta_classifier_path = ""
        with patch(
            "director_ai.core.scoring.scorer.Path.exists",
            return_value=False,
        ):
            result = scorer._get_meta_classifier()
            assert result is None

    def test_meta_classifier_path_import_error(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._meta_classifier_path = "/fake/path/classifier.pkl"
        with patch(
            "director_ai.core.scoring.scorer.Path.exists",
            return_value=False,
        ):
            result = scorer._get_meta_classifier()
            assert result is None

    def test_meta_classifier_load_fails_gracefully(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._meta_classifier_path = "/some/existing/path.pkl"
        with patch(
            "director_ai.core.scoring.meta_classifier.DatasetTypeClassifier",
            side_effect=Exception("load error"),
        ):
            result = scorer._get_meta_classifier()
            assert result is None


# ── Lines 669-699: _dialogue_factual_divergence NLI-required path ─────────────


class TestDialogueFactualDivergence:
    def test_raises_without_nli(self):
        scorer = CoherenceScorer(use_nli=False)
        with pytest.raises(RuntimeError, match="NLI model required"):
            scorer._dialogue_factual_divergence("Human: hi\nAssistant: hello", "hi")

    def test_with_mocked_nli(self):
        scorer = CoherenceScorer(use_nli=False)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_chunked.return_value = (0.3, [])
        scorer._nli = mock_nli

        with patch.object(
            scorer,
            "calculate_factual_divergence_with_evidence",
            return_value=(0.4, None),
        ):
            result, evidence = scorer._dialogue_factual_divergence(
                "Human: hi\nAssistant: hello", "hello"
            )
            assert 0.0 <= result <= 1.0

    def test_baseline_calibration_zero_denom(self):
        scorer = CoherenceScorer(use_nli=False)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_chunked.return_value = (0.5, [])
        scorer._nli = mock_nli
        scorer._dialogue_nli_baseline = 1.0

        with patch.object(
            scorer,
            "calculate_factual_divergence_with_evidence",
            return_value=(0.5, None),
        ):
            result, _ = scorer._dialogue_factual_divergence("p", "r")
            assert 0.0 <= result <= 1.0


# ── Line 723, 750: _summarization_factual_divergence ─────────────────────────


class TestSummarizationFactualDivergence:
    def test_raises_without_nli(self):
        scorer = CoherenceScorer(use_nli=False)
        with pytest.raises(RuntimeError, match="NLI model required"):
            scorer._summarization_factual_divergence("Summarize: some text", "Summary.")

    def test_zero_baseline_path(self):
        scorer = CoherenceScorer(use_nli=False)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_chunked.return_value = (0.3, [])
        mock_nli.score_claim_coverage_with_attribution.return_value = (
            0.8,
            [0.2, 0.3],
            ["claim1", "claim2"],
            [],
        )
        scorer._nli = mock_nli
        scorer._summarization_nli_baseline = 0.0

        mock_evidence = MagicMock()
        with patch.object(
            scorer,
            "calculate_factual_divergence_with_evidence",
            return_value=(0.4, mock_evidence),
        ):
            result, ev = scorer._summarization_factual_divergence("doc", "summary")
            assert 0.0 <= result <= 1.0

    def test_claim_coverage_disabled(self):
        scorer = CoherenceScorer(use_nli=False)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_chunked.return_value = (0.3, [])
        scorer._nli = mock_nli
        scorer._claim_coverage_enabled = False

        with patch.object(
            scorer,
            "calculate_factual_divergence_with_evidence",
            return_value=(0.4, None),
        ):
            result, ev = scorer._summarization_factual_divergence("doc", "summary")
            assert 0.0 <= result <= 1.0


# ── Lines 764-772: summarization evidence attribution assignment ──────────────


class TestSummarizationEvidenceAttribution:
    def test_evidence_attribution_populated(self):
        scorer = CoherenceScorer(use_nli=False)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_chunked.return_value = (0.3, [])
        mock_nli.score_claim_coverage_with_attribution.return_value = (
            0.7,
            [0.1, 0.2],
            ["c1", "c2"],
            ["attr1"],
        )
        scorer._nli = mock_nli

        from director_ai.core.types import EvidenceChunk, ScoringEvidence

        real_evidence = ScoringEvidence(
            chunks=[EvidenceChunk(text="ctx", distance=0.0, source="test")],
            nli_premise="ctx",
            nli_hypothesis="summary",
            nli_score=0.4,
        )

        with patch.object(
            scorer,
            "calculate_factual_divergence_with_evidence",
            return_value=(0.4, real_evidence),
        ):
            result, ev = scorer._summarization_factual_divergence("doc", "summary")
            assert ev is not None
            assert ev.claim_coverage == 0.7
            assert ev.claims == ["c1", "c2"]


# ── Lines 807-827: calculate_factual_divergence use_prompt_as_premise ─────────


class TestFactualDivergencePromptAsPremise:
    def test_prompt_as_premise_no_nli_neutral(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._use_prompt_as_premise = True
        result = scorer.calculate_factual_divergence("prompt text", "output text")
        assert result == DIVERGENCE_NEUTRAL

    def test_prompt_as_premise_with_mocked_nli(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._use_prompt_as_premise = True
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_chunked.return_value = (0.3, [])
        scorer._nli = mock_nli
        result = scorer.calculate_factual_divergence("prompt", "output")
        assert 0.0 <= result <= 1.0

    def test_qa_premise_ratio_applied(self):
        store = GroundTruthStore()
        store.add("question", "The answer is 42.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._qa_premise_ratio = 0.9
        result = scorer.calculate_factual_divergence(
            "What is the answer to the question?", "The answer is 42."
        )
        assert 0.0 <= result <= 1.0


# ── Lines 844-855: retrieval abstention threshold branch ─────────────────────


class TestRetrievalAbstention:
    def test_abstention_threshold_neutral(self):
        from director_ai.core.vector_store import VectorGroundTruthStore

        store = VectorGroundTruthStore()
        store.add_fact("sky", "The sky is blue.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._retrieval_abstention_threshold = 0.9999

        result = scorer.calculate_factual_divergence(
            "completely unrelated query xyz abc", "some output"
        )
        assert 0.0 <= result <= 1.0


# ── Lines 860: confidence_weighted_agg in factual divergence ─────────────────


class TestConfidenceWeightedAgg:
    def test_confidence_weighted_agg_no_nli(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._confidence_weighted_agg = True
        result = scorer.calculate_factual_divergence("sky", "The sky is blue.")
        assert 0.0 <= result <= 1.0


# ── Lines 884-899: RAG claim decomposition in factual divergence ─────────────


class TestRagClaimDecomposition:
    def test_rag_claim_decomposition_with_mocked_nli(self):
        store = GroundTruthStore()
        store.add("context", "The sky is blue. Water is wet. Fire is hot.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_chunked.return_value = (0.3, [])
        mock_nli._split_sentences.return_value = [
            "The sky is blue.",
            "Water is wet.",
            "Fire is hot.",
        ]
        scorer._nli = mock_nli
        scorer._rag_claim_decomposition = True
        result = scorer.calculate_factual_divergence(
            "context",
            "The sky is blue. Water is wet. Fire is hot.",
        )
        assert 0.0 <= result <= 1.0

    def test_rag_claim_decomposition_disabled(self):
        store = GroundTruthStore()
        store.add("sky", "Blue sky.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._rag_claim_decomposition = False
        result = scorer.calculate_factual_divergence(
            "sky",
            "Blue sky.",
        )
        assert 0.0 <= result <= 1.0


# ── Lines 900-903: strict_mode / heuristic in factual with context ────────────


class TestFactualWithContextHeuristic:
    def test_heuristic_when_no_nli_no_strict(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(
            use_nli=False, ground_truth_store=store, strict_mode=False
        )
        result = scorer.calculate_factual_divergence("sky", "The sky is blue.")
        assert 0.0 <= result <= 1.0

    def test_strict_mode_returns_contradicted(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(
            use_nli=False, ground_truth_store=store, strict_mode=True
        )
        result = scorer.calculate_factual_divergence("sky", "The sky is blue.")
        assert result == DIVERGENCE_CONTRADICTED


# ── Lines 932-943: calculate_factual_divergence_with_evidence confidence-weighted ─


class TestFactualEvidenceConfidenceWeighted:
    def test_confidence_weighted_in_evidence_path(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._confidence_weighted_agg = True
        div, ev = scorer.calculate_factual_divergence_with_evidence(
            "sky", "The sky is blue."
        )
        assert 0.0 <= div <= 1.0


# ── Line 956: evidence path with use_prompt_as_premise + escalation ─────────


class TestFactualEvidencePromptAsPremise:
    def test_use_prompt_as_premise_evidence_path(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._use_prompt_as_premise = True
        div, ev = scorer.calculate_factual_divergence_with_evidence(
            "some prompt", "some output"
        )
        assert div == DIVERGENCE_NEUTRAL
        assert ev is None

    def test_use_prompt_as_premise_with_nli(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._use_prompt_as_premise = True
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.last_token_count = 10
        mock_nli.last_estimated_cost = 0.0
        mock_nli._score_chunked_with_counts.return_value = (0.3, [0.3], 1, 1)
        scorer._nli = mock_nli
        div, ev = scorer.calculate_factual_divergence_with_evidence(
            "some prompt", "some output"
        )
        assert 0.0 <= div <= 1.0
        assert ev is not None


# ── Lines 1010-1020: evidence path confidence_weighted with store ─────────────


class TestFactualEvidenceConfidenceWeightedWithStore:
    def test_confidence_weighted_agg_with_store_and_nli(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._confidence_weighted_agg = True
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.last_token_count = 5
        mock_nli._cost_per_token = 0.00001
        mock_nli.score_chunked_confidence_weighted.return_value = (0.35, [0.3, 0.4])
        scorer._nli = mock_nli
        div, ev = scorer.calculate_factual_divergence_with_evidence(
            "sky", "The sky is blue."
        )
        assert 0.0 <= div <= 1.0


# ── Line 1039: evidence path escalation via should_escalate ──────────────────


class TestFactualEvidenceEscalation:
    def test_escalation_triggered_in_evidence_path(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(
            use_nli=False,
            ground_truth_store=store,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
        )
        scorer._llm_judge_threshold = 0.9

        import sys
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        choice = MagicMock()
        choice.message.content = '{"verdict": "YES", "confidence": 80}'
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])

        with _patch.dict(sys.modules, {"openai": mock_openai}):
            div, ev = scorer.calculate_factual_divergence_with_evidence(
                "sky", "The sky is blue."
            )
            assert 0.0 <= div <= 1.0


# ── Lines 1054-1057: claim decomposition in evidence path ────────────────────


class TestFactualEvidenceClaimDecomposition:
    def test_claim_decomposition_in_evidence(self):
        store = GroundTruthStore()
        store.add("context", "Fact one. Fact two. Fact three.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.last_token_count = 10
        mock_nli._cost_per_token = 0.0
        mock_nli._score_chunked_with_counts.return_value = (0.2, [0.2], 2, 2)
        mock_nli.score_claim_coverage_with_attribution.return_value = (
            0.8,
            [0.1, 0.2, 0.3],
            ["c1", "c2", "c3"],
            [],
        )
        scorer._nli = mock_nli
        scorer._rag_claim_decomposition = True

        long_output = "Fact one is verified. Fact two is true. Fact three is confirmed."
        div, ev = scorer.calculate_factual_divergence_with_evidence(
            "context", long_output
        )
        assert 0.0 <= div <= 1.0


# ── Lines 1222-1223: heuristic_coherence dialogue path (no NLI) ──────────────


class TestHeuristicCoherenceDialoguePath:
    def test_dialogue_path_skipped_without_nli(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._auto_dialogue_profile = True
        dialogue_prompt = "Human: Hello\nAssistant: Hi\nHuman: How are you?"
        approved, cs = scorer.review(dialogue_prompt, "I'm fine, thank you.")
        assert isinstance(approved, bool)
        assert 0.0 <= cs.score <= 1.0


# ── Lines 1237-1238: heuristic_coherence summarization path with W_LOGIC=0 ───


class TestHeuristicCoherenceSummarizationPath:
    def test_summarization_path_wlogic_zero_no_nli(self):
        scorer = CoherenceScorer(use_nli=False, w_logic=0.0, w_fact=1.0)
        scorer._use_prompt_as_premise = True
        approved, cs = scorer.review("Summarize: text here", "Summary here.")
        assert isinstance(approved, bool)

    def test_wlogic_zero_regular_factual_path(self):
        scorer = CoherenceScorer(use_nli=False, w_logic=0.0, w_fact=1.0)
        approved, cs = scorer.review("q", "a")
        assert isinstance(approved, bool)


# ── Lines 1422-1428: meta-classifier active in review() ──────────────────────


class TestMetaClassifierInReview:
    def test_meta_classifier_adjusts_threshold(self):
        scorer = CoherenceScorer(use_nli=False)
        mock_clf = MagicMock()
        mock_clf.predict_threshold.return_value = (0.7, 0.9)
        scorer._meta_classifier = mock_clf
        scorer._meta_classifier_path = "/fake"

        approved, cs = scorer.review("sky question?", "The sky is blue.")
        assert isinstance(approved, bool)
        mock_clf.predict_threshold.assert_called_once()

    def test_meta_classifier_none_threshold_falls_back(self):
        scorer = CoherenceScorer(use_nli=False)
        mock_clf = MagicMock()
        mock_clf.predict_threshold.return_value = (None, 0.5)
        scorer._meta_classifier = mock_clf
        scorer._meta_classifier_path = "/fake"

        approved, cs = scorer.review("q", "a")
        assert isinstance(approved, bool)


# ── Lines 1410-1415: adaptive threshold in review() ──────────────────────────


class TestAdaptiveThreshold:
    def test_adaptive_threshold_enabled_dialogue(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._adaptive_threshold_enabled = True
        scorer._task_type_thresholds = {"dialogue": 0.3, "default": 0.5}
        dialogue = "Human: hi\nAssistant: hello\nHuman: bye"
        approved, cs = scorer.review(dialogue, "Goodbye!")
        assert isinstance(approved, bool)

    def test_adaptive_threshold_enabled_qa(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._adaptive_threshold_enabled = True
        scorer._task_type_thresholds = {"qa": 0.4}
        approved, cs = scorer.review("What is the sky color?", "Blue.")
        assert isinstance(approved, bool)


# ── _should_escalate task-type thresholds ─────────────────────────────────────


class TestShouldEscalate:
    def test_no_judge_enabled_returns_false(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._should_escalate(0.5) is False

    def test_judge_enabled_but_local_model_none(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="local",
        )
        assert scorer._should_escalate(0.5) is False

    def test_judge_enabled_openai_borderline(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
        )
        scorer._llm_judge_threshold = 0.4
        assert scorer._should_escalate(0.5) is True

    def test_task_type_threshold_dialogue(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
        )
        # dialogue threshold = 0.35, so abs(0.5 - 0.5) = 0.0 < 0.35 → True
        assert scorer._should_escalate(0.5, task_type="dialogue") is True

    def test_task_type_threshold_fact_check(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
        )
        # fact_check threshold = 0.20, abs(0.5 - 0.5) = 0.0 < 0.20 → True
        assert scorer._should_escalate(0.5, task_type="fact_check") is True
        # abs(0.1 - 0.5) = 0.4 > 0.20 → False (confident score, no escalation)
        assert scorer._should_escalate(0.1, task_type="fact_check") is False

    def test_task_type_threshold_differentiates(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
        )
        # Score 0.25: abs(0.25 - 0.5) = 0.25
        # dialogue threshold 0.35 → 0.25 < 0.35 → escalate
        # fact_check threshold 0.20 → 0.25 > 0.20 → do NOT escalate
        assert scorer._should_escalate(0.25, task_type="dialogue") is True
        assert scorer._should_escalate(0.25, task_type="fact_check") is False


# ── _detect_task_type branches ────────────────────────────────────────────────


class TestDetectTaskType:
    def test_dialogue(self):
        prompt = "Human: hi\nAssistant: hello\nHuman: how are you?"
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_summarization(self):
        assert (
            CoherenceScorer._detect_task_type("please summarize this")
            == "summarization"
        )

    def test_rag(self):
        assert (
            CoherenceScorer._detect_task_type("based on the context provided") == "rag"
        )

    def test_fact_check(self):
        assert CoherenceScorer._detect_task_type("verify this claim") == "fact_check"

    def test_qa(self):
        assert CoherenceScorer._detect_task_type("What is the answer?") == "qa"

    def test_default(self):
        assert CoherenceScorer._detect_task_type("just some text") == "default"


# ── review_batch ──────────────────────────────────────────────────────────────


class TestReviewBatch:
    def test_empty_batch(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer.review_batch([]) == []

    def test_non_empty_batch(self):
        scorer = CoherenceScorer(use_nli=False)
        results = scorer.review_batch([("q1", "a1"), ("q2", "a2")])
        assert len(results) == 2
        for approved, cs in results:
            assert isinstance(approved, bool)
            assert 0.0 <= cs.score <= 1.0


# ── _resolve_agg_profile dialogue override ───────────────────────────────────


class TestResolveAggProfile:
    def test_dialogue_returns_min_mean(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._auto_dialogue_profile = True
        scorer._use_prompt_as_premise = False
        prompt = "Human: hi\nAssistant: hello\nHuman: bye"
        fi, fo, li, lo = scorer._resolve_agg_profile(prompt)
        assert fi == "min"
        assert fo == "mean"

    def test_non_dialogue_returns_defaults(self):
        scorer = CoherenceScorer(use_nli=False)
        fi, fo, li, lo = scorer._resolve_agg_profile("plain question")
        assert fi == scorer._fact_inner_agg
