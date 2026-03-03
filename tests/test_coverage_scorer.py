"""Coverage tests for scorer.py — CoherenceScorer."""

from __future__ import annotations

import asyncio
import warnings

import pytest

from director_ai.core import CoherenceScorer, GroundTruthStore


class TestScorerInit:
    def test_bad_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            CoherenceScorer(threshold=1.5)

    def test_bad_soft_limit(self):
        with pytest.raises(ValueError, match="soft_limit"):
            CoherenceScorer(threshold=0.5, soft_limit=1.5)

    def test_soft_limit_below_threshold(self):
        with pytest.raises(ValueError, match="soft_limit.*threshold"):
            CoherenceScorer(threshold=0.7, soft_limit=0.5)

    def test_bad_w_logic(self):
        with pytest.raises(ValueError, match="w_logic"):
            CoherenceScorer(w_logic=-0.1, w_fact=1.1)

    def test_bad_w_fact(self):
        with pytest.raises(ValueError, match="w_fact"):
            CoherenceScorer(w_logic=0.5, w_fact=1.5)

    def test_weights_not_summing_to_1(self):
        with pytest.raises(ValueError, match="w_logic.*w_fact.*1.0"):
            CoherenceScorer(w_logic=0.3, w_fact=0.3)

    def test_hybrid_needs_provider(self):
        with pytest.raises(ValueError, match="hybrid"):
            CoherenceScorer(scorer_backend="hybrid")


class TestScorerReview:
    def test_review_approved(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        approved, cs = scorer.review("sky?", "The sky is blue.")
        assert isinstance(approved, bool)
        assert 0.0 <= cs.score <= 1.0

    def test_review_rejected_low_coherence(self):
        scorer = CoherenceScorer(threshold=0.99, use_nli=False)
        approved, cs = scorer.review("abc", "completely unrelated xyz stuff")
        assert not approved

    def test_review_warning(self):
        scorer = CoherenceScorer(
            threshold=0.2,
            soft_limit=0.9,
            use_nli=False,
        )
        approved, cs = scorer.review("sky?", "The sky is blue.")
        assert approved
        assert cs.warning

    def test_strict_mode_rejected(self):
        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            strict_mode=True,
        )
        store = GroundTruthStore()
        store.add("sky", "blue")
        scorer.ground_truth_store = store
        approved, cs = scorer.review("sky", "something")
        assert cs.strict_mode_rejected

    def test_review_with_cache(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False, cache_size=100)
        scorer.review("sky?", "blue.")
        approved2, cs2 = scorer.review("sky?", "blue.")
        assert 0.0 <= cs2.score <= 1.0

    def test_history_update(self):
        scorer = CoherenceScorer(threshold=0.1, use_nli=False, history_window=3)
        scorer.review("q", "The sky is blue and consistent with reality.")
        assert len(scorer.history) >= 1


class TestScorerFactual:
    def test_no_store(self):
        scorer = CoherenceScorer(use_nli=False)
        div = scorer.calculate_factual_divergence("q", "a")
        assert div == 0.5

    def test_store_no_match(self):
        store = GroundTruthStore()
        store.add("capital", "Paris is the capital of France.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        div = scorer.calculate_factual_divergence("zzz", "yyy")
        assert div == 0.5

    def test_store_with_match(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        div = scorer.calculate_factual_divergence("sky", "The sky is blue.")
        assert 0.0 <= div <= 1.0


class TestScorerFactualEvidence:
    def test_no_store_evidence(self):
        scorer = CoherenceScorer(use_nli=False)
        div, ev = scorer.calculate_factual_divergence_with_evidence("q", "a")
        assert div == 0.5
        assert ev is None

    def test_store_evidence(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        div, ev = scorer.calculate_factual_divergence_with_evidence(
            "sky", "The sky is blue."
        )
        assert ev is not None
        assert ev.nli_premise is not None

    def test_strict_mode_evidence(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(
            use_nli=False,
            strict_mode=True,
            ground_truth_store=store,
        )
        div, ev = scorer.calculate_factual_divergence_with_evidence("sky", "a")
        assert div == 0.9


class TestScorerLogical:
    def test_heuristic_aligned(self):
        result = CoherenceScorer._heuristic_logical("consistent with reality", "q")
        assert result < 0.3

    def test_heuristic_contradicted(self):
        result = CoherenceScorer._heuristic_logical("the opposite is true", "q")
        assert result > 0.8

    def test_heuristic_neutral(self):
        result = CoherenceScorer._heuristic_logical("depends on your perspective", "q")
        assert abs(result - 0.5) < 0.01

    def test_heuristic_no_prompt(self):
        result = CoherenceScorer._heuristic_logical("some text", "")
        assert result == 0.5

    def test_calculate_logical_strict(self):
        scorer = CoherenceScorer(use_nli=False, strict_mode=True)
        div = scorer.calculate_logical_divergence("q", "a")
        assert div == 0.9


class TestScorerHeuristicFactual:
    def test_negation_asymmetry(self):
        div = CoherenceScorer._heuristic_factual(
            "The sky is blue.", "The sky is not blue."
        )
        assert div > 0.2

    def test_novel_entities(self):
        div = CoherenceScorer._heuristic_factual(
            "The sky is blue.", "Planet Mars is red."
        )
        assert div > 0.3

    def test_empty_inputs(self):
        div = CoherenceScorer._heuristic_factual("", "something")
        assert div == 0.5


class TestScorerParseJudgeReply:
    def test_json_yes(self):
        assert CoherenceScorer._parse_judge_reply(
            '{"verdict": "YES", "confidence": 90}'
        )

    def test_json_no(self):
        assert not CoherenceScorer._parse_judge_reply(
            '{"verdict": "NO", "confidence": 80}'
        )

    def test_plain_yes(self):
        assert CoherenceScorer._parse_judge_reply("I think YES this is correct")

    def test_plain_no(self):
        assert not CoherenceScorer._parse_judge_reply("I think this is incorrect")

    def test_bad_json(self):
        assert CoherenceScorer._parse_judge_reply("YES definitely") is True


class TestScorerAsync:
    def test_areview(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)

        async def run():
            return await scorer.areview("sky?", "The sky is blue.")

        approved, cs = asyncio.get_event_loop().run_until_complete(run())
        assert isinstance(approved, bool)


class TestScorerDeprecated:
    def test_calculate_factual_entropy(self):
        scorer = CoherenceScorer(use_nli=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = scorer.calculate_factual_entropy("q", "a")
        assert 0.0 <= result <= 1.0

    def test_calculate_logical_entropy(self):
        scorer = CoherenceScorer(use_nli=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = scorer.calculate_logical_entropy("q", "a")
        assert 0.0 <= result <= 1.0

    def test_simulate_future_state(self):
        scorer = CoherenceScorer(use_nli=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = scorer.simulate_future_state("q", "a")
        assert 0.0 <= result <= 1.0

    def test_review_action(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            approved, score = scorer.review_action("q", "consistent with reality")
        assert isinstance(approved, bool)
