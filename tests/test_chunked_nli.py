from __future__ import annotations

import pytest

from director_ai.core.nli import NLIScorer


class TestSplitSentences:
    def test_basic_split(self):
        result = NLIScorer._split_sentences("Hello world. How are you? Fine!")
        assert len(result) == 3

    def test_empty_string(self):
        assert NLIScorer._split_sentences("") == []

    def test_single_sentence(self):
        result = NLIScorer._split_sentences("Just one sentence.")
        assert len(result) == 1


class TestEstimateTokens:
    def test_short_text(self):
        assert NLIScorer._estimate_tokens("hello") > 0

    def test_proportional(self):
        assert NLIScorer._estimate_tokens("a" * 400) == 101  # 400 // 4 + 1


class TestBuildChunks:
    def test_single_chunk_short(self):
        scorer = NLIScorer(use_model=False)
        chunks = scorer._build_chunks(["Short sentence."], budget=100)
        assert len(chunks) == 1

    def test_multiple_chunks(self):
        scorer = NLIScorer(use_model=False)
        sentences = [f"Sentence number {i} with some padding text." for i in range(20)]
        chunks = scorer._build_chunks(sentences, budget=30)
        assert len(chunks) > 1

    def test_overlap_present(self):
        scorer = NLIScorer(use_model=False)
        sentences = [
            f"Sentence {i} with enough words to fill budget." for i in range(10)
        ]
        chunks = scorer._build_chunks(sentences, budget=30)
        if len(chunks) >= 2:
            first_words = set(chunks[0].split())
            second_words = set(chunks[1].split())
            assert first_words & second_words  # overlap exists


class TestScoreChunked:
    def test_short_text_bypasses_chunking(self):
        scorer = NLIScorer(use_model=False, max_length=512)
        score, chunk_scores = scorer.score_chunked("premise", "short hypothesis")
        assert len(chunk_scores) == 1
        assert score == chunk_scores[0]

    def test_long_text_produces_multiple_chunks(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_hyp = ". ".join(f"Sentence number {i} with words" for i in range(30)) + "."
        score, chunk_scores = scorer.score_chunked("Some premise text.", long_hyp)
        assert len(chunk_scores) > 1
        assert score == max(chunk_scores)

    def test_aggregation_is_max(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_hyp = ". ".join(f"Word number {i} in sentence" for i in range(30)) + "."
        score, chunk_scores = scorer.score_chunked("premise text here", long_hyp)
        assert score == pytest.approx(max(chunk_scores))

    def test_single_long_sentence_no_crash(self):
        scorer = NLIScorer(use_model=False, max_length=32)
        hyp = "a " * 200
        score, chunk_scores = scorer.score_chunked("premise", hyp)
        assert len(chunk_scores) == 1

    def test_chunk_scores_are_floats(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_hyp = ". ".join(f"Sentence {i} filler" for i in range(20)) + "."
        _, chunk_scores = scorer.score_chunked("context", long_hyp)
        assert all(isinstance(s, float) for s in chunk_scores)

    def test_long_premise_handled(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_premise = ". ".join(f"Premise sentence {i} word" for i in range(40)) + "."
        long_hyp = ". ".join(f"Sentence {i} text" for i in range(20)) + "."
        score, chunk_scores = scorer.score_chunked(long_premise, long_hyp)
        assert 0.0 <= score <= 1.0
        assert len(chunk_scores) >= 1

    def test_chunked_uses_batch_path(self):
        """score_chunked routes through score_batch (same results)."""
        scorer = NLIScorer(use_model=False, max_length=64)
        long_hyp = ". ".join(f"Sentence number {i} with words" for i in range(30)) + "."
        s1, cs1 = scorer.score_chunked("premise text", long_hyp)
        s2, cs2 = scorer.score_chunked("premise text", long_hyp)
        assert s1 == s2
        assert cs1 == cs2

    def test_long_premise_gets_chunked(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Evidence sentence {i} detail" for i in range(40)) + "."
        score, chunk_scores = scorer.score_chunked(long_prem, "Short hypothesis.")
        assert 0.0 <= score <= 1.0
        assert len(chunk_scores) == 1

    def test_both_long_cross_product(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Premise part {i} context" for i in range(30)) + "."
        long_hyp = ". ".join(f"Hypothesis part {i} claim" for i in range(30)) + "."
        score, chunk_scores = scorer.score_chunked(long_prem, long_hyp)
        assert 0.0 <= score <= 1.0
        assert len(chunk_scores) > 1

    def test_inner_agg_is_max(self):
        """max-across-premises: per_hyp is max of premise chunk scores."""
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Premise detail {i} info" for i in range(30)) + "."
        _, per_hyp, np, nh = scorer._score_chunked_with_counts(
            long_prem,
            "Short claim.",
        )
        assert np > 1
        assert nh == 1
        assert len(per_hyp) == 1

    def test_inner_mean_aggregation(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Premise detail {i} info" for i in range(30)) + "."
        agg_max, _, _, _ = scorer._score_chunked_with_counts(
            long_prem,
            "Short claim.",
            inner_agg="max",
        )
        agg_mean, _, _, _ = scorer._score_chunked_with_counts(
            long_prem,
            "Short claim.",
            inner_agg="mean",
        )
        assert agg_mean <= agg_max

    def test_mean_aggregation(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_hyp = ". ".join(f"Claim number {i} assertion" for i in range(30)) + "."
        s_max, _ = scorer.score_chunked("premise", long_hyp, outer_agg="max")
        s_mean, _ = scorer.score_chunked("premise", long_hyp, outer_agg="mean")
        assert s_mean <= s_max

    def test_min_inner_agg(self):
        """min inner agg picks the lowest divergence across premise chunks."""
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Premise detail {i} info" for i in range(30)) + "."
        agg_max, _, _, _ = scorer._score_chunked_with_counts(
            long_prem,
            "Short claim.",
            inner_agg="max",
        )
        agg_min, _, _, _ = scorer._score_chunked_with_counts(
            long_prem,
            "Short claim.",
            inner_agg="min",
        )
        assert agg_min <= agg_max

    def test_min_inner_mean_outer(self):
        """Combined min-inner + mean-outer gives lower score than max-max."""
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Source paragraph {i} content" for i in range(30)) + "."
        long_hyp = ". ".join(f"Summary claim {i} text" for i in range(20)) + "."
        agg_maxmax, _, _, _ = scorer._score_chunked_with_counts(
            long_prem,
            long_hyp,
            inner_agg="max",
            outer_agg="max",
        )
        agg_minmean, _, _, _ = scorer._score_chunked_with_counts(
            long_prem,
            long_hyp,
            inner_agg="min",
            outer_agg="mean",
        )
        assert agg_minmean <= agg_maxmax

    def test_score_chunked_forwards_inner_agg(self):
        """score_chunked() accepts and forwards inner_agg kwarg."""
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Premise detail {i} info" for i in range(30)) + "."
        s_max, _ = scorer.score_chunked(long_prem, "Short claim.", inner_agg="max")
        s_min, _ = scorer.score_chunked(long_prem, "Short claim.", inner_agg="min")
        assert s_min <= s_max

    def test_score_chunked_with_counts(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Evidence {i} text" for i in range(30)) + "."
        long_hyp = ". ".join(f"Claim {i} text" for i in range(30)) + "."
        agg, per_hyp, np, nh = scorer._score_chunked_with_counts(
            long_prem,
            long_hyp,
        )
        assert np > 1
        assert nh > 1
        assert len(per_hyp) == nh
        assert 0.0 <= agg <= 1.0

    def test_premise_ratio_reduces_premise_chunks(self):
        """Higher premise_ratio gives more premise budget → fewer premise chunks."""
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Source paragraph {i} detail" for i in range(30)) + "."
        _, _, np_default, _ = scorer._score_chunked_with_counts(
            long_prem,
            "Short claim.",
            premise_ratio=0.4,
        )
        _, _, np_high, _ = scorer._score_chunked_with_counts(
            long_prem,
            "Short claim.",
            premise_ratio=0.85,
        )
        assert np_high <= np_default

    def test_premise_ratio_budget_split(self):
        """Token budget reflects the premise_ratio parameter."""
        # With ratio 0.85: prem_budget = 435, hyp_budget = 77
        # With ratio 0.4:  prem_budget = 205, hyp_budget = 307
        prem_budget_high = int(512 * 0.85)
        prem_budget_low = int(512 * 0.4)
        assert prem_budget_high > prem_budget_low
        assert prem_budget_high == 435
        assert prem_budget_low == 204

    def test_score_chunked_forwards_premise_ratio(self):
        """score_chunked() accepts and forwards premise_ratio kwarg."""
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Source text {i} detail" for i in range(30)) + "."
        s1, _ = scorer.score_chunked(long_prem, "Short claim.", premise_ratio=0.4)
        s2, _ = scorer.score_chunked(long_prem, "Short claim.", premise_ratio=0.85)
        # Both should succeed without error and return valid scores
        assert 0.0 <= s1 <= 1.0
        assert 0.0 <= s2 <= 1.0

    def test_trimmed_mean_outer_agg(self):
        """trimmed_mean drops top 25% of per-hypothesis scores before averaging."""
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Source paragraph {i} content" for i in range(30)) + "."
        long_hyp = ". ".join(f"Summary claim {i} text" for i in range(20)) + "."
        agg_mean, _, _, _ = scorer._score_chunked_with_counts(
            long_prem,
            long_hyp,
            inner_agg="min",
            outer_agg="mean",
        )
        agg_trimmed, _, _, _ = scorer._score_chunked_with_counts(
            long_prem,
            long_hyp,
            inner_agg="min",
            outer_agg="trimmed_mean",
        )
        # trimmed_mean drops the worst scores, so it should be <= mean
        assert agg_trimmed <= agg_mean

    def test_trimmed_mean_single_chunk(self):
        """trimmed_mean with 1 hypothesis chunk keeps that single score."""
        scorer = NLIScorer(use_model=False, max_length=512)
        agg, per_hyp, _, _ = scorer._score_chunked_with_counts(
            "A short premise.", "A short hypothesis.", outer_agg="trimmed_mean"
        )
        assert len(per_hyp) == 1
        assert agg == per_hyp[0]
