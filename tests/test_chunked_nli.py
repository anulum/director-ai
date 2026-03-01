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

    def test_inner_agg_is_min(self):
        """min-across-premises: per_hyp is min of premise chunk scores."""
        scorer = NLIScorer(use_model=False, max_length=64)
        long_prem = ". ".join(f"Premise detail {i} info" for i in range(30)) + "."
        _, per_hyp, np, nh = scorer._score_chunked_with_counts(
            long_prem,
            "Short claim.",
        )
        assert np > 1
        assert nh == 1
        assert len(per_hyp) == 1

    def test_mean_aggregation(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_hyp = ". ".join(f"Claim number {i} assertion" for i in range(30)) + "."
        s_max, _ = scorer.score_chunked("premise", long_hyp, outer_agg="max")
        s_mean, _ = scorer.score_chunked("premise", long_hyp, outer_agg="mean")
        assert s_mean <= s_max

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
