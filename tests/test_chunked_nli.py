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

    def test_premise_truncated_for_long_input(self):
        scorer = NLIScorer(use_model=False, max_length=64)
        long_premise = "word " * 500
        long_hyp = ". ".join(f"Sentence {i} text" for i in range(20)) + "."
        score, chunk_scores = scorer.score_chunked(long_premise, long_hyp)
        assert 0.0 <= score <= 1.0
        assert len(chunk_scores) >= 1
