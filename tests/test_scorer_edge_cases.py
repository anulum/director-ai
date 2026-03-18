# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from director_ai.core import CoherenceScorer


def test_empty_prompt():
    scorer = CoherenceScorer(use_nli=False)
    approved, score = scorer.review("", "Some response")
    assert isinstance(approved, bool)
    assert 0.0 <= score.score <= 1.0


def test_empty_response():
    scorer = CoherenceScorer(use_nli=False)
    approved, score = scorer.review("What is 2+2?", "")
    assert isinstance(approved, bool)


def test_very_long_response():
    scorer = CoherenceScorer(use_nli=False)
    long_text = "word " * 100_000  # 500K chars
    approved, score = scorer.review("Summarize", long_text)
    assert isinstance(approved, bool)


def test_unicode_emoji():
    scorer = CoherenceScorer(use_nli=False)
    approved, score = scorer.review("What is this? 🎉🥳", "It is a celebration 🎉")
    assert isinstance(approved, bool)


def test_null_bytes():
    scorer = CoherenceScorer(use_nli=False)
    approved, score = scorer.review("test\x00prompt", "test\x00response")
    assert isinstance(approved, bool)


def test_rtl_text():
    scorer = CoherenceScorer(use_nli=False)
    approved, score = scorer.review("ما هو 2+2؟", "الإجابة هي 4")
    assert isinstance(approved, bool)


def test_score_in_range():
    scorer = CoherenceScorer(use_nli=False)
    for _ in range(10):
        approved, score = scorer.review("What is AI?", "AI is artificial intelligence.")
        assert 0.0 <= score.score <= 1.0
        assert 0.0 <= score.h_logical <= 1.0
        assert 0.0 <= score.h_factual <= 1.0


def test_deterministic():
    scorer = CoherenceScorer(use_nli=False)
    _, s1 = scorer.review("What?", "Answer.")
    _, s2 = scorer.review("What?", "Answer.")
    assert s1.score == s2.score
