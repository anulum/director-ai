# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Property-Based Fuzz Tests (STRONG)
"""Multi-angle property-based fuzz tests for pipeline robustness (STRONG)."""

import os

import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _HAS_HYPOTHESIS = True
except ImportError:
    _HAS_HYPOTHESIS = False

    def settings(**_kw):  # type: ignore[misc]
        return lambda f: f

    def given(**_kw):  # type: ignore[misc]
        return lambda f: f

    class st:  # type: ignore[no-redef]  # noqa: N801
        @staticmethod
        def text(**_kw):
            return None

        @staticmethod
        def floats(**_kw):
            return None

        @staticmethod
        def lists(*_a, **_kw):
            return None


pytestmark = pytest.mark.skipif(not _HAS_HYPOTHESIS, reason="hypothesis not installed")

_MAX_EXAMPLES = int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "200"))


@settings(max_examples=_MAX_EXAMPLES, deadline=5000)
@given(
    prompt=st.text(min_size=0, max_size=500),
    action=st.text(min_size=0, max_size=500),
)
def test_review_never_crashes(prompt, action):
    from director_ai.core.scorer import CoherenceScorer

    scorer = CoherenceScorer(threshold=0.3, use_nli=False)
    approved, score = scorer.review(prompt, action)
    assert isinstance(approved, bool)
    assert 0.0 <= score.score <= 1.0


@settings(max_examples=_MAX_EXAMPLES, deadline=5000)
@given(text=st.text(min_size=0, max_size=1000))
def test_sanitizer_check_never_crashes(text):
    from director_ai.core.sanitizer import InputSanitizer

    san = InputSanitizer()
    result = san.check(text)
    assert isinstance(result.blocked, bool)


@settings(max_examples=_MAX_EXAMPLES, deadline=5000)
@given(text=st.text(min_size=0, max_size=1000))
def test_sanitizer_scrub_never_crashes(text):
    from director_ai.core.sanitizer import InputSanitizer

    result = InputSanitizer.scrub(text)
    assert isinstance(result, str)
    assert "\x00" not in result


@settings(max_examples=_MAX_EXAMPLES, deadline=5000)
@given(premise=st.text(min_size=1, max_size=300), hyp=st.text(min_size=1, max_size=300))
def test_scores_always_in_range(premise, hyp):
    from director_ai.core.lite_scorer import LiteScorer

    scorer = LiteScorer()
    s = scorer.score(premise, hyp)
    assert 0.0 <= s <= 1.0


@settings(max_examples=_MAX_EXAMPLES, deadline=5000)
@given(
    prompt=st.text(min_size=0, max_size=500),
    action=st.text(min_size=0, max_size=500),
    threshold=st.floats(min_value=0.0, max_value=1.0),
)
def test_review_with_any_threshold(prompt, action, threshold):
    from director_ai.core.scorer import CoherenceScorer

    scorer = CoherenceScorer(threshold=threshold, use_nli=False)
    approved, score = scorer.review(prompt, action)
    assert isinstance(approved, bool)
    assert 0.0 <= score.score <= 1.0


@settings(max_examples=_MAX_EXAMPLES // 2, deadline=5000)
@given(
    premise=st.text(min_size=1, max_size=200),
    hyp=st.text(min_size=1, max_size=200),
)
def test_lite_scorer_deterministic(premise, hyp):
    """Same input must always produce same score."""
    from director_ai.core.lite_scorer import LiteScorer

    scorer = LiteScorer()
    s1 = scorer.score(premise, hyp)
    s2 = scorer.score(premise, hyp)
    assert s1 == s2


@settings(max_examples=_MAX_EXAMPLES // 4, deadline=10000)
@given(
    texts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10),
)
def test_batch_scoring_length_matches(texts):
    """score_batch must return exactly len(pairs) scores."""
    from director_ai.core.lite_scorer import LiteScorer

    scorer = LiteScorer()
    pairs = [(t, t) for t in texts]
    results = scorer.score_batch(pairs)
    assert len(results) == len(pairs)
    assert all(0.0 <= s <= 1.0 for s in results)
