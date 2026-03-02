# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Property-Based Fuzz Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

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
