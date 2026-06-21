# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Standalone Director-Lite streaming-halt guard tests."""

from __future__ import annotations

import pytest
from director_ai_lite import StreamGuard, StreamResult, guard, streaming_guard
from director_ai_lite._coherence import combine_weighted_coherence
from director_ai_lite.guard import (
    _content_words,
    _factual_divergence,
)

_FACTS = {"fr": "Paris is the capital of France."}
_FACT_WORDS = _content_words("Paris is the capital of France.")


def test_content_words_drops_stopwords_and_short_tokens():
    assert _content_words("The cat is on a mat") == {"cat", "mat"}
    assert _content_words("") == set()


def test_factual_divergence_neutral_without_facts():
    assert _factual_divergence("anything at all", set()) == 0.5


def test_factual_divergence_zero_when_no_content_words():
    assert _factual_divergence("the is on a", _FACT_WORDS) == 0.0


def test_factual_divergence_full_when_unattested():
    assert _factual_divergence("Berlin Tokyo Mars", _FACT_WORDS) == 1.0


def test_factual_divergence_partial_grounding():
    # "capital" is grounded, "Mars" is not -> divergence 0.5.
    assert _factual_divergence("capital Mars", _FACT_WORDS) == pytest.approx(0.5)


def test_grounded_stream_is_not_halted():
    result = guard(
        ["Paris ", "is ", "the ", "capital ", "of ", "France."],
        facts=_FACTS,
        prompt="capital of France?",
    )
    assert result.halted is False
    assert result.halt_index == -1
    assert result.output == "Paris is the capital of France."


def test_ungrounded_drift_halts_and_truncates_output():
    result = guard(
        ["The ", "capital ", "is ", "Berlin ", "Tokyo ", "Mars ", "banana"],
        facts=_FACTS,
        threshold=0.5,
    )
    assert result.halted is True
    assert result.halt_index >= 0
    assert "below threshold" in result.halt_reason
    assert "banana" not in result.output


def test_no_facts_stays_neutral_above_low_threshold():
    result = guard(["hello ", "world"], threshold=0.4)
    assert result.halted is False


def test_stream_result_output_without_halt_returns_all_tokens():
    result = StreamResult(tokens=["a", "b", "c"])
    assert result.output == "abc"


def test_threshold_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="threshold must be in"):
        StreamGuard(threshold=1.5)
    with pytest.raises(ValueError, match="threshold must be in"):
        StreamGuard(threshold=-0.1)


def test_safe_text_returns_only_surviving_output():
    g = StreamGuard(_FACTS, threshold=0.5)
    text = g.safe_text(["Paris ", "is ", "Mars ", "Jupiter ", "banana"], prompt="q")
    assert "banana" not in text


def test_streaming_guard_one_call_matches_class():
    direct = streaming_guard(["Paris ", "France."], facts=_FACTS)
    assert isinstance(direct, StreamResult)
    assert direct.halted is False


def test_injected_scorer_overrides_heuristic():
    class _Score:
        score = 0.1

    class _Scorer:
        def review(self, prompt: str, text: str):
            return None, _Score()

    # The scorer forces a low coherence, so the stream halts on the first token
    # regardless of grounding.
    result = guard(["fully ", "grounded ", "Paris"], facts=_FACTS, scorer=_Scorer())
    assert result.halted is True
    assert result.halt_index == 0


def test_combine_weighted_coherence_plain_factual():
    # w_logic=0, w_fact=1, divergence 0.4 -> coherence 0.6, no NLI renormalisation.
    assert combine_weighted_coherence(
        h_logic=0.0,
        h_factual=0.4,
        w_logic=0.0,
        w_fact=1.0,
        nli_available=False,
        evidence_present=True,
        dialogue_route=False,
    ) == pytest.approx(0.6)


def test_combine_weighted_coherence_no_kb_renormalisation():
    # Neutral factual score, NLI available, no evidence -> the no-KB branch rescales.
    value = combine_weighted_coherence(
        h_logic=0.0,
        h_factual=0.5,
        w_logic=0.2,
        w_fact=0.8,
        nli_available=True,
        evidence_present=False,
        dialogue_route=False,
    )
    assert 0.0 <= value <= 1.0


def test_combine_weighted_coherence_clamps_to_unit_interval():
    assert (
        combine_weighted_coherence(
            h_logic=1.0,
            h_factual=1.0,
            w_logic=1.0,
            w_fact=1.0,
            nli_available=False,
            evidence_present=True,
            dialogue_route=False,
        )
        == 0.0
    )
