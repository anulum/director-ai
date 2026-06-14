# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — pre-generation hallucination forecaster tests

"""Risk-band, signal, polyglot-parity and guard-wiring coverage for the
pre-generation hallucination forecaster."""

from __future__ import annotations

import pytest

from director_ai.core.forecasting import (
    ForecastHistory,
    ForecastResult,
    HallucinationForecaster,
)
from director_ai.core.forecasting import hallucination_forecaster as hf
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.guard import ProductionGuard


class _StubStore:
    """Minimal store returning a fixed semicolon-joined context string."""

    def __init__(self, context: str | None) -> None:
        self._context = context

    def retrieve_context(self, _query: str) -> str | None:
        return self._context


# --------------------------------------------------------------------------- #
# _lexical_overlap — both backends                                            #
# --------------------------------------------------------------------------- #


def test_lexical_overlap_rust_path_matches_expected_jaccard() -> None:
    # With the kernel installed this exercises the Rust branch.
    assert hf._lexical_overlap("a b c", "b c d") == pytest.approx(0.5)


def test_lexical_overlap_python_fallback_parity(monkeypatch) -> None:
    monkeypatch.setattr(hf, "_RUST_FORECAST", False)
    monkeypatch.setattr(hf, "rust_word_overlap", None)
    assert hf._lexical_overlap("a b c", "b c d") == pytest.approx(0.5)


def test_lexical_overlap_python_fallback_empty_returns_zero(monkeypatch) -> None:
    monkeypatch.setattr(hf, "_RUST_FORECAST", False)
    monkeypatch.setattr(hf, "rust_word_overlap", None)
    assert hf._lexical_overlap("", "anything") == 0.0
    assert hf._lexical_overlap("anything", "") == 0.0


def test_rust_and_python_overlap_agree_across_cases(monkeypatch) -> None:
    cases = [
        ("the capital of france", "capital of france is paris"),
        ("photosynthesis in plants", "plants convert light to energy"),
        ("disjoint tokens here", "completely different words there"),
    ]
    rust = [hf._lexical_overlap(a, b) for a, b in cases]
    monkeypatch.setattr(hf, "_RUST_FORECAST", False)
    monkeypatch.setattr(hf, "rust_word_overlap", None)
    py = [hf._lexical_overlap(a, b) for a, b in cases]
    for r, p in zip(rust, py, strict=True):
        assert r == pytest.approx(p)


# --------------------------------------------------------------------------- #
# _as_facts / _has_anchor                                                      #
# --------------------------------------------------------------------------- #


def test_as_facts_handles_none_empty_and_splitting() -> None:
    assert hf._as_facts(None) == []
    assert hf._as_facts("") == []
    assert hf._as_facts(" a ; ; b ") == ["a", "b"]


def test_has_anchor_detects_digit_proper_noun_and_rejects_plain() -> None:
    assert hf._has_anchor("launched in 1969") is True
    assert hf._has_anchor("the Eiffel tower") is True  # capitalised non-first
    assert hf._has_anchor("tell me something vague") is False
    assert hf._has_anchor("Capital first word only") is False
    assert hf._has_anchor("") is False


# --------------------------------------------------------------------------- #
# ForecastHistory                                                              #
# --------------------------------------------------------------------------- #


def test_history_signature_buckets_and_empty_prompt() -> None:
    assert ForecastHistory.signature("") == "|short|vague"
    sig = ForecastHistory.signature("What is the capital of France in 1789?")
    assert sig.startswith("what|") and sig.endswith("|anchored")


def test_history_rate_unseen_is_none_then_tracks_outcomes() -> None:
    history = ForecastHistory()
    assert history.rate("What is the capital of Atlantis?") is None
    history.record("What is the capital of Atlantis?", hallucinated=True)
    history.record("What is the capital of Atlantis?", hallucinated=False)
    assert history.rate("What is the capital of Atlantis?") == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# HallucinationForecaster validation                                          #
# --------------------------------------------------------------------------- #


def test_post_init_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        HallucinationForecaster(weight_ambiguity=-0.1)


def test_post_init_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="ground_threshold"):
        HallucinationForecaster(ground_threshold=0.8, review_threshold=0.5)


def test_post_init_rejects_out_of_range_prior() -> None:
    with pytest.raises(ValueError, match="no_kb_prior"):
        HallucinationForecaster(no_kb_prior=1.5)


# --------------------------------------------------------------------------- #
# ambiguity signal                                                             #
# --------------------------------------------------------------------------- #


def test_ambiguity_empty_prompt_is_maximal() -> None:
    assert HallucinationForecaster().ambiguity("   ") == 1.0


def test_ambiguity_vague_short_unanchored_scores_high() -> None:
    score = HallucinationForecaster().ambiguity("tell me something")
    assert score >= 0.7


def test_ambiguity_specific_anchored_scores_low() -> None:
    score = HallucinationForecaster().ambiguity(
        "What was the launch date of Apollo 11 in 1969?"
    )
    assert score <= 0.1


def test_ambiguity_multi_intent_adds_weight() -> None:
    # Identical word tokens (the "?" is stripped by the tokeniser); only the
    # number of question marks differs, isolating the multi-intent term.
    f = HallucinationForecaster()
    single = f.ambiguity("Apollo 11 in 1969 area Florida?")
    multi = f.ambiguity("Apollo 11 in 1969? area Florida?")
    assert multi > single


# --------------------------------------------------------------------------- #
# kb_coverage signal                                                           #
# --------------------------------------------------------------------------- #


def test_kb_coverage_none_without_store() -> None:
    assert HallucinationForecaster().kb_coverage("anything", None) is None


def test_kb_coverage_zero_when_nothing_retrieved() -> None:
    f = HallucinationForecaster()
    assert f.kb_coverage("anything", _StubStore(None)) == 0.0
    assert f.kb_coverage("anything", _StubStore("")) == 0.0


def test_kb_coverage_takes_best_overlap_across_facts() -> None:
    f = HallucinationForecaster()
    store = _StubStore("unrelated text; the capital of France is Paris")
    coverage = f.kb_coverage("what is the capital of France", store)
    assert 0.0 < coverage <= 1.0


# --------------------------------------------------------------------------- #
# forecast integration                                                         #
# --------------------------------------------------------------------------- #


def test_forecast_no_store_flags_missing_kb_and_returns_result() -> None:
    result = HallucinationForecaster().forecast("What is the capital of France?")
    assert isinstance(result, ForecastResult)
    assert result.kb_coverage is None
    assert "no knowledge base supplied" in result.rationale


def test_forecast_high_risk_under_specified_recommends_review() -> None:
    result = HallucinationForecaster(no_kb_prior=1.0).forecast("tell me something")
    assert result.recommendation == "human_review"
    assert "under-specified prompt" in result.rationale


def test_forecast_well_grounded_low_risk_recommends_proceed() -> None:
    store = _StubStore("the capital of France is Paris located in France")
    result = HallucinationForecaster().forecast(
        "What is the capital of France located in France?", store=store
    )
    assert result.recommendation == "proceed"
    assert result.rationale == ("well-specified and grounded",)


def test_forecast_weak_coverage_appends_rationale() -> None:
    store = _StubStore("totally unrelated content about weather")
    result = HallucinationForecaster().forecast(
        "What is the capital of France in 1789?", store=store
    )
    assert "weak knowledge-base coverage" in result.rationale


def test_forecast_history_signal_raises_risk_and_rationale() -> None:
    history = ForecastHistory()
    prompt = "What is the capital of Atlantis in 1900?"
    for _ in range(3):
        history.record(prompt, hallucinated=True)
    f = HallucinationForecaster(history=history)
    result = f.forecast(prompt)
    assert result.pattern_risk == pytest.approx(1.0)
    assert "this prompt shape has hallucinated before" in result.rationale


def test_forecast_zero_weights_yield_zero_risk() -> None:
    f = HallucinationForecaster(
        weight_ambiguity=0.0, weight_kb=0.0, weight_history=0.0
    )
    result = f.forecast("tell me something")
    assert result.risk == 0.0
    assert result.recommendation == "proceed"


def test_forecast_history_absent_match_drops_history_weight() -> None:
    # History object present but this prompt unseen -> pattern_risk 0, weight reallocated.
    f = HallucinationForecaster(history=ForecastHistory())
    result = f.forecast("What is the capital of France in 1789?")
    assert result.pattern_risk == 0.0


# --------------------------------------------------------------------------- #
# ProductionGuard wiring                                                       #
# --------------------------------------------------------------------------- #


def test_guard_forecast_uses_its_ground_truth_store() -> None:
    guard = ProductionGuard()
    guard._store.add("capital of France", "The capital of France is Paris.")
    result = guard.forecast("What is the capital of France?")
    assert isinstance(result, ForecastResult)
    assert result.kb_coverage is not None


def test_guard_forecaster_and_history_persist_across_calls() -> None:
    guard = ProductionGuard()
    first = guard._ensure_forecaster()
    assert guard._ensure_forecaster() is first
    prompt = "What is the capital of Atlantis in 1900?"
    guard.forecast_history.record(prompt, hallucinated=True)
    guard.forecast_history.record(prompt, hallucinated=True)
    result = guard.forecast(prompt)
    assert result.pattern_risk == pytest.approx(1.0)


def test_real_keyword_store_round_trip_through_guard() -> None:
    store = GroundTruthStore()
    store.add("capital of France", "The capital of France is Paris.")
    guard = ProductionGuard(store=store)
    result = guard.forecast("What is the capital of France?")
    assert 0.0 <= result.risk <= 1.0
    assert result.recommendation in {"proceed", "ground", "human_review"}
