# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — per-segment adaptive threshold tests

"""Routing, cold-start fallback, per-segment specialisation and serialisation
coverage for the segmented adaptive-threshold learner."""

from __future__ import annotations

import random

import pytest

from director_ai.core.calibration.segmented_threshold import (
    SegmentedThresholdLearner,
    SegmentRecommendation,
)


def _learner(**overrides):
    kwargs = {
        "candidate_thresholds": [0.3, 0.5, 0.7, 0.9],
        "current_threshold": 0.5,
        "min_samples": 10,
        "random_seed": 42,
    }
    kwargs.update(overrides)
    return SegmentedThresholdLearner(**kwargs)


def _feed(learner, segment, n, approve_at, *, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        score = rng.random()
        learner.observe(score, human_approved=(score >= approve_at), segment=segment)


# --------------------------------------------------------------------------- #
# construction & validation                                                    #
# --------------------------------------------------------------------------- #


def test_promote_after_defaults_to_min_samples():
    learner = _learner(min_samples=15)
    _feed(learner, "x", 14, 0.5)
    assert learner.recommend(segment="x").source == "global"
    _feed(learner, "x", 1, 0.5)
    assert learner.recommend(segment="x").source == "segment"


def test_promote_after_custom_value():
    learner = _learner(promote_after=5)
    _feed(learner, "x", 5, 0.5)
    assert learner.recommend(segment="x").source == "segment"


def test_promote_after_must_be_positive():
    with pytest.raises(ValueError, match="promote_after"):
        _learner(promote_after=0)


@pytest.mark.parametrize("bad", ["", "   ", "__global__"])
def test_invalid_segment_keys_raise(bad):
    learner = _learner()
    with pytest.raises(ValueError):
        learner.observe(0.5, True, segment=bad)


def test_segment_key_is_stripped():
    learner = _learner(promote_after=1)
    learner.observe(0.9, True, segment="  clinical  ")
    assert learner.segments() == ["clinical"]


# --------------------------------------------------------------------------- #
# routing & cold start                                                         #
# --------------------------------------------------------------------------- #


def test_cold_segment_falls_back_to_global_pool():
    learner = _learner()
    _feed(learner, "seen", 30, 0.5)
    rec = learner.recommend(segment="brand_new")
    assert isinstance(rec, SegmentRecommendation)
    assert rec.source == "global"
    assert rec.feedback_count == 0


def test_observe_updates_segment_and_global_counts():
    learner = _learner()
    report = learner.observe(0.8, True, segment="a")
    assert report.total_feedback == 1
    # The global pool also saw it: a second segment's global fallback reflects it.
    assert learner.report().total_feedback == 1
    learner.observe(0.4, False, segment="b")
    assert learner.report().total_feedback == 2


def test_seedless_learner_is_constructible_and_observes():
    learner = _learner(random_seed=None)
    rec = learner.observe(0.7, True, segment="x")
    assert rec.total_feedback == 1


# --------------------------------------------------------------------------- #
# per-segment specialisation                                                   #
# --------------------------------------------------------------------------- #


def test_segments_recover_distinct_optimal_thresholds():
    learner = _learner(min_samples=10)
    # clinical: only very-high coherence is approved -> a high threshold is best.
    _feed(learner, "clinical", 80, 0.8, seed=1)
    # chat: most answers approved -> a low threshold is best.
    _feed(learner, "chat", 80, 0.3, seed=2)

    clinical = learner.recommend(segment="clinical").recommendation
    chat = learner.recommend(segment="chat").recommendation
    assert clinical.recommended_threshold is not None
    assert chat.recommended_threshold is not None
    assert clinical.recommended_threshold > chat.recommended_threshold


# --------------------------------------------------------------------------- #
# reporting & serialisation                                                    #
# --------------------------------------------------------------------------- #


def test_report_global_and_per_segment():
    learner = _learner()
    _feed(learner, "a", 5, 0.5)
    assert learner.report().total_feedback == 5  # global
    assert learner.report(segment="a").total_feedback == 5
    # An unseen segment reports an empty (freshly created) learner.
    assert learner.report(segment="unseen").total_feedback == 0


def test_to_dict_round_trip_shape():
    learner = _learner(promote_after=3)
    _feed(learner, "a", 4, 0.5)
    payload = learner.to_dict()
    assert payload["promote_after"] == 3
    assert payload["global"]["total_feedback"] == 4
    assert payload["segments"]["a"]["feedback_count"] == 4
    assert "report" in payload["segments"]["a"]


def test_segments_listed_in_first_seen_order():
    learner = _learner()
    learner.observe(0.5, True, segment="z")
    learner.observe(0.5, True, segment="a")
    assert learner.segments() == ["z", "a"]
