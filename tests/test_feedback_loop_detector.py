# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Feedback-loop detector contracts
"""Behavioural tests for compliance feedback-loop detection."""

from director_ai.compliance.feedback_loop_detector import (
    FeedbackLoopDetector,
    _jaccard_similarity,
    _trigram_set,
)


def test_record_output_updates_detector_buffer() -> None:
    detector = FeedbackLoopDetector()

    detector.record_output("Some output here for testing", 0.9)

    assert detector.buffer_size == 1


def test_matching_prior_output_returns_high_severity_alert() -> None:
    detector = FeedbackLoopDetector()
    prior_output = "Previously seen output text"
    detector.record_output(prior_output, 0.8)

    alert = detector.check_input(prior_output)

    assert alert is not None
    assert alert.matched_output == prior_output
    assert alert.output_timestamp == 0.8
    assert alert.similarity >= detector.similarity_threshold
    assert alert.severity == "high"


def test_check_and_record_does_not_alert_on_distinct_first_input() -> None:
    detector = FeedbackLoopDetector()

    alert = detector.check_and_record(
        "What is the refund policy?",
        "The refund policy is 30 days.",
        1.0,
    )

    assert alert is None or alert.similarity >= detector.similarity_threshold


def test_short_text_trigrams_and_empty_similarity_are_stable() -> None:
    assert _trigram_set("AI") == {"ai"}
    assert _jaccard_similarity(set(), {"abc"}) == 0.0
    assert _jaccard_similarity({"abc"}, set()) == 0.0


def test_short_outputs_and_inputs_are_ignored() -> None:
    detector = FeedbackLoopDetector(min_text_length=20)

    detector.record_output("too short", 1.0)
    alert = detector.check_input("also short")

    assert detector.buffer_size == 0
    assert alert is None


def test_buffer_size_limit_evicts_oldest_output() -> None:
    detector = FeedbackLoopDetector(max_buffer_size=2, min_text_length=3)

    detector.record_output("first output", 1.0)
    detector.record_output("second output", 2.0)
    detector.record_output("third output", 3.0)

    assert detector.buffer_size == 2
    assert detector.check_input("first output") is None
    alert = detector.check_input("third output")
    assert alert is not None
    assert alert.output_timestamp == 3.0


def test_check_input_returns_none_when_no_prior_output_crosses_threshold() -> None:
    detector = FeedbackLoopDetector(similarity_threshold=0.7, min_text_length=3)
    detector.record_output("alpha beta gamma", 1.0)
    detector.record_output("delta epsilon zeta", 2.0)

    assert detector.check_input("unrelated theta iota") is None


def test_severity_thresholds_cover_medium_and_low_alerts() -> None:
    high_context = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

    medium_detector = FeedbackLoopDetector(similarity_threshold=0.5)
    medium_detector.record_output(high_context, 1.0)
    medium_alert = medium_detector.check_input(
        "prefix alpha beta gamma delta epsilon zeta eta theta iota suffix"
    )

    low_detector = FeedbackLoopDetector(similarity_threshold=0.4)
    low_detector.record_output(high_context, 2.0)
    low_alert = low_detector.check_input(
        "prefix alpha beta gamma delta epsilon zeta suffix"
    )

    assert medium_alert is not None
    assert 0.6 < medium_alert.similarity <= 0.8
    assert medium_alert.severity == "medium"
    assert low_alert is not None
    assert 0.4 <= low_alert.similarity <= 0.6
    assert low_alert.severity == "low"
