# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Feedback-loop detector contracts
"""Behavioural tests for compliance feedback-loop detection."""

from director_ai.compliance.feedback_loop_detector import FeedbackLoopDetector


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
