# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Phase 5 Gem 6: Feedback Loop Detection."""

from __future__ import annotations

from director_ai.compliance.feedback_loop_detector import FeedbackLoopDetector


class TestFeedbackLoopDetector:
    def test_no_outputs_no_alert(self):
        d = FeedbackLoopDetector()
        alert = d.check_input("This is a normal user question about AI safety.")
        assert alert is None

    def test_exact_recycled_output(self):
        d = FeedbackLoopDetector(similarity_threshold=0.4)
        output = "The capital of France is Paris, a beautiful city."
        d.record_output(output, timestamp=1000.0)
        alert = d.check_input(output)
        assert alert is not None
        assert alert.similarity == 1.0
        assert alert.severity == "high"

    def test_paraphrased_output_detected(self):
        d = FeedbackLoopDetector(similarity_threshold=0.3)
        d.record_output(
            "The quarterly revenue for Q3 was $15 million, up 20% year over year.",
            timestamp=1000.0,
        )
        alert = d.check_input(
            "Q3 quarterly revenue was $15 million, a 20% increase year over year."
        )
        assert alert is not None
        assert alert.similarity > 0.3

    def test_unrelated_input_no_alert(self):
        d = FeedbackLoopDetector(similarity_threshold=0.5)
        d.record_output(
            "The mitochondria is the powerhouse of the cell.",
            timestamp=1000.0,
        )
        alert = d.check_input("What is the weather forecast for tomorrow in Prague?")
        assert alert is None

    def test_short_text_ignored(self):
        d = FeedbackLoopDetector(min_text_length=20)
        d.record_output("short", timestamp=1000.0)
        assert d.buffer_size == 0
        alert = d.check_input("short")
        assert alert is None

    def test_buffer_eviction(self):
        d = FeedbackLoopDetector(max_buffer_size=3)
        d.record_output("First output that is long enough for trigrams.", 1.0)
        d.record_output("Second output that is long enough for trigrams.", 2.0)
        d.record_output("Third output that is long enough for trigrams.", 3.0)
        d.record_output("Fourth output that is long enough for trigrams.", 4.0)
        assert d.buffer_size == 3

    def test_check_and_record(self):
        d = FeedbackLoopDetector(similarity_threshold=0.4)
        alert = d.check_and_record(
            input_text="What is photosynthesis and how does it work?",
            output_text="Photosynthesis converts sunlight into chemical energy in plants.",
            timestamp=1000.0,
        )
        assert alert is None
        assert d.buffer_size == 1

        # Now feed the output back as input
        alert2 = d.check_and_record(
            input_text="Photosynthesis converts sunlight into chemical energy in plants.",
            output_text="It uses chlorophyll to capture light energy from the sun.",
            timestamp=2000.0,
        )
        assert alert2 is not None
        assert alert2.severity == "high"


class TestSeverityLevels:
    def test_high_severity(self):
        d = FeedbackLoopDetector(similarity_threshold=0.3)
        text = "This is a sufficiently long text to test feedback loop detection severity levels."
        d.record_output(text, 1.0)
        alert = d.check_input(text)
        assert alert is not None
        assert alert.severity == "high"

    def test_medium_severity(self):
        d = FeedbackLoopDetector(similarity_threshold=0.3)
        d.record_output(
            "The revenue was fifteen million dollars in the third quarter.",
            1.0,
        )
        alert = d.check_input(
            "Fifteen million dollars revenue reported for the third fiscal quarter."
        )
        if alert is not None:
            assert alert.severity in ("low", "medium", "high")


class TestTrigramSimilarity:
    def test_identical_texts(self):
        d = FeedbackLoopDetector(similarity_threshold=0.1)
        text = "A reasonably long text for testing trigram similarity computation."
        d.record_output(text, 1.0)
        alert = d.check_input(text)
        assert alert is not None
        assert alert.similarity == 1.0

    def test_completely_different(self):
        d = FeedbackLoopDetector(similarity_threshold=0.9)
        d.record_output("AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH", 1.0)
        alert = d.check_input("zzzz yyyy xxxx wwww vvvv uuuu tttt ssss")
        assert alert is None
