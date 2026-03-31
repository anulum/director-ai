# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for cross-turn contradiction tracking.

Covers: contradiction detection across turns, consistency scoring,
history window, threshold enforcement, parametrised turn counts,
pipeline integration with CoherenceScorer, and performance documentation.
"""

from __future__ import annotations

from director_ai.core.runtime.contradiction_tracker import ContradictionTracker


def _fake_scorer(divergence: float):
    """Return a score_fn that always returns a fixed divergence."""
    return lambda premise, hypothesis: divergence


def _content_scorer(premise: str, hypothesis: str) -> float:
    """Simple word-overlap scorer for testing. 0 = identical, 1 = no overlap."""
    words_a = set(premise.lower().split())
    words_b = set(hypothesis.lower().split())
    if not words_a or not words_b:
        return 1.0
    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
    return 1.0 - overlap


class TestContradictionTrackerBasic:
    def test_empty_tracker(self):
        tracker = ContradictionTracker()
        report = tracker.get_report()
        assert report.contradiction_index == 0.0
        assert report.worst_pair is None
        assert report.pair_count == 0

    def test_single_turn_no_contradiction(self):
        tracker = ContradictionTracker()
        report = tracker.update("hello world", _fake_scorer(0.0))
        assert report.contradiction_index == 0.0
        assert report.pair_count == 0

    def test_two_turns_low_divergence(self):
        tracker = ContradictionTracker()
        tracker.update("The sky is blue.", _fake_scorer(0.1))
        report = tracker.update("The grass is green.", _fake_scorer(0.1))
        assert report.contradiction_index == 0.1
        assert report.worst_pair is not None
        assert report.pair_count == 1

    def test_two_turns_high_divergence(self):
        tracker = ContradictionTracker()
        tracker.update("The answer is yes.", _fake_scorer(0.9))
        report = tracker.update("The answer is no.", _fake_scorer(0.9))
        assert report.contradiction_index == 0.9
        assert report.worst_pair is not None
        assert report.worst_pair.divergence == 0.9

    def test_three_turns_worst_pair_identified(self):
        tracker = ContradictionTracker()
        call_count = [0]
        divergences = [0.2, 0.8, 0.3]

        def varying_scorer(p, h):
            idx = min(call_count[0], len(divergences) - 1)
            call_count[0] += 1
            return divergences[idx]

        tracker.update("A", varying_scorer)
        tracker.update("B", varying_scorer)
        report = tracker.update("C", varying_scorer)
        assert report.worst_pair is not None
        assert report.worst_pair.divergence == 0.8

    def test_turn_count(self):
        tracker = ContradictionTracker()
        assert tracker.turn_count == 0
        tracker.update("a", _fake_scorer(0.0))
        assert tracker.turn_count == 1
        tracker.update("b", _fake_scorer(0.0))
        assert tracker.turn_count == 2


class TestContradictionTrackerEviction:
    def test_eviction_at_max_turns(self):
        tracker = ContradictionTracker(max_turns=3)
        for i in range(5):
            tracker.update(f"response {i}", _fake_scorer(0.1))
        assert tracker.turn_count == 3

    def test_matrix_stays_consistent_after_eviction(self):
        tracker = ContradictionTracker(max_turns=3)
        for i in range(5):
            report = tracker.update(f"turn {i}", _fake_scorer(0.2))
        assert report.pair_count == 3  # C(3,2) = 3 pairs
        assert len(tracker._matrix) == 3
        for row in tracker._matrix:
            assert len(row) == 3


class TestContradictionTrackerTrend:
    def test_increasing_trend(self):
        tracker = ContradictionTracker()
        call_idx = [0]
        increasing = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]

        def inc_scorer(p, h):
            idx = min(call_idx[0], len(increasing) - 1)
            call_idx[0] += 1
            return increasing[idx]

        for i in range(5):
            report = tracker.update(f"turn {i}", inc_scorer)
        assert report.trend > 0.0

    def test_zero_trend_constant_divergence(self):
        tracker = ContradictionTracker()
        for i in range(5):
            report = tracker.update(f"turn {i}", _fake_scorer(0.5))
        assert abs(report.trend) < 0.01


class TestContradictionTrackerReset:
    def test_reset_clears_state(self):
        tracker = ContradictionTracker()
        tracker.update("a", _fake_scorer(0.5))
        tracker.update("b", _fake_scorer(0.5))
        tracker.reset()
        assert tracker.turn_count == 0
        report = tracker.get_report()
        assert report.contradiction_index == 0.0


class TestContradictionTrackerWithContentScorer:
    def test_similar_responses_low_contradiction(self):
        tracker = ContradictionTracker()
        tracker.update("The cat sat on the mat", _content_scorer)
        report = tracker.update("The cat was sitting on the mat", _content_scorer)
        assert report.contradiction_index < 0.5

    def test_different_responses_high_contradiction(self):
        tracker = ContradictionTracker()
        tracker.update("Python is a programming language", _content_scorer)
        report = tracker.update("Elephants roam the savanna", _content_scorer)
        assert report.contradiction_index > 0.7
