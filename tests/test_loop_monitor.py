# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for agentic loop monitor pipeline (STRONG)."""

from __future__ import annotations

from director_ai.agentic.loop_monitor import LoopMonitor


def _no_drift(goal, action):
    return 0.0


_NO_DRIFT = {"goal_drift_scorer": _no_drift}


class TestBasicOperation:
    def test_single_step_ok(self):
        m = LoopMonitor(goal="Find revenue data", **_NO_DRIFT)
        v = m.check_step(action="search", args="revenue Q3 2025")
        assert not v.should_halt
        assert v.step_number == 1

    def test_step_count_increments(self):
        m = LoopMonitor(goal="test", max_steps=100, **_NO_DRIFT)
        for i in range(5):
            v = m.check_step(action=f"tool_{i}")
        assert v.step_number == 5

    def test_status_tracks_steps(self):
        m = LoopMonitor(goal="test", **_NO_DRIFT)
        m.check_step(action="a")
        m.check_step(action="b")
        s = m.status()
        assert s.total_steps == 2
        assert not s.halted


class TestStepLimit:
    def test_exceeds_max_steps(self):
        m = LoopMonitor(goal="test", max_steps=3, **_NO_DRIFT)
        m.check_step(action="a")
        m.check_step(action="b")
        m.check_step(action="c")
        v = m.check_step(action="d")
        assert v.should_halt
        assert any("Step limit" in r for r in v.reasons)

    def test_at_limit_is_ok(self):
        m = LoopMonitor(goal="test", max_steps=3, **_NO_DRIFT)
        m.check_step(action="a")
        m.check_step(action="b")
        v = m.check_step(action="c")
        assert not v.should_halt


class TestTokenBudget:
    def test_exceeds_token_budget(self):
        m = LoopMonitor(goal="test", max_tokens=1000, **_NO_DRIFT)
        m.check_step(action="a", tokens=500)
        v = m.check_step(action="b", tokens=600)
        assert v.should_halt
        assert any("Token budget" in r for r in v.reasons)

    def test_low_budget_warns(self):
        m = LoopMonitor(goal="test", max_tokens=1000, **_NO_DRIFT)
        v = m.check_step(action="a", tokens=920)
        assert v.should_warn
        assert any("low" in r.lower() for r in v.reasons)

    def test_budget_remaining_pct(self):
        m = LoopMonitor(goal="test", max_tokens=1000, **_NO_DRIFT)
        v = m.check_step(action="a", tokens=300)
        assert abs(v.budget_remaining_pct - 0.7) < 0.01


class TestCircularDetection:
    def test_detects_circular_calls(self):
        m = LoopMonitor(goal="test", circular_threshold=3, **_NO_DRIFT)
        m.check_step(action="search", args="query X")
        m.check_step(action="search", args="query X")
        v = m.check_step(action="search", args="query X")
        assert v.should_warn
        assert any("Circular" in r for r in v.reasons)

    def test_different_args_not_circular(self):
        m = LoopMonitor(goal="test", circular_threshold=3, **_NO_DRIFT)
        m.check_step(action="search", args="query A")
        m.check_step(action="search", args="query B")
        v = m.check_step(action="search", args="query C")
        assert not v.should_warn

    def test_severe_circular_halts(self):
        m = LoopMonitor(goal="test", circular_threshold=2, **_NO_DRIFT)
        for _ in range(4):
            v = m.check_step(action="stuck_tool", args="same_args")
        assert v.should_halt
        assert any("Severe circular" in r for r in v.reasons)


class TestGoalDrift:
    def test_aligned_action(self):
        # Jaccard measures word overlap between goal and "action(args)"
        m = LoopMonitor(goal="find quarterly revenue")
        v = m.check_step(action="find quarterly revenue")
        assert v.goal_drift_score < 0.3

    def test_drifted_action(self):
        m = LoopMonitor(goal="Find quarterly revenue", goal_drift_threshold=0.5)
        v = m.check_step(action="delete_database", args="production")
        assert v.goal_drift_score > 0.5
        assert v.should_warn

    def test_custom_drift_scorer(self):
        def always_drifted(goal, action):
            return 0.95

        m = LoopMonitor(goal="test", goal_drift_scorer=always_drifted)
        v = m.check_step(action="anything")
        assert v.should_halt
        assert v.goal_drift_score == 0.95


class TestWallTime:
    def test_time_limit(self):
        import time

        m = LoopMonitor(goal="test", max_wall_seconds=0.01, **_NO_DRIFT)
        time.sleep(0.02)
        v = m.check_step(action="a")
        assert v.should_halt
        assert any("Wall time" in r for r in v.reasons)


class TestLoopStatus:
    def test_status_after_halt(self):
        m = LoopMonitor(goal="test", max_steps=2, **_NO_DRIFT)
        m.check_step(action="a")
        m.check_step(action="b")
        m.check_step(action="c")
        s = m.status()
        assert s.halted
        assert "Step limit" in s.halt_reason

    def test_circular_count_in_status(self):
        m = LoopMonitor(goal="test", circular_threshold=2, **_NO_DRIFT)
        m.check_step(action="x", args="y")
        m.check_step(action="x", args="y")
        s = m.status()
        assert s.circular_detections == 1

    def test_drift_count_in_status(self):
        m = LoopMonitor(
            goal="revenue analysis",
            goal_drift_threshold=0.3,
        )
        m.check_step(action="completely_unrelated_zzz_tool")
        s = m.status()
        assert s.goal_drift_alerts >= 1


class TestJaccardDrift:
    def test_identical(self):
        score = LoopMonitor._jaccard_drift("find revenue", "find revenue")
        assert score == 0.0

    def test_no_overlap(self):
        score = LoopMonitor._jaccard_drift("find revenue", "delete database")
        assert score == 1.0

    def test_partial_overlap(self):
        score = LoopMonitor._jaccard_drift("find revenue data", "find sales data")
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        assert LoopMonitor._jaccard_drift("", "test") == 1.0
        assert LoopMonitor._jaccard_drift("test", "") == 1.0
