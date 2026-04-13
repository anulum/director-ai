# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.agentic.swarm_metrics``.

Covers handoff recording, quarantine tracking, report generation,
per-agent rates, reset, thread safety, and edge cases.
"""

from __future__ import annotations

import threading

from director_ai.agentic.swarm_metrics import SwarmMetrics

# ── Handoff recording ──────────────────────────────────────────────────


class TestHandoffRecording:
    def test_single_handoff(self):
        m = SwarmMetrics()
        m.record_handoff("a", "b", 0.2)
        r = m.report()
        assert r["swarm"]["total_handoffs"] == 1

    def test_multiple_handoffs(self):
        m = SwarmMetrics()
        for i in range(10):
            m.record_handoff("a", "b", 0.1 * i)
        r = m.report()
        assert r["swarm"]["total_handoffs"] == 10

    def test_flagged_count(self):
        m = SwarmMetrics(hallucination_threshold=0.5)
        m.record_handoff("a", "b", 0.3)  # ok
        m.record_handoff("a", "b", 0.8)  # flagged
        m.record_handoff("a", "b", 0.6)  # flagged
        r = m.report()
        assert r["agents"]["a"]["flagged"] == 2

    def test_tracks_both_agents(self):
        m = SwarmMetrics()
        m.record_handoff("src", "dst", 0.1)
        r = m.report()
        assert "src" in r["agents"]
        assert "dst" in r["agents"]


# ── Quarantine tracking ────────────────────────────────────────────────


class TestQuarantineTracking:
    def test_quarantine_recorded(self):
        m = SwarmMetrics()
        m.record_quarantine("a")
        r = m.report()
        assert r["swarm"]["total_quarantines"] == 1
        assert r["agents"]["a"]["quarantined"] is True

    def test_cascade_event(self):
        m = SwarmMetrics()
        m.record_quarantine("a", cascade=True)
        r = m.report()
        assert r["swarm"]["cascade_events"] == 1

    def test_multiple_quarantines(self):
        m = SwarmMetrics()
        m.record_quarantine("a")
        m.record_quarantine("a")
        r = m.report()
        assert r["agents"]["a"]["quarantine_count"] == 2


# ── Report generation ──────────────────────────────────────────────────


class TestReport:
    def test_empty_report(self):
        m = SwarmMetrics()
        r = m.report()
        assert r["swarm"]["total_handoffs"] == 0
        assert r["swarm"]["active_agents"] == 0
        assert r["agents"] == {}

    def test_hallucination_rate(self):
        m = SwarmMetrics(hallucination_threshold=0.5)
        m.record_handoff("a", "b", 0.3)
        m.record_handoff("a", "b", 0.8)
        r = m.report()
        assert r["agents"]["a"]["hallucination_rate"] == 0.5

    def test_mean_score(self):
        m = SwarmMetrics()
        m.record_handoff("a", "b", 0.2)
        m.record_handoff("a", "b", 0.4)
        r = m.report()
        assert r["agents"]["a"]["mean_score"] == 0.3

    def test_quarantined_count(self):
        m = SwarmMetrics()
        m.record_handoff("a", "b", 0.1)
        m.record_handoff("c", "d", 0.1)
        m.record_quarantine("a")
        r = m.report()
        assert r["swarm"]["quarantined_agents"] == 1
        assert r["swarm"]["active_agents"] == 4  # a, b, c, d


# ── Per-agent rate ─────────────────────────────────────────────────────


class TestPerAgentRate:
    def test_rate_for_known_agent(self):
        m = SwarmMetrics(hallucination_threshold=0.5)
        m.record_handoff("a", "b", 0.8)
        m.record_handoff("a", "b", 0.3)
        assert m.agent_hallucination_rate("a") == 0.5

    def test_rate_for_unknown_agent(self):
        m = SwarmMetrics()
        assert m.agent_hallucination_rate("ghost") == 0.0

    def test_rate_zero_handoffs(self):
        m = SwarmMetrics()
        m.record_quarantine("a")
        assert m.agent_hallucination_rate("a") == 0.0


# ── Reset ──────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all(self):
        m = SwarmMetrics()
        m.record_handoff("a", "b", 0.5)
        m.record_quarantine("a")
        m.reset()
        r = m.report()
        assert r["swarm"]["total_handoffs"] == 0
        assert r["swarm"]["total_quarantines"] == 0
        assert r["agents"] == {}


# ── Thread safety ──────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_recording(self):
        m = SwarmMetrics()
        errors: list[Exception] = []

        def record_batch() -> None:
            try:
                for i in range(100):
                    m.record_handoff(f"agent-{i % 5}", "dst", 0.1 * (i % 10))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record_batch) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        r = m.report()
        assert r["swarm"]["total_handoffs"] == 400
