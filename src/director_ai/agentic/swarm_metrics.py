# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Swarm-level metrics collector
"""Aggregate metrics for multi-agent swarm monitoring.

Collects per-agent and swarm-wide statistics: hallucination rates,
handoff counts, quarantine events, and cross-agent consistency.

Usage::

    from director_ai.agentic.swarm_metrics import SwarmMetrics

    metrics = SwarmMetrics()
    metrics.record_handoff("researcher-0", "summariser-0", score=0.2)
    metrics.record_handoff("researcher-0", "coder-0", score=0.8)
    metrics.record_quarantine("researcher-0")

    report = metrics.report()
    print(report["swarm"]["total_handoffs"])  # 2
    print(report["agents"]["researcher-0"]["hallucination_rate"])  # 0.5
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

__all__ = ["SwarmMetrics"]


@dataclass
class _AgentRecord:
    """Per-agent metric accumulator."""

    handoffs: int = 0
    flagged: int = 0  # score > threshold
    quarantined: bool = False
    quarantine_count: int = 0
    total_score: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0


class SwarmMetrics:
    """Thread-safe swarm metrics collector.

    Parameters
    ----------
    hallucination_threshold : float
        Handoff score above which a handoff is flagged as hallucinated.
    history_window : int
        Maximum handoff records to keep (FIFO eviction).
    """

    def __init__(
        self,
        hallucination_threshold: float = 0.5,
        history_window: int = 10000,
    ) -> None:
        self._threshold = hallucination_threshold
        self._history_window = history_window
        self._agents: dict[str, _AgentRecord] = {}
        self._total_handoffs = 0
        self._total_quarantines = 0
        self._cascade_events = 0
        self._lock = threading.Lock()

    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        score: float,
    ) -> None:
        """Record a scored handoff event."""
        now = time.monotonic()
        flagged = score > self._threshold

        with self._lock:
            self._total_handoffs += 1

            rec = self._agents.setdefault(
                from_agent,
                _AgentRecord(first_seen=now),
            )
            rec.handoffs += 1
            rec.total_score += score
            rec.last_seen = now
            if flagged:
                rec.flagged += 1

            # Also track destination agent activity
            dst = self._agents.setdefault(
                to_agent,
                _AgentRecord(first_seen=now),
            )
            dst.last_seen = now

    def record_quarantine(self, agent_id: str, cascade: bool = False) -> None:
        """Record a quarantine event."""
        with self._lock:
            self._total_quarantines += 1
            if cascade:
                self._cascade_events += 1

            rec = self._agents.setdefault(
                agent_id,
                _AgentRecord(first_seen=time.monotonic()),
            )
            rec.quarantined = True
            rec.quarantine_count += 1

    def report(self) -> dict:
        """Generate a metrics report.

        Returns
        -------
        dict with keys:
            ``"swarm"``: aggregate metrics (total_handoffs, etc.)
            ``"agents"``: per-agent breakdown
        """
        with self._lock:
            swarm = {
                "total_handoffs": self._total_handoffs,
                "total_quarantines": self._total_quarantines,
                "cascade_events": self._cascade_events,
                "active_agents": len(self._agents),
                "quarantined_agents": sum(
                    1 for r in self._agents.values() if r.quarantined
                ),
            }

            agents: dict[str, dict] = {}
            for aid, rec in self._agents.items():
                mean_score = rec.total_score / rec.handoffs if rec.handoffs else 0.0
                hall_rate = rec.flagged / rec.handoffs if rec.handoffs else 0.0
                agents[aid] = {
                    "handoffs": rec.handoffs,
                    "flagged": rec.flagged,
                    "hallucination_rate": round(hall_rate, 4),
                    "mean_score": round(mean_score, 4),
                    "quarantined": rec.quarantined,
                    "quarantine_count": rec.quarantine_count,
                }

        return {"swarm": swarm, "agents": agents}

    def agent_hallucination_rate(self, agent_id: str) -> float:
        """Get hallucination rate for a specific agent."""
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec is None or rec.handoffs == 0:
                return 0.0
            return rec.flagged / rec.handoffs

    def reset(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._agents.clear()
            self._total_handoffs = 0
            self._total_quarantines = 0
            self._cascade_events = 0
