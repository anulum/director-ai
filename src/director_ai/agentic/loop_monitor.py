# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Agentic loop monitor — detect runaway, circular, and drifting agents.

Monitors AI agent execution loops for:
- Circular tool calls (same tool + similar args repeated)
- Goal drift (current action diverges from original objective)
- Resource budget exhaustion (tokens, API calls, wall time)
- Reasoning degradation (coherence dropping across steps)

Usage::

    monitor = LoopMonitor(goal="Find the quarterly revenue for Q3 2025")
    for step in agent_loop:
        verdict = monitor.check_step(
            action=step.tool_name,
            args=step.tool_args,
            result=step.result,
        )
        if verdict.should_halt:
            break
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field

__all__ = ["LoopMonitor", "StepVerdict", "LoopStatus"]


@dataclass
class StepVerdict:
    """Verdict for a single agentic step."""

    step_number: int
    should_halt: bool
    should_warn: bool
    reasons: list[str] = field(default_factory=list)
    goal_drift_score: float = 0.0  # 0 = aligned, 1 = completely off-goal
    budget_remaining_pct: float = 1.0


@dataclass
class LoopStatus:
    """Overall status of the monitored agent loop."""

    total_steps: int
    circular_detections: int
    goal_drift_alerts: int
    budget_exhausted: bool
    halted: bool
    halt_reason: str = ""
    step_verdicts: list[StepVerdict] = field(default_factory=list)


class LoopMonitor:
    """Monitor an AI agent loop for safety issues.

    Parameters
    ----------
    goal : str
        The original objective the agent should be pursuing.
    max_steps : int
        Maximum allowed steps before forced halt.
    max_tokens : int
        Token budget (cumulative input + output tokens).
    max_wall_seconds : float
        Wall clock time limit in seconds.
    circular_threshold : int
        Number of times the same action+args hash can repeat before flagging.
    goal_drift_threshold : float
        Goal drift score (0-1) above which to warn. Default 0.6.
    goal_drift_scorer : callable | None
        Function(goal: str, action_description: str) -> float (0-1).
        If None, a simple Jaccard overlap heuristic is used.
    """

    def __init__(
        self,
        goal: str,
        max_steps: int = 50,
        max_tokens: int = 500_000,
        max_wall_seconds: float = 300.0,
        circular_threshold: int = 3,
        goal_drift_threshold: float = 0.6,
        goal_drift_scorer=None,
    ):
        self._goal = goal
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._max_wall_seconds = max_wall_seconds
        self._circular_threshold = circular_threshold
        self._goal_drift_threshold = goal_drift_threshold
        self._drift_scorer = goal_drift_scorer or self._jaccard_drift

        self._steps: list[StepVerdict] = []
        self._action_hashes: Counter[str] = Counter()
        self._tokens_used = 0
        self._start_time = time.monotonic()
        self._halted = False
        self._halt_reason = ""
        self._circular_count = 0
        self._drift_count = 0

    def check_step(
        self,
        action: str,
        args: str = "",
        result: str = "",
        tokens: int = 0,
    ) -> StepVerdict:
        """Evaluate a single agent step.

        Parameters
        ----------
        action : str
            Tool or function name being called.
        args : str
            Serialized arguments (for circular detection).
        result : str
            Tool output (for coherence tracking).
        tokens : int
            Tokens consumed by this step.

        Returns
        -------
        StepVerdict
            Whether to continue, warn, or halt.
        """
        step_num = len(self._steps) + 1
        reasons: list[str] = []
        should_halt = False
        should_warn = False

        # 1. Step count limit
        if step_num > self._max_steps:
            should_halt = True
            reasons.append(f"Step limit exceeded ({step_num}/{self._max_steps})")

        # 2. Token budget
        self._tokens_used += tokens
        budget_pct = max(0.0, 1.0 - self._tokens_used / self._max_tokens)
        if self._tokens_used > self._max_tokens:
            should_halt = True
            reasons.append(
                f"Token budget exhausted ({self._tokens_used:,}/{self._max_tokens:,})"
            )
        elif budget_pct < 0.1:
            should_warn = True
            reasons.append(f"Token budget low ({budget_pct:.0%} remaining)")

        # 3. Wall time
        elapsed = time.monotonic() - self._start_time
        if elapsed > self._max_wall_seconds:
            should_halt = True
            reasons.append(
                f"Wall time exceeded ({elapsed:.0f}s/{self._max_wall_seconds:.0f}s)"
            )

        # 4. Circular detection
        action_hash = self._hash_action(action, args)
        self._action_hashes[action_hash] += 1
        repeats = self._action_hashes[action_hash]
        if repeats >= self._circular_threshold:
            should_warn = True
            self._circular_count += 1
            reasons.append(f"Circular call detected: {action} repeated {repeats}x")
        if repeats >= self._circular_threshold * 2:
            should_halt = True
            reasons.append(f"Severe circular loop: {action} repeated {repeats}x")

        # 5. Goal drift
        action_desc = f"{action}({args})" if args else action
        drift = self._drift_scorer(self._goal, action_desc)
        if drift > self._goal_drift_threshold:
            should_warn = True
            self._drift_count += 1
            reasons.append(
                f"Goal drift detected: {drift:.2f} "
                f"(threshold {self._goal_drift_threshold})"
            )
        if drift > 0.9:
            should_halt = True
            reasons.append(f"Severe goal drift: {drift:.2f}")

        verdict = StepVerdict(
            step_number=step_num,
            should_halt=should_halt,
            should_warn=should_warn,
            reasons=reasons,
            goal_drift_score=drift,
            budget_remaining_pct=budget_pct,
        )
        self._steps.append(verdict)

        if should_halt and not self._halted:
            self._halted = True
            self._halt_reason = "; ".join(reasons)

        return verdict

    def status(self) -> LoopStatus:
        """Return the overall loop status."""
        return LoopStatus(
            total_steps=len(self._steps),
            circular_detections=self._circular_count,
            goal_drift_alerts=self._drift_count,
            budget_exhausted=self._tokens_used >= self._max_tokens,
            halted=self._halted,
            halt_reason=self._halt_reason,
            step_verdicts=list(self._steps),
        )

    @staticmethod
    def _hash_action(action: str, args: str) -> str:
        key = f"{action}::{args}"
        return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]

    @staticmethod
    def _jaccard_drift(goal: str, action_desc: str) -> float:
        """Simple Jaccard-distance heuristic for goal drift.

        Returns 0.0 (perfectly aligned) to 1.0 (completely off-topic).
        For production use, replace with NLI-based scorer.
        """
        goal_words = set(goal.lower().split())
        action_words = set(action_desc.lower().split())
        if not goal_words or not action_words:
            return 1.0
        intersection = goal_words & action_words
        union = goal_words | action_words
        similarity = len(intersection) / len(union) if union else 0.0
        return 1.0 - similarity
