# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Cross-turn contradiction tracking for multi-turn conversations.

Scores each new response against every prior response individually
(not concatenated) to build a pairwise contradiction matrix. Reports
which specific turns contradict and how self-consistency evolves.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ContradictionReport", "ContradictionTracker"]


@dataclass
class ContradictionPair:
    """A pair of turns with measured contradiction."""

    turn_a: int
    turn_b: int
    divergence: float


@dataclass
class ContradictionReport:
    """Summary of self-contradiction across a conversation."""

    contradiction_index: float  # 0-1, max pairwise divergence
    worst_pair: ContradictionPair | None
    trend: float  # positive = contradictions increasing over turns
    pair_count: int


class ContradictionTracker:
    """Tracks pairwise contradiction between conversation turns.

    After each turn, call ``update()`` with the new response text and an
    NLI scoring function. The tracker scores the new response against
    each prior response and maintains the contradiction matrix.

    Parameters
    ----------
    max_turns : int
        Maximum number of turns to track (older turns evicted FIFO).
    """

    def __init__(self, max_turns: int = 20):
        self._max_turns = max_turns
        self._responses: list[str] = []
        self._matrix: list[list[float]] = []  # [i][j] = divergence(i, j)

    @property
    def turn_count(self) -> int:
        return len(self._responses)

    def update(
        self,
        response: str,
        score_fn,
    ) -> ContradictionReport:
        """Add a new turn and score it against all prior turns.

        Parameters
        ----------
        response : str
            The new response text.
        score_fn : callable
            ``score_fn(premise, hypothesis) -> float`` returning divergence
            in [0, 1] where 1 = full contradiction.

        Returns
        -------
        ContradictionReport
        """
        n = len(self._responses)
        evict = n >= self._max_turns

        # Compute scores BEFORE any mutation so exception leaves state intact
        score_against = self._responses[1:] if evict else self._responses
        new_row: list[float] = []
        for i in range(len(score_against)):
            div = score_fn(score_against[i], response)
            new_row.append(div)

        # Now mutate (no external calls from here)
        if evict:
            self._responses.pop(0)
            self._matrix.pop(0)
            for row in self._matrix:
                row.pop(0)

        self._responses.append(response)

        for i, row in enumerate(self._matrix):
            row.append(new_row[i])
        self._matrix.append(new_row + [0.0])

        return self._build_report()

    def _build_report(self) -> ContradictionReport:
        n = len(self._responses)
        if n < 2:
            return ContradictionReport(
                contradiction_index=0.0,
                worst_pair=None,
                trend=0.0,
                pair_count=0,
            )

        worst = ContradictionPair(0, 0, 0.0)
        all_divs: list[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                d = self._matrix[i][j]
                all_divs.append(d)
                if d > worst.divergence:
                    worst = ContradictionPair(i, j, d)

        # Trend: compare average divergence of recent pairs vs older pairs
        trend = 0.0
        if len(all_divs) >= 3:
            mid = len(all_divs) // 2
            old_avg = sum(all_divs[:mid]) / mid
            new_avg = sum(all_divs[mid:]) / (len(all_divs) - mid)
            trend = new_avg - old_avg

        return ContradictionReport(
            contradiction_index=worst.divergence,
            worst_pair=worst if worst.divergence > 0.0 else None,
            trend=trend,
            pair_count=len(all_divs),
        )

    def get_report(self) -> ContradictionReport:
        """Get the current contradiction report without adding a turn."""
        return self._build_report()

    def reset(self) -> None:
        """Clear all tracked turns."""
        self._responses.clear()
        self._matrix.clear()
