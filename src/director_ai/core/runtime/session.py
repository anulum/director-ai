# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Multi-Turn Conversation Session

"""Track multi-turn conversations for cross-turn coherence scoring.

Usage::

    session = ConversationSession()
    session.add_turn("What is AI?", "AI is ...", 0.85)
    context = session.context_text  # prior responses as NLI premise
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from .contradiction_tracker import ContradictionReport, ContradictionTracker

__all__ = ["ConversationSession", "Turn"]


@dataclass
class Turn:
    """A single conversation turn."""

    prompt: str
    response: str
    score: float
    turn_index: int


class ConversationSession:
    """Thread-safe multi-turn conversation tracker.

    Parameters
    ----------
    max_turns : int — maximum turns retained (FIFO eviction).
    session_id : str | None — auto-generated UUID if not provided.

    """

    def __init__(self, max_turns: int = 20, session_id: str | None = None) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self.session_id = session_id or str(uuid.uuid4())
        self.max_turns = max_turns
        self._turns: list[Turn] = []
        self._lock = threading.Lock()
        self._turn_counter = 0
        self._contradiction_tracker = ContradictionTracker(max_turns=max_turns)

    def add_turn(self, prompt: str, response: str, score: float) -> Turn:
        """Append a turn, evicting oldest if at capacity."""
        with self._lock:
            idx = self._turn_counter
            self._turn_counter += 1
            turn = Turn(prompt=prompt, response=response, score=score, turn_index=idx)
            self._turns.append(turn)
            if len(self._turns) > self.max_turns:
                self._turns.pop(0)
            return turn

    @property
    def turns(self) -> list[Turn]:
        with self._lock:
            return list(self._turns)

    @property
    def context_text(self) -> str:
        """Concatenated prior responses for NLI premise construction."""
        with self._lock:
            return " ".join(t.response for t in self._turns)

    def update_contradictions(self, response: str, score_fn) -> ContradictionReport:
        """Score the new response against all prior responses for contradictions.

        Parameters
        ----------
        response : str
            The new response text.
        score_fn : callable
            ``score_fn(premise, hypothesis) -> float`` returning divergence.
        """
        with self._lock:
            return self._contradiction_tracker.update(response, score_fn)

    def get_contradiction_report(self) -> ContradictionReport:
        """Get the current contradiction report without adding a turn."""
        with self._lock:
            return self._contradiction_tracker.get_report()

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)
