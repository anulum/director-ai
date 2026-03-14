# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Multi-Turn Conversation Session
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
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

    def add_turn(self, prompt: str, response: str, score: float) -> Turn:
        """Append a turn, evicting oldest if at capacity."""
        with self._lock:
            idx = len(self._turns)
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

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)
