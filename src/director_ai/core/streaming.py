# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Streaming Token-by-Token Kernel Oversight
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Streaming oversight for token-by-token coherence monitoring.

Provides ``StreamingKernel`` which extends ``SafetyKernel`` with:
- Async token processing via generator protocol
- Sliding window coherence checks
- Real-time halt with partial output recovery
- Token-level divergence tracking
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

from .kernel import SafetyKernel

logger = logging.getLogger("DirectorAI.Streaming")


@dataclass
class TokenEvent:
    """A single token event in the stream."""

    token: str
    index: int
    coherence: float
    timestamp: float
    halted: bool = False


@dataclass
class StreamSession:
    """Tracks state of a streaming oversight session."""

    tokens: list[str] = field(default_factory=list)
    events: list[TokenEvent] = field(default_factory=list)
    coherence_history: list[float] = field(default_factory=list)
    halted: bool = False
    halt_index: int = -1
    halt_reason: str = ""
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def output(self) -> str:
        if self.halted and self.halt_index >= 0:
            return "".join(self.tokens[: self.halt_index])
        return "".join(self.tokens)

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def avg_coherence(self) -> float:
        if not self.coherence_history:
            return 0.0
        return max(
            0.0, min(1.0, sum(self.coherence_history) / len(self.coherence_history))
        )

    @property
    def min_coherence(self) -> float:
        if not self.coherence_history:
            return 0.0
        return max(0.0, min(1.0, min(self.coherence_history)))

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class StreamingKernel(SafetyKernel):
    """Streaming token-by-token safety kernel with sliding window oversight.

    Extends ``SafetyKernel`` with token-level monitoring and a sliding
    window coherence check that can catch gradual degradation.

    Parameters
    ----------
    hard_limit : float — absolute coherence floor (halt if below).
    window_size : int — number of tokens in sliding coherence window.
    window_threshold : float — halt if sliding window average drops below this.
    trend_window : int — tokens to check for downward trend.
    trend_threshold : float — halt if coherence drops this much over trend window.
    """

    def __init__(
        self,
        hard_limit: float = 0.5,
        window_size: int = 10,
        window_threshold: float = 0.55,
        trend_window: int = 5,
        trend_threshold: float = 0.15,
    ) -> None:
        if not (0.0 <= hard_limit <= 1.0):
            raise ValueError(f"hard_limit must be in [0, 1], got {hard_limit}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if not (0.0 <= window_threshold <= 1.0):
            raise ValueError(
                f"window_threshold must be in [0, 1], got {window_threshold}"
            )
        if trend_window < 2:
            raise ValueError(f"trend_window must be >= 2, got {trend_window}")
        if trend_threshold <= 0:
            raise ValueError(f"trend_threshold must be > 0, got {trend_threshold}")
        super().__init__(hard_limit=hard_limit)
        self.window_size = window_size
        self.window_threshold = window_threshold
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold

    def stream_tokens(
        self,
        token_generator,
        coherence_callback,
    ) -> StreamSession:
        """Process tokens one by one with sliding window oversight.

        Parameters
        ----------
        token_generator : iterable of str — token source.
        coherence_callback : callable(str) -> float — per-token coherence.

        Returns
        -------
        StreamSession with full oversight trace.
        """
        session = StreamSession(start_time=time.monotonic())
        window: deque[float] = deque(maxlen=self.window_size)

        for i, token in enumerate(token_generator):
            if not self.is_active:
                session.halted = True
                session.halt_index = i
                session.halt_reason = "kernel_inactive"
                break

            if not isinstance(token, str):
                token = str(token)

            try:
                score = float(coherence_callback(token))
            except Exception:
                logger.error("Coherence callback raised — treating as score=0")
                score = 0.0

            now = time.monotonic()

            event = TokenEvent(
                token=token,
                index=i,
                coherence=score,
                timestamp=now,
            )

            session.tokens.append(token)
            session.coherence_history.append(score)
            window.append(score)

            # Hard limit
            if score < self.hard_limit:
                event.halted = True
                session.halted = True
                session.halt_index = i
                session.halt_reason = f"hard_limit ({score:.4f} < {self.hard_limit})"
                self.emergency_stop()
                session.events.append(event)
                break

            # Sliding window average
            if len(window) >= self.window_size:
                avg = sum(window) / len(window)
                if avg < self.window_threshold:
                    event.halted = True
                    session.halted = True
                    session.halt_index = i
                    session.halt_reason = (
                        f"window_avg ({avg:.4f} < {self.window_threshold})"
                    )
                    session.events.append(event)
                    break

            # Downward trend
            if len(session.coherence_history) >= self.trend_window:
                recent = session.coherence_history[-self.trend_window :]
                drop = recent[0] - recent[-1]
                if drop > self.trend_threshold:
                    event.halted = True
                    session.halted = True
                    session.halt_index = i
                    session.halt_reason = (
                        f"downward_trend ({drop:.4f} > {self.trend_threshold})"
                    )
                    session.events.append(event)
                    break

            session.events.append(event)

        session.end_time = time.monotonic()
        return session

    def stream_output(self, token_generator, coherence_callback) -> str:
        """Backward-compatible: returns string output or interrupt message."""
        session = self.stream_tokens(token_generator, coherence_callback)
        if session.halted:
            return f"[KERNEL INTERRUPT: {session.halt_reason}]"
        return session.output
