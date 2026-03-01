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
from typing import TYPE_CHECKING

from .kernel import SafetyKernel
from .types import HaltEvidence

if TYPE_CHECKING:
    from .scorer import CoherenceScorer

logger = logging.getLogger("DirectorAI.Streaming")


@dataclass
class TokenEvent:
    """A single token event in the stream."""

    token: str
    index: int
    coherence: float
    timestamp: float
    halted: bool = False
    warning: bool = False
    evidence: str | None = None
    halt_evidence: HaltEvidence | None = None
    debug_info: dict | None = None


@dataclass
class StreamSession:
    """Tracks state of a streaming oversight session."""

    tokens: list[str] = field(default_factory=list)
    events: list[TokenEvent] = field(default_factory=list)
    coherence_history: list[float] = field(default_factory=list)
    halted: bool = False
    halt_index: int = -1
    halt_reason: str = ""
    halt_evidence: str | None = None
    halt_evidence_structured: HaltEvidence | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    warning_count: int = 0
    debug_log: list[dict] = field(default_factory=list)

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
        return sum(self.coherence_history) / len(self.coherence_history)

    @property
    def min_coherence(self) -> float:
        if not self.coherence_history:
            return 0.0
        return min(self.coherence_history)

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
        on_halt=None,
        soft_limit: float = 0.6,
        streaming_debug: bool = False,
    ) -> None:
        super().__init__(hard_limit=hard_limit, on_halt=on_halt)
        self.window_size = window_size
        self.window_threshold = window_threshold
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.soft_limit = soft_limit
        self.streaming_debug = streaming_debug

    @staticmethod
    def _suggested_action(reason: str) -> str:
        if "hard_limit" in reason:
            return "Reduce generation temperature or add KB facts."
        if "window_avg" in reason:
            return "Context may be drifting from grounded facts."
        if "downward_trend" in reason:
            return "Response quality degrading; rephrase the prompt."
        return "Review the generated output for factual accuracy."

    def stream_tokens(
        self,
        token_generator,
        coherence_callback,
        evidence_callback=None,
        scorer: CoherenceScorer | None = None,
        top_k: int = 3,
    ) -> StreamSession:
        """Process tokens one by one with sliding window oversight.

        Parameters
        ----------
        token_generator : iterable of str — token source.
        coherence_callback : callable(str) -> float — per-token coherence.
        evidence_callback : callable(str) -> str | None — optional, returns
            human-readable evidence snippet explaining the coherence score.
            Called only on halt events to avoid overhead on every token.
        scorer : CoherenceScorer | None — when provided, halt events
            include structured HaltEvidence with top-K contradicting chunks.
        top_k : int — number of evidence chunks to include (default 3).

        Returns
        -------
        StreamSession with full oversight trace.
        """
        session = StreamSession(start_time=time.monotonic())
        window: deque[float] = deque(maxlen=self.window_size)

        def _halt(event: TokenEvent, reason: str) -> None:
            event.halted = True
            session.halted = True
            session.halt_index = event.index
            session.halt_reason = reason
            if evidence_callback:
                ev = evidence_callback("".join(session.tokens))
                event.evidence = ev
                session.halt_evidence = ev
            if scorer is not None:
                accumulated = "".join(session.tokens)
                _, cs = scorer.review("", accumulated)
                chunks = []
                nli_scores = None
                if cs.evidence and cs.evidence.chunks:
                    sorted_chunks = sorted(cs.evidence.chunks, key=lambda c: c.distance)
                    chunks = sorted_chunks[:top_k]
                    if cs.evidence.chunk_scores:
                        nli_scores = cs.evidence.chunk_scores[:top_k]
                structured = HaltEvidence(
                    reason=reason,
                    last_score=cs.score,
                    evidence_chunks=chunks,
                    nli_scores=nli_scores,
                    suggested_action=self._suggested_action(reason),
                )
                event.halt_evidence = structured
                session.halt_evidence_structured = structured

        for i, token in enumerate(token_generator):
            if not self.is_active:
                session.halted = True
                session.halt_index = i
                session.halt_reason = "kernel_inactive"
                break

            score = coherence_callback(token)
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

            if self.streaming_debug:
                w_avg = sum(window) / len(window) if window else 0.0
                recent = session.coherence_history[-self.trend_window :]
                t_drop = (recent[0] - recent[-1]) if len(recent) >= 2 else 0.0
                snap = {
                    "index": i,
                    "coherence": score,
                    "window_avg": round(w_avg, 6),
                    "trend_drop": round(t_drop, 6),
                    "accumulated_tokens": len(session.tokens),
                }
                event.debug_info = snap
                session.debug_log.append(snap)

            if score < self.hard_limit:
                _halt(event, f"hard_limit ({score:.4f} < {self.hard_limit})")
                self.emergency_stop()
                session.events.append(event)
                break

            if score < self.soft_limit:
                event.warning = True
                session.warning_count += 1

            if len(window) >= self.window_size:
                avg = sum(window) / len(window)
                if avg < self.window_threshold:
                    _halt(event, f"window_avg ({avg:.4f} < {self.window_threshold})")
                    session.events.append(event)
                    break

            if len(session.coherence_history) >= self.trend_window:
                recent = session.coherence_history[-self.trend_window :]
                drop = recent[0] - recent[-1]
                if drop > self.trend_threshold:
                    reason = f"downward_trend ({drop:.4f} > {self.trend_threshold})"
                    _halt(event, reason)
                    session.events.append(event)
                    break

            session.events.append(event)

        session.end_time = time.monotonic()
        if session.halted and self.on_halt:
            self.on_halt(session)
        return session

    def stream_output(self, token_generator, coherence_callback) -> str:
        """Backward-compatible: returns string output or interrupt message."""
        session = self.stream_tokens(token_generator, coherence_callback)
        if session.halted:
            return f"[KERNEL INTERRUPT: {session.halt_reason}]"
        return session.output
