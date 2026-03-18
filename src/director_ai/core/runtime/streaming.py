# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Streaming Token-by-Token Kernel Oversight

"""Streaming oversight for token-by-token coherence monitoring.

Provides ``StreamingKernel`` which extends ``HaltMonitor`` with:
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

from ..otel import trace_streaming
from ..types import HaltEvidence
from .kernel import HaltMonitor

__all__ = ["StreamSession", "StreamingKernel", "TokenEvent"]

if TYPE_CHECKING:  # pragma: no cover
    from ..scoring.scorer import CoherenceScorer

logger = logging.getLogger("DirectorAI.Streaming")


def _trend_drop(values: list[float] | deque) -> float:
    """Linear regression slope drop over a window of coherence scores.

    Returns the projected drop magnitude: -slope * (n - 1).
    Positive values indicate downward trend.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den > 1e-12 else 0.0
    return -slope * (n - 1)


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
    soft_halted: bool = False
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
        if self.soft_halted:
            return "".join(self.tokens)
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


class StreamingKernel(HaltMonitor):
    """Streaming token-by-token safety kernel with sliding window oversight.

    Extends ``HaltMonitor`` with token-level monitoring and a sliding
    window coherence check that can catch gradual degradation.

    Parameters
    ----------
    hard_limit : float â€” absolute coherence floor (halt if below).
    window_size : int â€” number of tokens in sliding coherence window.
    window_threshold : float â€” halt if sliding window average drops below this.
    trend_window : int â€” tokens to check for downward trend.
    trend_threshold : float â€” halt if coherence drops this much over trend window.

    """

    _SENTENCE_ENDS = frozenset(".!?")
    _SOFT_HALT_CAP = 50

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
        halt_mode: str = "hard",
        score_every_n: int = 1,
        adaptive: bool = False,
        max_cadence: int = 8,
    ) -> None:
        super().__init__(hard_limit=hard_limit, on_halt=on_halt)
        if halt_mode not in ("hard", "soft"):
            raise ValueError(f"halt_mode must be 'hard' or 'soft', got {halt_mode!r}")
        if score_every_n < 1:
            raise ValueError(f"score_every_n must be >= 1, got {score_every_n}")
        if max_cadence < 1:
            raise ValueError(f"max_cadence must be >= 1, got {max_cadence}")
        self.window_size = window_size
        self.window_threshold = window_threshold
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.soft_limit = soft_limit
        self.streaming_debug = streaming_debug
        self.halt_mode = halt_mode
        self.score_every_n = score_every_n
        self.adaptive = adaptive
        self.max_cadence = max_cadence
        self._window: deque[float] = deque(maxlen=window_size)
        self._history: list[float] = []

    def check_halt(self, score: float) -> bool:
        """Evaluate halt conditions for a single score update.

        Maintains internal sliding window and trend history.
        Returns True if any halt condition is met.
        """
        self._window.append(score)
        self._history.append(score)
        if score < self.hard_limit:
            return True
        if len(self._window) >= self.window_size:
            avg = sum(self._window) / len(self._window)
            if avg < self.window_threshold:
                return True
        if len(self._history) >= self.trend_window:
            recent = list(self._history)[-self.trend_window :]
            if _trend_drop(recent) > self.trend_threshold:
                return True
        return False

    def reset_state(self) -> None:
        """Clear internal window/trend state and re-arm kernel for a new stream."""
        self._window.clear()
        self._history.clear()
        self.reactivate()

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
        prompt: str = "",
    ) -> StreamSession:
        """Process tokens one by one with sliding window oversight.

        Parameters
        ----------
        token_generator : iterable of str â€” token source.
        coherence_callback : callable(str) -> float â€” receives the
            accumulated output so far (not the individual token) and
            returns a coherence score in [0, 1]. Called every
            ``score_every_n`` tokens; cadence adapts when ``adaptive=True``.
        evidence_callback : callable(str) -> str | None â€” optional, returns
            human-readable evidence snippet explaining the coherence score.
            Called only on halt events to avoid overhead on every token.
        scorer : CoherenceScorer | None â€” when provided, halt events
            include structured HaltEvidence with top-K contradicting chunks.
        top_k : int â€” number of evidence chunks to include (default 3).
        prompt : str â€” original user prompt, passed to scorer.review() for
            KB/RAG context in halt evidence.

        Returns
        -------
        StreamSession with full oversight trace.

        """
        session = StreamSession(start_time=time.monotonic())
        window: deque[float] = deque(maxlen=self.window_size)
        _soft_halt_pending = False
        _soft_halt_reason = ""
        _soft_halt_extra_tokens = 0

        def _finalize_halt(event: TokenEvent, reason: str) -> None:
            event.halted = True
            session.halted = True
            if session.halt_index < 0:
                session.halt_index = event.index
            session.halt_reason = reason
            if evidence_callback:
                ev = evidence_callback("".join(session.tokens))
                event.evidence = ev
                session.halt_evidence = ev
            if scorer is not None:
                accumulated = "".join(session.tokens)
                _, cs = scorer.review(prompt, accumulated)
                chunks = []
                nli_scores = None
                if cs.evidence and cs.evidence.chunks:
                    sorted_chunks = sorted(cs.evidence.chunks, key=lambda c: c.distance)
                    chunks = sorted_chunks[:top_k]
                    if cs.evidence.chunk_scores:  # pragma: no branch
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

        def _is_sentence_boundary(tok: str) -> bool:
            stripped = tok.rstrip()
            return bool(stripped) and stripped[-1] in self._SENTENCE_ENDS

        cadence = self.score_every_n
        last_score = 0.5

        for i, token in enumerate(token_generator):
            if not self.is_active:
                session.halted = True
                session.halt_index = i
                session.halt_reason = "kernel_inactive"
                break

            if i % cadence == 0:
                accumulated = "".join(session.tokens) + token
                try:
                    score = coherence_callback(accumulated)
                except (TimeoutError, OSError):
                    session.halted = True
                    session.halt_index = i
                    session.halt_reason = "callback_timeout"
                    break
                last_score = score
                if self.adaptive:
                    w_avg = sum(window) / len(window) if window else score
                    if w_avg > self.soft_limit and cadence < self.max_cadence:
                        cadence = min(cadence + 1, self.max_cadence)
                    elif score < self.soft_limit:
                        cadence = 1
            else:
                score = last_score
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
                t_drop = _trend_drop(recent)
                snap = {
                    "index": i,
                    "coherence": score,
                    "window_avg": round(w_avg, 6),
                    "trend_drop": round(t_drop, 6),
                    "accumulated_tokens": len(session.tokens),
                }
                event.debug_info = snap
                session.debug_log.append(snap)

            # If soft-halt pending, check for sentence boundary or cap
            if _soft_halt_pending:
                _soft_halt_extra_tokens += 1
                session.events.append(event)
                at_cap = _soft_halt_extra_tokens >= self._SOFT_HALT_CAP
                if _is_sentence_boundary(token) or at_cap:
                    session.soft_halted = True
                    _finalize_halt(event, _soft_halt_reason)
                    break
                continue

            halt_reason = ""
            if score < self.hard_limit:
                halt_reason = f"hard_limit ({score:.4f} < {self.hard_limit})"
            elif len(window) >= self.window_size:
                avg = sum(window) / len(window)
                if avg < self.window_threshold:
                    halt_reason = f"window_avg ({avg:.4f} < {self.window_threshold})"
            if not halt_reason and len(session.coherence_history) >= self.trend_window:
                recent = session.coherence_history[-self.trend_window :]
                drop = _trend_drop(recent)
                if drop > self.trend_threshold:
                    halt_reason = (
                        f"downward_trend ({drop:.4f} > {self.trend_threshold})"
                    )

            if halt_reason:
                if self.halt_mode == "soft" and "hard_limit" not in halt_reason:
                    _soft_halt_pending = True
                    _soft_halt_reason = halt_reason
                    session.halt_index = event.index
                    session.events.append(event)
                    if _is_sentence_boundary(token):
                        session.soft_halted = True
                        _finalize_halt(event, halt_reason)
                        break
                    continue
                # Hard halt
                _finalize_halt(event, halt_reason)
                if "hard_limit" in halt_reason:
                    self.emergency_stop()
                session.events.append(event)
                break

            if score < self.soft_limit:
                event.warning = True
                session.warning_count += 1

            session.events.append(event)

        session.end_time = time.monotonic()
        if session.halted and self.on_halt:
            self.on_halt(session)

        with trace_streaming() as span:
            span.set_attribute("stream.halted", session.halted)
            span.set_attribute("stream.soft_halted", session.soft_halted)
            span.set_attribute("stream.halt_reason", session.halt_reason)
            span.set_attribute("stream.token_count", session.token_count)
            span.set_attribute("stream.warning_count", session.warning_count)
            if session.coherence_history:
                span.set_attribute("stream.avg_coherence", session.avg_coherence)
        return session

    def stream_output(self, token_generator, coherence_callback) -> str:
        """Backward-compatible: returns string output or interrupt message."""
        session = self.stream_tokens(token_generator, coherence_callback)
        if session.halted:
            return f"[KERNEL INTERRUPT: {session.halt_reason}]"
        return session.output
