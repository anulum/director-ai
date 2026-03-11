# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Async Streaming Kernel for WebSocket Production
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Async streaming oversight for WebSocket production use.

Provides ``AsyncStreamingKernel`` — an async/await version of
``StreamingKernel`` that yields ``TokenEvent`` objects as they arrive.

Usage::

    kernel = AsyncStreamingKernel()

    async for event in kernel.stream_tokens(token_gen, score_fn):
        send_to_websocket(event)
        if event.halted:
            break
"""

from __future__ import annotations

import asyncio
import logging

__all__ = ["AsyncStreamingKernel"]
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable

from .kernel import SafetyKernel
from .streaming import StreamSession, TokenEvent

logger = logging.getLogger("DirectorAI.AsyncStreaming")

# Callback types — sync or async
CoherenceCallback = Callable[[str], float] | Callable[[str], Awaitable[float]]


class AsyncStreamingKernel(SafetyKernel):
    """Async streaming token-by-token safety kernel for WebSocket use.

    Mirrors ``StreamingKernel`` but uses ``async for`` / ``await``.

    Parameters
    ----------
    hard_limit : float — absolute coherence floor (halt if below).
    window_size : int — number of tokens in sliding coherence window.
    window_threshold : float — halt if sliding window average drops below.
    trend_window : int — tokens to check for downward trend.
    trend_threshold : float — halt if coherence drops this much.
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
        token_timeout: float = 0.0,
        total_timeout: float = 0.0,
        halt_mode: str = "hard",
        score_every_n: int = 1,
        adaptive: bool = False,
        max_cadence: int = 8,
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
        if halt_mode not in ("hard", "soft"):
            raise ValueError(f"halt_mode must be 'hard' or 'soft', got {halt_mode!r}")
        if score_every_n < 1:
            raise ValueError(f"score_every_n must be >= 1, got {score_every_n}")
        if max_cadence < 1:
            raise ValueError(f"max_cadence must be >= 1, got {max_cadence}")
        super().__init__(
            hard_limit=hard_limit,
            on_halt=on_halt,
            token_timeout=token_timeout,
            total_timeout=total_timeout,
        )
        self.window_size = window_size
        self.window_threshold = window_threshold
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.soft_limit = soft_limit
        self.halt_mode = halt_mode
        self.score_every_n = score_every_n
        self.adaptive = adaptive
        self.max_cadence = max_cadence

    async def stream_tokens(
        self,
        token_source,
        coherence_callback: CoherenceCallback,
    ) -> AsyncIterator[TokenEvent]:
        """Async generator yielding TokenEvents with oversight checks.

        Parameters
        ----------
        token_source : async iterable of str — token source.
        coherence_callback : (str) -> float OR async (str) -> Awaitable[float].
            Receives the accumulated output so far (not the individual
            token). Called every ``score_every_n`` tokens.

        Yields
        ------
        TokenEvent for each token, with ``halted=True`` on the final if halted.
        """
        window: deque[float] = deque(maxlen=self.window_size)
        coherence_history: list[float] = []
        accumulated_tokens: list[str] = []
        i = 0
        stream_start = time.monotonic()
        cadence = self.score_every_n
        last_score = 0.5
        _soft_halt_pending = False
        _soft_halt_extra_tokens = 0

        async for token in self._iter_tokens(token_source):
            if not isinstance(token, str):
                token = str(token)

            if not self.is_active:
                yield TokenEvent(
                    token=token,
                    index=i,
                    coherence=0.0,
                    timestamp=time.monotonic(),
                    halted=True,
                )
                return

            # Total timeout check
            elapsed = time.monotonic() - stream_start
            if self.total_timeout > 0 and elapsed > self.total_timeout:
                self.emergency_stop()
                yield TokenEvent(
                    token=token,
                    index=i,
                    coherence=0.0,
                    timestamp=time.monotonic(),
                    halted=True,
                )
                return

            if i % cadence == 0:
                token_start = time.monotonic()
                accumulated = "".join(accumulated_tokens) + token
                try:
                    score = await self._call_callback(coherence_callback, accumulated)
                except (TypeError, ValueError, RuntimeError) as exc:
                    logger.warning(
                        "Coherence callback error — using last score: %s", exc
                    )
                    score = last_score
                last_score = score
                if self.adaptive:
                    w_avg = sum(window) / len(window) if window else score
                    if w_avg > self.soft_limit and cadence < self.max_cadence:
                        cadence = min(cadence + 1, self.max_cadence)
                    elif score < self.soft_limit:
                        cadence = 1
            else:
                token_start = time.monotonic()
                score = last_score
            now = time.monotonic()

            # Token timeout check
            if (
                self.token_timeout > 0 and (now - token_start) > self.token_timeout
            ):  # pragma: no cover
                self.emergency_stop()
                yield TokenEvent(
                    token=token,
                    index=i,
                    coherence=score,
                    timestamp=now,
                    halted=True,
                )
                return

            event = TokenEvent(
                token=token,
                index=i,
                coherence=score,
                timestamp=now,
            )

            accumulated_tokens.append(token)
            coherence_history.append(score)
            window.append(score)

            # Soft-halt pending: yield token, check sentence boundary or cap
            if _soft_halt_pending:
                _soft_halt_extra_tokens += 1
                at_cap = _soft_halt_extra_tokens >= self._SOFT_HALT_CAP
                if self._is_sentence_boundary(token) or at_cap:
                    event.halted = True
                    yield event
                    return
                yield event
                i += 1
                continue

            # Hard limit — always immediate halt
            if score < self.hard_limit:
                event.halted = True
                self.emergency_stop()
                yield event
                return

            # Soft zone warning
            if score < self.soft_limit:
                event.warning = True

            # Sliding window average
            halt_reason = ""
            if len(window) >= self.window_size:
                avg = sum(window) / len(window)
                if avg < self.window_threshold:
                    halt_reason = "window_avg"

            # Downward trend (linear regression, matching sync kernel)
            if not halt_reason and len(coherence_history) >= self.trend_window:
                recent = coherence_history[-self.trend_window :]
                n = len(recent)
                x_mean = (n - 1) / 2.0
                y_mean = sum(recent) / n
                num = sum((j - x_mean) * (y - y_mean) for j, y in enumerate(recent))
                den = sum((j - x_mean) ** 2 for j in range(n))
                slope = num / den if den > 1e-12 else 0.0
                if -slope * (n - 1) > self.trend_threshold:
                    halt_reason = "downward_trend"

            if halt_reason:
                if self.halt_mode == "soft":
                    _soft_halt_pending = True
                    if self._is_sentence_boundary(token):
                        event.halted = True
                        yield event
                        return
                    yield event
                    i += 1
                    continue
                event.halted = True
                yield event
                return

            yield event
            i += 1

    @staticmethod
    def _is_sentence_boundary(token: str) -> bool:
        stripped = token.rstrip()
        return bool(stripped) and stripped[-1] in AsyncStreamingKernel._SENTENCE_ENDS

    async def stream_to_session(
        self,
        token_source,
        coherence_callback: CoherenceCallback,
    ) -> StreamSession:
        """Collect all events into a StreamSession (convenience wrapper)."""
        session = StreamSession(start_time=time.monotonic())

        async for event in self.stream_tokens(token_source, coherence_callback):
            session.tokens.append(event.token)
            session.events.append(event)
            session.coherence_history.append(event.coherence)
            if event.warning:
                session.warning_count += 1
            if event.halted:
                session.halted = True
                session.halt_index = event.index
                session.halt_reason = self._halt_reason(event, session)
                break

        session.end_time = time.monotonic()
        if session.halted and self.on_halt:
            self.on_halt(session)
        return session

    def _halt_reason(self, event: TokenEvent, session: StreamSession) -> str:
        """Determine halt reason from context."""
        if event.coherence < self.hard_limit:
            return f"hard_limit ({event.coherence:.4f} < {self.hard_limit})"
        if len(session.coherence_history) >= self.window_size:
            window = session.coherence_history[-self.window_size :]
            avg = sum(window) / len(window)
            if avg < self.window_threshold:  # pragma: no branch
                return f"window_avg ({avg:.4f} < {self.window_threshold})"
        if len(session.coherence_history) >= self.trend_window:
            recent = session.coherence_history[-self.trend_window :]
            n = len(recent)
            x_mean = (n - 1) / 2.0
            y_mean = sum(recent) / n
            num = sum((j - x_mean) * (y - y_mean) for j, y in enumerate(recent))
            den = sum((j - x_mean) ** 2 for j in range(n))
            slope = num / den if den > 1e-12 else 0.0
            drop = -slope * (n - 1)
            if drop > self.trend_threshold:
                return f"downward_trend ({drop:.4f} > {self.trend_threshold})"
        if not self.is_active:  # pragma: no cover
            return "kernel_inactive"
        return "unknown"

    @staticmethod
    async def _iter_tokens(source):
        """Wrap sync or async iterables into async iteration."""
        if hasattr(source, "__aiter__"):
            async for token in source:
                yield token
        else:
            for token in source:
                yield token

    @staticmethod
    async def _call_callback(callback: CoherenceCallback, token: str) -> float:
        """Call sync or async callback."""
        result = callback(token)  # type: ignore[arg-type]
        if asyncio.iscoroutine(result):
            return float(await result)
        return float(result)  # type: ignore[arg-type]
