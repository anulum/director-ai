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

        Yields
        ------
        TokenEvent for each token, with ``halted=True`` on the final if halted.
        """
        window: deque[float] = deque(maxlen=self.window_size)
        coherence_history: list[float] = []
        i = 0
        stream_start = time.monotonic()

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

            token_start = time.monotonic()
            try:
                score = await self._call_callback(coherence_callback, token)
            except (TypeError, ValueError, RuntimeError) as exc:
                logger.error("Coherence callback raised %s — treating as score=0", exc)
                score = 0.0
            now = time.monotonic()

            # Token timeout check
            if self.token_timeout > 0 and (now - token_start) > self.token_timeout:
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

            coherence_history.append(score)
            window.append(score)

            # Hard limit
            if score < self.hard_limit:
                event.halted = True
                self.emergency_stop()
                yield event
                return

            # Soft zone warning
            if score < self.soft_limit:
                event.warning = True

            # Sliding window average
            if len(window) >= self.window_size:
                avg = sum(window) / len(window)
                if avg < self.window_threshold:
                    event.halted = True
                    yield event
                    return

            # Downward trend
            if len(coherence_history) >= self.trend_window:
                recent = coherence_history[-self.trend_window :]
                drop = recent[0] - recent[-1]
                if drop > self.trend_threshold:
                    event.halted = True
                    yield event
                    return

            yield event
            i += 1

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
            if avg < self.window_threshold:
                return f"window_avg ({avg:.4f} < {self.window_threshold})"
        if len(session.coherence_history) >= self.trend_window:
            recent = session.coherence_history[-self.trend_window :]
            drop = recent[0] - recent[-1]
            if drop > self.trend_threshold:
                return f"downward_trend ({drop:.4f} > {self.trend_threshold})"
        if not self.is_active:
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
