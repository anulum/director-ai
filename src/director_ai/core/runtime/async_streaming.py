# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Async Streaming Kernel for WebSocket Production

"""Async streaming oversight for WebSocket production use.

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

import inspect
import logging
import math

__all__ = ["AsyncStreamingKernel"]
import time
from collections import deque
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
)
from dataclasses import dataclass

from .kernel import HaltMonitor
from .streaming import StreamSession, TokenEvent, _mean, _trend_drop

logger = logging.getLogger("DirectorAI.AsyncStreaming")

# Callback types — sync or async
CoherenceCallback = Callable[[str], float] | Callable[[str], Awaitable[float]]

# Consecutive coherence-callback errors tolerated before the stream fails closed
# (KIMI3-D4). Within budget a transient error reuses the last score; on the Nth
# consecutive failure the stream halts rather than streaming unscored forever
# (fail-open) on a broken scorer. A successful call resets the counter.
_MAX_CONSECUTIVE_CALLBACK_ERRORS = 3


@dataclass
class _StreamHaltState:
    """Preserve one invocation's stop cause and deferred-halt metadata."""

    reason: str = ""
    index: int = -1
    soft_halted: bool = False


class AsyncStreamingKernel(HaltMonitor):
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
        on_halt: Callable[[StreamSession], None] | None = None,
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
                f"window_threshold must be in [0, 1], got {window_threshold}",
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
            token_timeout=token_timeout,
            total_timeout=total_timeout,
        )
        # The base HaltMonitor invokes its ``on_halt`` with a coherence score;
        # the streaming kernels instead fire on a completed StreamSession, so
        # the session-level callback is kept on a separate attribute.
        self._on_halt_session = on_halt
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
        token_source: AsyncIterable[str] | Iterable[str],
        coherence_callback: CoherenceCallback,
    ) -> AsyncIterator[TokenEvent]:
        """Async generator yielding TokenEvents with oversight checks.

        Parameters
        ----------
        token_source : async iterable of str or iterable of str
            Token source in generation order.
        coherence_callback : callable
            Returns a float directly or through an awaitable such as a
            coroutine, task or future.
            Receives the accumulated output so far (not the individual
            token). Called every ``score_every_n`` tokens.
            NaN and either infinity become zero before history, adaptive
            decisions or halt checks, matching the native streaming kernel.

        Yields
        ------
        TokenEvent
            One event per observed token, with ``halted=True`` on the token
            that stops the stream. A soft window or trend halt waits for a
            sentence boundary or at most 50 additional tokens. A hard-limit
            violation still stops immediately. Every actual halt leaves the
            kernel inactive until explicit reactivation.
        """
        async for event in self._stream_tokens(
            token_source, coherence_callback, _StreamHaltState()
        ):
            yield event

    async def _stream_tokens(
        self,
        token_source: AsyncIterable[str] | Iterable[str],
        coherence_callback: CoherenceCallback,
        halt_state: _StreamHaltState,
    ) -> AsyncIterator[TokenEvent]:
        """Run the shared event engine with invocation-local deferred halt state.

        Parameters
        ----------
        token_source : async iterable of str or iterable of str
            Actual upstream token source.
        coherence_callback : callable
            Host scorer invoked with accumulated output.
        halt_state : _StreamHaltState
            Actual stop cause and soft-halt state shared with the collector.
            Each public invocation creates a separate state object.

        Yields
        ------
        TokenEvent
            Events from the same scoring and halt path for both public APIs.
        """
        window: deque[float] = deque(maxlen=self.window_size)
        coherence_history: list[float] = []
        accumulated_tokens: list[str] = []
        i = 0
        stream_start = time.monotonic()
        cadence = self.score_every_n
        last_score = 0.5
        consecutive_callback_errors = 0
        _soft_halt_pending = False
        _soft_halt_extra_tokens = 0

        async for token in self._iter_tokens(token_source):
            if not isinstance(token, str):
                token = str(token)

            if not self.is_active:
                halt_state.reason = "kernel_inactive"
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
                halt_state.reason = "total_timeout"
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
                    consecutive_callback_errors = 0
                except (
                    TypeError,
                    ValueError,
                    RuntimeError,
                    TimeoutError,
                    OSError,
                ) as exc:
                    consecutive_callback_errors += 1
                    if consecutive_callback_errors >= _MAX_CONSECUTIVE_CALLBACK_ERRORS:
                        # Fail closed (KIMI3-D4): a scorer that keeps erroring
                        # must not stream unscored forever on a stale last score.
                        logger.error(
                            "Coherence callback failed %d times consecutively — "
                            "failing closed (halting stream): %s",
                            consecutive_callback_errors,
                            exc,
                        )
                        halt_state.reason = "callback_error"
                        self.emergency_stop()
                        yield TokenEvent(
                            token=token,
                            index=i,
                            coherence=0.0,
                            timestamp=time.monotonic(),
                            halted=True,
                        )
                        return
                    logger.warning(
                        "Coherence callback error %d/%d — using last score: %s",
                        consecutive_callback_errors,
                        _MAX_CONSECUTIVE_CALLBACK_ERRORS,
                        exc,
                    )
                    score = last_score
                score = score if math.isfinite(score) else 0.0
                last_score = score
                if self.adaptive:
                    w_avg = _mean(window) if window else score
                    if w_avg > self.soft_limit and cadence < self.max_cadence:
                        cadence = min(cadence + 1, self.max_cadence)
                    elif score < self.soft_limit:
                        cadence = 1
            else:
                token_start = time.monotonic()
                score = last_score
            now = time.monotonic()

            # Token timeout check
            if self.token_timeout > 0 and (now - token_start) > self.token_timeout:
                halt_state.reason = "token_timeout"
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

            if score < self.hard_limit:
                halt_state.reason = ""
                event.halted = True
                self.emergency_stop()
                yield event
                return

            # Soft-halt pending: yield token, check sentence boundary or cap
            if _soft_halt_pending:
                _soft_halt_extra_tokens += 1
                at_cap = _soft_halt_extra_tokens >= self._SOFT_HALT_CAP
                if self._is_sentence_boundary(token) or at_cap:
                    event.halted = True
                    halt_state.soft_halted = True
                    self.emergency_stop()
                    yield event
                    return
                yield event
                i += 1
                continue

            # Sliding window average
            halt_reason = ""
            if len(window) >= self.window_size:
                avg = _mean(window)
                if avg < self.window_threshold:
                    halt_reason = f"window_avg ({avg:.4f} < {self.window_threshold})"

            # Downward trend (linear regression, matching sync kernel)
            if not halt_reason and len(coherence_history) >= self.trend_window:
                recent = coherence_history[-self.trend_window :]
                drop = _trend_drop(recent)
                if drop > self.trend_threshold:
                    halt_reason = (
                        f"downward_trend ({drop:.4f} > {self.trend_threshold})"
                    )

            if halt_reason:
                if self.halt_mode == "soft":
                    _soft_halt_pending = True
                    halt_state.reason = halt_reason
                    halt_state.index = event.index
                    if self._is_sentence_boundary(token):
                        event.halted = True
                        halt_state.soft_halted = True
                        self.emergency_stop()
                        yield event
                        return
                    yield event
                    i += 1
                    continue
                event.halted = True
                self.emergency_stop()
                yield event
                return

            if score < self.soft_limit:
                event.warning = True
            yield event
            i += 1

    @staticmethod
    def _is_sentence_boundary(token: str) -> bool:
        stripped = token.rstrip()
        return bool(stripped) and stripped[-1] in AsyncStreamingKernel._SENTENCE_ENDS

    async def stream_to_session(
        self,
        token_source: AsyncIterable[str] | Iterable[str],
        coherence_callback: CoherenceCallback,
    ) -> StreamSession:
        """Collect events and preserve the actual deferred halt decision.

        Parameters
        ----------
        token_source : async iterable of str or iterable of str
            Upstream tokens passed through the shared event engine.
        coherence_callback : callable
            Host scorer returning a score directly or through an awaitable.

        Returns
        -------
        StreamSession
            Observed tokens, scores and warnings. Completed soft halts retain
            their initial window or trend reason and trigger index, and their
            output includes the delivered sentence. Hard halts exclude the
            violating token. The configured session callback runs once after
            a halted session has been collected. Deadline and exhausted
            callback failures retain their operational stop causes instead
            of being inferred from the fallback coherence value.
        """
        session = StreamSession(start_time=time.monotonic())
        halt_state = _StreamHaltState()

        async for event in self._stream_tokens(
            token_source, coherence_callback, halt_state
        ):
            session.tokens.append(event.token)
            session.events.append(event)
            session.coherence_history.append(event.coherence)
            if event.warning:
                session.warning_count += 1
            if event.halted:
                session.halted = True
                session.soft_halted = halt_state.soft_halted
                session.halt_index = (
                    halt_state.index if halt_state.soft_halted else event.index
                )
                session.halt_reason = halt_state.reason or self._halt_reason(
                    event, session
                )
                break

        session.end_time = time.monotonic()
        if session.halted and self._on_halt_session:
            self._on_halt_session(session)
        return session

    def _halt_reason(self, event: TokenEvent, session: StreamSession) -> str:
        """Determine halt reason from context."""
        if event.coherence < self.hard_limit:
            return f"hard_limit ({event.coherence:.4f} < {self.hard_limit})"
        if len(session.coherence_history) >= self.window_size:
            window = session.coherence_history[-self.window_size :]
            avg = _mean(window)
            if avg < self.window_threshold:  # pragma: no branch
                return f"window_avg ({avg:.4f} < {self.window_threshold})"
        if len(session.coherence_history) >= self.trend_window:
            recent = session.coherence_history[-self.trend_window :]
            drop = _trend_drop(recent)
            if drop > self.trend_threshold:
                return f"downward_trend ({drop:.4f} > {self.trend_threshold})"
        if not self.is_active:
            return "kernel_inactive"
        if session.soft_halted:
            return "soft_halt"
        return "halt_condition_not_identified"

    @staticmethod
    async def _iter_tokens(
        source: AsyncIterable[str] | Iterable[str],
    ) -> AsyncIterator[str]:
        """Wrap sync or async iterables into async iteration."""
        if hasattr(source, "__aiter__"):
            async for token in source:
                yield token
        else:
            for token in source:
                yield token

    @staticmethod
    async def _call_callback(callback: CoherenceCallback, token: str) -> float:
        """Resolve one score from the callback before evaluating the token."""
        result: float | Awaitable[float] = callback(token)
        if inspect.isawaitable(result):
            return float(await result)
        return float(result)
