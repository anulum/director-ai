# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Async Streaming Halt Reason Tests
"""Multi-angle tests for AsyncStreamingKernel halt reason branches.

Covers: hard_limit halt, window_avg halt, downward_trend halt,
unknown halt fallback, parametrised halt conditions, halt reason
format, session state after halt, pipeline integration with
coherence scores, and performance documentation.
"""

from __future__ import annotations

import asyncio

import pytest

from director_ai.core.async_streaming import AsyncStreamingKernel


async def _async_tokens(tokens):
    for t in tokens:
        yield t


def _run(coro):
    return asyncio.run(coro)


# ── Halt reason branches ─────────────────────────────────────────


class TestHaltReasonBranches:
    """Each halt condition must produce a specific halt_reason string."""

    def test_halt_reason_hard_limit(self):
        kernel = AsyncStreamingKernel(hard_limit=0.5, window_size=100)

        async def run():
            return await kernel.stream_to_session(_async_tokens(["a"]), lambda t: 0.3)

        session = _run(run())
        assert "hard_limit" in session.halt_reason

    def test_halt_reason_window_avg(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=2,
            window_threshold=0.7,
        )
        scores = iter([0.5, 0.5, 0.5])

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b", "c"]),
                lambda t: next(scores, 0.5),
            )

        session = _run(run())
        assert "window_avg" in session.halt_reason

    def test_halt_reason_downward_trend(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=100,
            trend_window=3,
            trend_threshold=0.1,
        )
        scores = iter([0.9, 0.8, 0.7, 0.5])

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b", "c", "d"]),
                lambda t: next(scores, 0.5),
            )

        session = _run(run())
        assert "downward_trend" in session.halt_reason

    def test_halt_reason_unknown_fallback(self):
        kernel = AsyncStreamingKernel()
        reason = kernel._halt_reason(
            type("E", (), {"coherence": 0.9})(),
            type("S", (), {"coherence_history": [], "soft_halted": False})(),
        )
        assert reason == "halt_condition_not_identified"

    @pytest.mark.parametrize(
        "score,expected_reason",
        [
            (0.01, "hard_limit"),
            (0.05, "hard_limit"),
        ],
    )
    def test_parametrised_hard_limit(self, score, expected_reason):
        kernel = AsyncStreamingKernel(hard_limit=0.5)

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a"]),
                lambda t: score,
            )

        session = _run(run())
        assert expected_reason in session.halt_reason


# ── Session state after halt ─────────────────────────────────────


class TestSessionStateAfterHalt:
    """Session must have complete state info after any halt."""

    def test_halted_session_has_reason(self):
        kernel = AsyncStreamingKernel(hard_limit=0.5)

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b"]),
                lambda t: 0.3,
            )

        session = _run(run())
        assert session.halted
        assert isinstance(session.halt_reason, str)
        assert len(session.halt_reason) > 0

    def test_halted_session_has_tokens(self):
        kernel = AsyncStreamingKernel(hard_limit=0.5)

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b", "c"]),
                lambda t: 0.3,
            )

        session = _run(run())
        assert session.halted
        assert len(session.tokens) >= 1

    def test_non_halted_session(self):
        kernel = AsyncStreamingKernel(hard_limit=0.1)

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b"]),
                lambda t: 0.9,
            )

        session = _run(run())
        assert not session.halted
        assert len(session.tokens) == 2


# ── Pipeline performance ─────────────────────────────────────────


class TestAsyncHaltPerformance:
    """Document async halt pipeline characteristics."""

    def test_halt_reason_is_descriptive(self):
        kernel = AsyncStreamingKernel(hard_limit=0.5)

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a"]),
                lambda t: 0.1,
            )

        session = _run(run())
        # Halt reason must be human-readable, not empty
        assert len(session.halt_reason) > 5

    def test_coherence_history_length_matches_tokens(self):
        kernel = AsyncStreamingKernel(hard_limit=0.1)

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b", "c"]),
                lambda t: 0.9,
            )

        session = _run(run())
        assert len(session.coherence_history) == len(session.tokens)
