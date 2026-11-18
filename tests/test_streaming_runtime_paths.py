# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real streaming cadence, deferred halts and deadlines
"""Exercise public stream behavior with real native statistics and elapsed time."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator

import pytest

from director_ai.core.kernel import SafetyKernel
from director_ai.core.runtime.async_streaming import AsyncStreamingKernel
from director_ai.core.runtime.streaming import (
    StreamingKernel,
    StreamSession,
    TokenEvent,
)

Kernel = StreamingKernel | AsyncStreamingKernel
KernelFactory = type[StreamingKernel] | type[AsyncStreamingKernel]
FACTORIES = [StreamingKernel, AsyncStreamingKernel]


def _collect(
    kernel: Kernel, tokens: list[str], callback: Callable[[str], float]
) -> StreamSession:
    """Collect through the public API with unchanged host inputs and callbacks."""
    if isinstance(kernel, AsyncStreamingKernel):
        return asyncio.run(kernel.stream_to_session(tokens, callback))
    return kernel.stream_tokens(tokens, callback)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    ("tokens", "observed"),
    [(["A", ".", "unused"], 2), (["A", "B", "C", ".", "unused"], 4), (["x"] * 60, 52)],
    ids=["trigger-at-boundary", "later-boundary", "fifty-extra-token-cap"],
)
def test_soft_halt_session_retains_the_trigger_and_all_delivered_tokens(
    factory: KernelFactory, tokens: list[str], observed: int
) -> None:
    """Deferred halts preserve their original cause, offset and sentence output."""
    callbacks: list[StreamSession] = []
    kernel = factory(
        hard_limit=0.3,
        window_size=2,
        window_threshold=0.6,
        trend_threshold=1.0,
        halt_mode="soft",
        on_halt=callbacks.append,
    )
    session = _collect(kernel, tokens, lambda _text: 0.5)
    assert session.halted
    assert session.soft_halted
    assert session.halt_index == 1
    assert session.halt_reason.startswith("window_avg (0.5000")
    assert session.tokens == tokens[:observed]
    assert session.output == "".join(tokens[:observed])
    assert session.events[-1].halted
    assert session.warning_count == 1
    assert callbacks == [session]
    assert not kernel.is_active
    kernel.reactivate()
    resumed = _collect(kernel, ["safe"], lambda _text: 0.9)
    assert not resumed.halted
    assert not resumed.soft_halted
    assert resumed.halt_index == -1
    assert resumed.output == "safe"
    assert callbacks == [session]


@pytest.mark.parametrize("factory", FACTORIES)
def test_hard_limit_overrides_a_pending_soft_halt(factory: KernelFactory) -> None:
    """A later hard violation replaces the deferred offset and excludes its token."""
    values = iter([0.5, 0.5, 0.2])
    kernel = factory(
        hard_limit=0.3, window_size=2, window_threshold=0.6, halt_mode="soft"
    )
    session = _collect(kernel, ["A", "B", "bad", "unused"], lambda _text: next(values))
    assert session.halted
    assert not session.soft_halted
    assert session.halt_index == 2
    assert session.output == "AB"
    assert session.halt_reason.startswith("hard_limit (0.2000")
    assert session.warning_count == 1


@pytest.mark.parametrize("factory", FACTORIES)
def test_empty_session_metrics_and_duration_are_well_defined(
    factory: KernelFactory,
) -> None:
    """An empty real stream produces no events, zero scores and a finite duration."""
    session = _collect(factory(), [], float)
    assert not session.halted
    assert session.token_count == 0
    assert session.output == ""
    assert session.events == []
    assert session.avg_coherence == session.min_coherence == 0.0
    assert session.duration_ms >= 0.0


@pytest.mark.parametrize("factory", FACTORIES)
def test_adaptive_cadence_recovers_from_low_scores_and_respects_its_cap(
    factory: KernelFactory,
) -> None:
    """Actual callback offsets expose cadence growth, reuse, reset and recovery."""
    observed: list[str] = []
    values = iter([0.9, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9, 0.9])

    def score(text: str) -> float:
        """Record the actual accumulated text at every scoring invocation."""
        observed.append(text)
        return next(values)

    kernel = factory(
        hard_limit=0.0,
        window_size=1,
        window_threshold=0.0,
        trend_threshold=2.0,
        adaptive=True,
        max_cadence=3,
    )
    session = _collect(kernel, list("abcdefghijkl"), score)
    assert not session.halted
    assert observed == [
        "a",
        "abc",
        "abcd",
        "abcdefg",
        "abcdefgh",
        "abcdefghi",
        "abcdefghij",
        "abcdefghijk",
    ]
    assert session.coherence_history == [0.9] * 6 + [0.4, 0.4] + [0.9] * 4
    assert session.warning_count == 2
    assert session.output == "abcdefghijkl"


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("halt_mode", ["invalid", ""])
def test_invalid_halt_mode_is_refused(factory: KernelFactory, halt_mode: str) -> None:
    """Both public constructors reject unsupported halt delivery modes."""
    with pytest.raises(ValueError, match="halt_mode"):
        factory(halt_mode=halt_mode)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_scoring_cadence_is_refused(factory: KernelFactory, value: int) -> None:
    """Zero and negative callback intervals or adaptive caps cannot be admitted."""
    with pytest.raises(ValueError, match="score_every_n"):
        factory(score_every_n=value)
    with pytest.raises(ValueError, match="max_cadence"):
        factory(max_cadence=value)


@pytest.mark.parametrize("limit", [-0.1, 1.1, float("nan")])
def test_basic_and_async_gates_refuse_invalid_hard_limits(limit: float) -> None:
    """Both constructor paths reject values outside their documented score domain."""
    with pytest.raises(ValueError, match="hard_limit"):
        SafetyKernel(hard_limit=limit)
    with pytest.raises(ValueError, match="hard_limit"):
        AsyncStreamingKernel(hard_limit=limit)


def test_basic_gate_measures_callback_and_total_elapsed_deadlines() -> None:
    """Actual elapsed waits trigger deadlines without a patched clock or scorer."""

    def slow_score(_text: str) -> float:
        """Perform a real host wait exceeding the configured callback budget."""
        time.sleep(0.01)
        return 0.9

    token_gate = SafetyKernel(token_timeout=0.001)
    assert (
        token_gate.stream_output(["late"], slow_score)
        == "[HALT: TOKEN TIMEOUT EXCEEDED]"
    )
    assert not token_gate.is_active

    def delayed_source() -> Iterator[str]:
        """Supply a first token before a real delay and a second after it."""
        yield "first"
        time.sleep(0.01)
        yield "late"

    observed: list[str] = []

    def score(text: str) -> float:
        """Record the last admitted token before the source exceeded its budget."""
        observed.append(text)
        return 0.9

    total_gate = SafetyKernel(total_timeout=0.001)
    assert (
        total_gate.stream_output(delayed_source(), score)
        == "[HALT: TOTAL TIMEOUT EXCEEDED]"
    )
    assert observed == ["first"]
    assert not total_gate.is_active
    assert (
        SafetyKernel(token_timeout=60.0, total_timeout=60.0).stream_output(
            ["ok"], score
        )
        == "ok"
    )
    assert (
        SafetyKernel().stream_output(["bad"], lambda _text: 0.1)
        == "[HALT: COHERENCE BELOW THRESHOLD]"
    )


@pytest.mark.parametrize("scores", [[0.1], [0.5, 0.5], [0.9, 0.8, 0.5]])
def test_async_event_generator_finishes_without_a_consumer_break(
    scores: list[float],
) -> None:
    """The real async generator terminates itself after each configured halt."""

    async def collect() -> list[TokenEvent]:
        """Exhaust every yielded event without simulating caller-side stopping."""
        values = iter(scores)
        kernel = AsyncStreamingKernel(
            window_size=2, window_threshold=0.55, trend_window=3
        )
        return [
            event
            async for event in kernel.stream_tokens(
                ["x"] * 5, lambda _text: next(values)
            )
        ]

    events = asyncio.run(collect())
    assert len(events) == len(scores)
    assert events[-1].halted


def test_async_source_and_callback_are_awaited_through_the_public_api() -> None:
    """Real async iteration handles empty and nonempty sources without adapters."""

    async def collect(tokens: list[str]) -> StreamSession:
        """Collect a real asynchronous producer and scoring coroutine."""

        async def source() -> AsyncIterator[str]:
            """Yield actual upstream tokens across scheduling boundaries."""
            for token in tokens:
                await asyncio.sleep(0)
                yield token

        async def score(_text: str) -> float:
            """Resolve the host score through a real coroutine."""
            await asyncio.sleep(0)
            return 0.9

        return await AsyncStreamingKernel().stream_to_session(source(), score)

    assert asyncio.run(collect(["one", "two"])).output == "onetwo"
    assert asyncio.run(collect([])).token_count == 0


def test_async_dynamic_wire_tokens_are_normalised_before_scoring() -> None:
    """Actual JSON values exercise the runtime guard at an untyped input boundary."""
    wire_tokens = json.loads("[7, true, null]")
    session = asyncio.run(
        AsyncStreamingKernel().stream_to_session(wire_tokens, lambda _text: 0.9)
    )
    assert session.tokens == ["7", "True", "None"]
    assert session.output == "7TrueNone"


def test_async_success_resets_the_consecutive_callback_error_budget() -> None:
    """Real score parsing failures separated by success never exhaust the budget."""
    inputs = iter(["0.9", "invalid", "invalid", "0.9", "invalid", "invalid", "0.9"])

    def parse_score(_text: str) -> float:
        """Convert actual host score input, including genuinely invalid decimals."""
        return float(next(inputs))

    kernel = AsyncStreamingKernel()
    session = asyncio.run(kernel.stream_to_session(list("abcdefg"), parse_score))
    assert not session.halted
    assert kernel.is_active
    assert session.coherence_history == [0.9] * 7


@pytest.mark.parametrize("deadline", ["token", "total"])
def test_async_deadlines_preserve_the_actual_stop_cause(deadline: str) -> None:
    """A real deadline must not be reported as a low coherence measurement."""

    async def collect() -> StreamSession:
        """Exceed a real producer or callback budget without replacing the clock."""
        if deadline == "token":

            async def slow_score(_text: str) -> float:
                """Wait longer than the allowed per-callback processing time."""
                await asyncio.sleep(0.01)
                return 0.9

            return await AsyncStreamingKernel(token_timeout=0.001).stream_to_session(
                ["late"], slow_score
            )

        async def delayed_source() -> AsyncIterator[str]:
            """Admit one token before an actual producer delay exceeds the budget."""
            yield "first"
            await asyncio.sleep(0.01)
            yield "late"

        return await AsyncStreamingKernel(total_timeout=0.001).stream_to_session(
            delayed_source(), lambda _text: 0.9
        )

    session = asyncio.run(collect())
    assert session.halted
    assert not session.soft_halted
    assert session.halt_reason == f"{deadline}_timeout"


def test_async_callback_error_budget_preserves_the_actual_stop_cause() -> None:
    """Exhausted real numeric parsing errors are not evidence of contradiction."""
    inputs = iter(["not-a-score"] * 3)

    def parse_score(_text: str) -> float:
        """Read and parse each actual invalid host input through the callback."""
        return float(next(inputs))

    kernel = AsyncStreamingKernel()
    session = asyncio.run(kernel.stream_to_session(list("abcd"), parse_score))
    assert session.halted
    assert not kernel.is_active
    assert session.halt_index == 2
    assert session.halt_reason == "callback_error"


@pytest.mark.parametrize("entrypoint", ["events", "session"])
def test_async_scoring_awaits_pending_futures(entrypoint: str) -> None:
    """An event-loop score must control delivery and the eventual halt."""

    async def collect() -> tuple[list[TokenEvent], list[str]]:
        """Return the events and actual inputs seen by the scoring callback."""
        loop = asyncio.get_running_loop()
        inputs: list[str] = []
        values = iter([0.9, 0.2])

        def score(text: str) -> asyncio.Future[float]:
            """Complete the next score after the callback returns."""
            inputs.append(text)
            pending: asyncio.Future[float] = loop.create_future()
            loop.call_soon(pending.set_result, next(values))
            return pending

        kernel = AsyncStreamingKernel(hard_limit=0.4)
        tokens = ["first", " second", " unused"]
        if entrypoint == "events":
            events = [event async for event in kernel.stream_tokens(tokens, score)]
        else:
            session = await kernel.stream_to_session(tokens, score)
            assert session.output == "first"
            events = session.events
        assert not kernel.is_active
        return events, inputs

    events, inputs = asyncio.run(collect())
    assert inputs == ["first", "first second"]
    assert [(event.coherence, event.halted) for event in events] == [
        (0.9, False),
        (0.2, True),
    ]


@pytest.mark.parametrize("failure", ["callback_error", "token_timeout"])
def test_async_future_failure_terminates_event_iterator(failure: str) -> None:
    """An event consumer can exhaust the stream after the operational halt."""

    async def collect() -> tuple[list[TokenEvent], list[str]]:
        """Collect without breaking when the kernel emits a halt."""
        loop = asyncio.get_running_loop()
        inputs: list[str] = []

        def score(text: str) -> asyncio.Future[float]:
            """Resolve the pending score after a failure or exceeded deadline."""
            inputs.append(text)
            pending: asyncio.Future[float] = loop.create_future()
            if failure == "callback_error":
                loop.call_soon(pending.set_exception, ValueError("score unavailable"))
            else:
                loop.call_later(0.02, pending.set_result, 0.9)
            return pending

        kernel = AsyncStreamingKernel(
            token_timeout=0.001 if failure == "token_timeout" else 0.0,
        )
        events = [event async for event in kernel.stream_tokens(list("abcd"), score)]
        assert not kernel.is_active
        return events, inputs

    events, inputs = asyncio.run(collect())
    if failure == "callback_error":
        assert inputs == ["a", "ab", "abc"]
        assert [event.halted for event in events] == [False, False, True]
        assert events[-1].coherence == 0.0
    else:
        assert inputs == ["a"]
        assert [event.halted for event in events] == [True]
        assert events[-1].coherence == 0.9


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    ("window_size", "scores", "trigger", "reason"),
    [
        (2, [0.5, 0.5, 0.9, 0.9], 1, "window_avg (0.5000"),
        (10, [0.95, 0.85, 0.55, 0.95, 0.95], 2, "downward_trend (0.4000"),
    ],
)
def test_soft_halt_keeps_its_original_cause_after_scores_recover(
    factory: KernelFactory,
    window_size: int,
    scores: list[float],
    trigger: int,
    reason: str,
) -> None:
    """The final sentence score cannot replace the initial deferred decision."""
    values = iter(scores)
    tokens = ["word "] * (len(scores) - 1) + ["."]
    kernel = factory(
        hard_limit=0.3,
        window_size=window_size,
        window_threshold=0.6,
        trend_window=3,
        trend_threshold=0.2,
        halt_mode="soft",
    )
    session = _collect(kernel, tokens + ["unused"], lambda _text: next(values))
    assert session.halted and session.soft_halted
    assert session.halt_index == trigger
    assert session.halt_reason.startswith(reason)
    assert session.coherence_history == scores
    assert session.output == "".join(tokens)


@pytest.mark.parametrize("scores", [[0.54, 0.54, 0.54], [0.95, 0.85, 0.55]])
def test_incremental_gate_evaluates_window_and_trend_and_can_reset(
    scores: list[float],
) -> None:
    """Incremental public checks retain history until the caller explicitly resets."""
    kernel = StreamingKernel(window_size=3, trend_window=3, trend_threshold=0.2)
    assert [kernel.check_halt(score) for score in scores] == [False, False, True]
    kernel.reset_state()
    assert not kernel.check_halt(0.9)


def test_sync_debug_snapshots_and_legacy_output_follow_real_stream_decisions() -> None:
    """Expose measured diagnostics and exercise both branches of the string API."""
    values = iter([0.9, 0.8, 0.7])
    kernel = StreamingKernel(streaming_debug=True, trend_window=3)
    session = kernel.stream_tokens(list("abc"), lambda _text: next(values))
    assert session.halted
    assert len(session.debug_log) == len(session.events) == 3
    assert session.debug_log[-1] == {
        "index": 2,
        "coherence": 0.7,
        "window_avg": 0.8,
        "trend_drop": 0.2,
        "accumulated_tokens": 3,
    }
    assert session.events[-1].debug_info == session.debug_log[-1]
    kernel.reset_state()
    assert kernel.stream_output(["safe"], lambda _text: 0.9) == "safe"
    assert kernel.stream_output(["bad"], lambda _text: 0.1).startswith(
        "[KERNEL INTERRUPT: hard_limit"
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_async_window_threshold_refuses_invalid_score_bounds(threshold: float) -> None:
    """Keep the async constructor's existing finite probability-domain guard."""
    with pytest.raises(ValueError, match="window_threshold"):
        AsyncStreamingKernel(window_threshold=threshold)
