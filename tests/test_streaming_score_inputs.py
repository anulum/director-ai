# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — synchronous and asynchronous streaming score inputs
"""Exercise host score inputs through real streaming APIs and native statistics."""

from __future__ import annotations

import asyncio
import math

import pytest

from director_ai.core.runtime.async_streaming import AsyncStreamingKernel
from director_ai.core.runtime.streaming import StreamingKernel, StreamSession

Kernel = StreamingKernel | AsyncStreamingKernel
KernelFactory = type[StreamingKernel] | type[AsyncStreamingKernel]
FACTORIES = [StreamingKernel, AsyncStreamingKernel]
NONFINITE = [float("nan"), float("inf"), -float("inf")]


def _collect(kernel: Kernel, scores: list[float]) -> StreamSession:
    """Run an actual stream with the supplied host score for each token."""
    tokens = [f"token-{index}" for index in range(len(scores))]
    values = iter(scores)
    if isinstance(kernel, AsyncStreamingKernel):
        return asyncio.run(kernel.stream_to_session(tokens, lambda _text: next(values)))
    return kernel.stream_tokens(tokens, lambda _text: next(values))


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("score", NONFINITE)
def test_nonfinite_score_is_zero_before_history_and_halt(
    factory: KernelFactory, score: float
) -> None:
    """Reject an unscored token through the normal halt, trace and metrics path."""
    callbacks: list[StreamSession] = []
    kernel = factory(on_halt=callbacks.append)
    session = _collect(kernel, [0.9, 0.9, score, 0.9])

    assert session.halted
    assert session.halt_index == 2
    assert session.halt_reason.startswith("hard_limit (0.0000")
    assert session.tokens == ["token-0", "token-1", "token-2"]
    assert session.output == "token-0token-1"
    assert session.coherence_history == [0.9, 0.9, 0.0]
    assert [event.coherence for event in session.events] == [0.9, 0.9, 0.0]
    assert session.events[-1].halted
    assert math.isfinite(session.avg_coherence)
    assert session.min_coherence == 0.0
    assert callbacks == [session]
    assert not kernel.is_active


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("score", NONFINITE)
def test_zero_hard_limit_preserves_strict_comparison_for_nonfinite_score(
    factory: KernelFactory, score: float
) -> None:
    """A zero fallback can pass when the configured limits permit zero."""
    session = _collect(factory(hard_limit=0.0), [score])
    assert not session.halted
    assert session.coherence_history == [0.0]
    assert session.events[0].coherence == 0.0
    assert session.avg_coherence == 0.0


@pytest.mark.parametrize("score", NONFINITE)
def test_sync_incremental_gate_uses_the_same_zero_fallback(score: float) -> None:
    """The public incremental gate must not store a nonfinite observation."""
    kernel = StreamingKernel(window_size=2, window_threshold=0.0, trend_window=2)
    assert kernel.check_halt(score)
    assert not kernel.check_halt(0.9)
    kernel.reset_state()
    assert not kernel.check_halt(0.9)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    ("window_size", "trend_window", "trend_threshold", "field"),
    [
        (0, 3, 0.2, "window_size"),
        (-1, 3, 0.2, "window_size"),
        (3, 0, 0.2, "trend_window"),
        (3, 1, 0.2, "trend_window"),
        (3, 3, 0.0, "trend_threshold"),
        (3, 3, -0.1, "trend_threshold"),
    ],
)
def test_invalid_window_configuration_is_refused_before_streaming(
    factory: KernelFactory,
    window_size: int,
    trend_window: int,
    trend_threshold: float,
    field: str,
) -> None:
    """Both Python constructors enforce the native window and trend minima."""
    with pytest.raises(ValueError, match=field):
        factory(
            window_size=window_size,
            trend_window=trend_window,
            trend_threshold=trend_threshold,
        )


@pytest.mark.parametrize("factory", FACTORIES)
def test_minimum_valid_windows_accept_a_complete_stream(factory: KernelFactory) -> None:
    """Keep the inclusive constructor boundaries usable through real scoring."""
    session = _collect(factory(window_size=1, trend_window=2), [0.9, 0.9])
    assert not session.halted
    assert session.coherence_history == [0.9, 0.9]


@pytest.mark.parametrize("score", NONFINITE)
def test_async_callback_nonfinite_result_halts_after_adaptive_reuse(
    score: float,
) -> None:
    """Sanitise an awaited score before adaptive decisions or event emission."""
    values = iter([0.9, score])

    async def callback(_text: str) -> float:
        """Supply actual async callback results at the selected score cadence."""
        return next(values)

    kernel = AsyncStreamingKernel(adaptive=True)
    session = asyncio.run(kernel.stream_to_session(["a", "b", "c", "d"], callback))
    assert session.halted
    assert session.halt_index == 2
    assert session.coherence_history == [0.9, 0.9, 0.0]
    assert session.halt_reason.startswith("hard_limit")


def test_sync_cadence_reuses_the_sanitised_score() -> None:
    """A skipped scoring step must reuse zero rather than the original NaN."""
    kernel = StreamingKernel(hard_limit=0.0, window_threshold=0.0, score_every_n=2)
    session = kernel.stream_tokens(["a", "b"], lambda _text: float("nan"))
    assert not session.halted
    assert session.coherence_history == [0.0, 0.0]
    assert session.avg_coherence == 0.0


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    ("scores", "reason"),
    [
        ([0.9, 0.1], "hard_limit"),
        ([0.54, 0.54, 0.54], "window_avg"),
        ([0.95, 0.85, 0.55], "downward_trend"),
    ],
)
def test_finite_scores_preserve_each_existing_halt_decision(
    factory: KernelFactory, scores: list[float], reason: str
) -> None:
    """Keep the three existing decisions and full finite token history intact."""
    session = _collect(
        factory(window_size=3, trend_window=3, trend_threshold=0.2), scores
    )
    assert session.halted
    assert session.halt_reason.startswith(reason)
    assert session.coherence_history == scores
    assert session.halt_index == len(scores) - 1


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    ("scores", "halted"),
    [
        ([0.8, 1.0, 0.8, 0.5, 0.8], True),
        ([0.9, 0.5, 0.8, 1.0, 0.7], False),
    ],
)
def test_five_point_regression_uses_all_scores_instead_of_only_endpoints(
    factory: KernelFactory, scores: list[float], halted: bool
) -> None:
    """Keep both the declining and rising regression decisions across hosts."""
    session = _collect(factory(trend_window=5, trend_threshold=0.15), scores)
    assert session.halted is halted
    assert session.coherence_history == scores
    if halted:
        assert session.halt_reason.startswith("downward_trend (0.2000")
        assert session.halt_index == 4
    else:
        assert session.output == "token-0token-1token-2token-3token-4"


@pytest.mark.parametrize("factory", FACTORIES)
def test_recent_window_halts_even_when_the_full_history_average_passes(
    factory: KernelFactory,
) -> None:
    """Evict old high scores while keeping the full audit history available."""
    scores = [0.9, 0.9, 0.9, 0.6, 0.6, 0.6]
    kernel = factory(window_size=3, window_threshold=0.7, trend_threshold=1.0)
    session = _collect(kernel, scores)
    assert session.halted
    assert session.halt_index == 5
    assert session.coherence_history == scores
    assert session.halt_reason.startswith("window_avg (0.6000")
    assert not kernel.is_active
