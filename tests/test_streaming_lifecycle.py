# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — streaming halt and reactivation lifecycle
"""Exercise persistent halt state through public sync, async and basic gates."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from director_ai.core.kernel import SafetyKernel
from director_ai.core.runtime.async_streaming import AsyncStreamingKernel
from director_ai.core.runtime.streaming import StreamingKernel, StreamSession

Kernel = StreamingKernel | AsyncStreamingKernel
KernelFactory = type[StreamingKernel] | type[AsyncStreamingKernel]
FACTORIES = [StreamingKernel, AsyncStreamingKernel]


def _collect(
    kernel: Kernel, tokens: list[str], callback: Callable[[str], float]
) -> StreamSession:
    """Collect a real token stream without changing its callback or state."""
    if isinstance(kernel, AsyncStreamingKernel):
        return asyncio.run(kernel.stream_to_session(tokens, callback))
    return kernel.stream_tokens(tokens, callback)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("scores", [[0.1], [0.54, 0.54, 0.54], [0.95, 0.85, 0.55]])
def test_every_halt_blocks_another_stream_until_explicit_reactivation(
    factory: KernelFactory, scores: list[float]
) -> None:
    """A window or trend halt must persist just as a hard-limit halt does."""
    kernel = factory(window_size=3, trend_window=3, trend_threshold=0.2)
    values = iter(scores)
    halted = _collect(kernel, ["observed"] * len(scores), lambda _text: next(values))
    assert halted.halted
    assert not kernel.is_active

    called: list[str] = []

    def score(text: str) -> float:
        """Record each actual scoring call while returning a passing score."""
        called.append(text)
        return 0.9

    blocked = _collect(kernel, ["must-not-emit"], score)
    assert blocked.halted
    assert blocked.output == ""
    assert called == []
    kernel.reactivate()
    resumed = _collect(kernel, ["safe"], score)
    assert not resumed.halted
    assert resumed.output == "safe"
    assert called == ["safe"]


@pytest.mark.parametrize("factory", FACTORIES)
def test_emergency_stop_blocks_scoring_before_the_first_stream(
    factory: KernelFactory,
) -> None:
    """An explicit external stop must take effect before invoking a scorer."""
    kernel = factory()
    kernel.emergency_stop()
    calls: list[str] = []

    def score(text: str) -> float:
        """Record whether an inactive stream invoked its scoring callback."""
        calls.append(text)
        return 0.9

    session = _collect(kernel, ["blocked"], score)
    assert session.halted
    assert session.output == ""
    assert calls == []
    kernel.reactivate()
    resumed = _collect(kernel, ["resumed"], score)
    assert not resumed.halted
    assert resumed.output == "resumed"
    assert calls == ["resumed"]


def test_basic_gate_emergency_stop_and_reactivation_use_the_public_alias() -> None:
    """The public basic gate must enforce its stop flag and permit rearming."""
    kernel = SafetyKernel()
    kernel.emergency_stop()
    calls: list[str] = []

    def score(text: str) -> float:
        """Record actual scoring rather than replacing the gate implementation."""
        calls.append(text)
        return 0.9

    assert kernel.stream_output(["blocked"], score).startswith("[HALT:")
    assert calls == []
    kernel.reactivate()
    assert kernel.stream_output(["safe"], score) == "safe"
    assert calls == ["safe"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_basic_gate_nonfinite_score_halts_with_zero_and_preserves_stop(
    score: float,
) -> None:
    """Basic, streaming and WASM paths must agree on the nonfinite fallback."""
    stopped: list[float] = []
    kernel = SafetyKernel(on_halt=stopped.append)
    scores = iter([0.9, score, 0.9])
    assert kernel.stream_output(["a", "b", "c"], lambda _text: next(scores)).startswith(
        "[HALT:"
    )
    assert stopped == [0.0]
    assert not kernel.is_active
    assert kernel.stream_output(["must-not-emit"], lambda _text: 0.9).startswith(
        "[HALT:"
    )


@pytest.mark.parametrize("factory", FACTORIES)
def test_successful_stream_does_not_deactivate_the_kernel(
    factory: KernelFactory,
) -> None:
    """Normal completion must leave the gate armed for a subsequent stream."""
    kernel = factory()
    for _ in range(2):
        session = _collect(kernel, ["safe"], lambda _text: 0.9)
        assert session.output == "safe"
        assert not session.halted
        assert kernel.is_active


@pytest.mark.parametrize("factory", FACTORIES)
def test_real_callback_io_failure_leaves_the_stream_stopped(
    factory: KernelFactory, tmp_path: Path
) -> None:
    """A failed real score read must leave the exhausted error path inactive."""
    missing = tmp_path / "score-not-produced.txt"

    def read_score(_text: str) -> float:
        """Read a host score from an actual absent file."""
        return float(missing.read_text())

    kernel = factory()
    session = _collect(kernel, ["a", "b", "c", "d"], read_score)
    assert session.halted
    assert not kernel.is_active


def test_basic_gate_real_callback_io_failure_stops_the_monitor(tmp_path: Path) -> None:
    """Keep the basic gate's existing callback failure stop behavior intact."""
    missing = tmp_path / "score-not-produced.txt"

    def read_score(_text: str) -> float:
        """Read a real missing score file through the production callback."""
        return float(missing.read_text())

    kernel = SafetyKernel()
    assert kernel.stream_output(["a"], read_score) == "[HALT: CALLBACK TIMEOUT]"
    assert not kernel.is_active
