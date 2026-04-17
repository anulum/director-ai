# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming Debug Mode Tests
"""Multi-angle tests for streaming kernel debug mode.

Covers: debug log population, debug disabled path, event debug_info,
window_avg tracking, accumulated token count, halt event debug,
parametrised debug toggle, pipeline performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core.streaming import StreamingKernel


def _constant_coherence(val: float):
    def cb(token):
        return val

    return cb


def test_debug_log_populated_when_enabled():
    kernel = StreamingKernel(hard_limit=0.3, streaming_debug=True)
    tokens = ["The", "sky", "is", "blue", "today"]
    session = kernel.stream_tokens(iter(tokens), _constant_coherence(0.8))
    assert not session.halted
    assert len(session.debug_log) == 5
    for snap in session.debug_log:
        assert "index" in snap
        assert "coherence" in snap
        assert "window_avg" in snap
        assert "trend_drop" in snap
        assert "accumulated_tokens" in snap


def test_debug_log_empty_when_disabled():
    kernel = StreamingKernel(hard_limit=0.3, streaming_debug=False)
    tokens = ["The", "sky", "is", "blue"]
    session = kernel.stream_tokens(iter(tokens), _constant_coherence(0.8))
    assert session.debug_log == []


def test_debug_info_on_token_events():
    kernel = StreamingKernel(hard_limit=0.3, streaming_debug=True)
    tokens = ["alpha", "beta", "gamma"]
    session = kernel.stream_tokens(iter(tokens), _constant_coherence(0.7))
    for event in session.events:
        assert event.debug_info is not None
        assert event.debug_info["coherence"] == pytest.approx(0.7)


def test_debug_info_none_when_disabled():
    kernel = StreamingKernel(hard_limit=0.3, streaming_debug=False)
    tokens = ["alpha", "beta"]
    session = kernel.stream_tokens(iter(tokens), _constant_coherence(0.7))
    for event in session.events:
        assert event.debug_info is None


def test_debug_window_avg_tracks_scores():
    scores = iter([0.9, 0.8, 0.7, 0.6, 0.5])

    def cb(token):
        return next(scores)

    kernel = StreamingKernel(
        hard_limit=0.3,
        window_size=3,
        streaming_debug=True,
    )
    tokens = ["a", "b", "c", "d", "e"]
    session = kernel.stream_tokens(iter(tokens), cb)
    # After 3 tokens (0.9, 0.8, 0.7), window_avg = 0.8
    assert session.debug_log[2]["window_avg"] == pytest.approx(0.8, abs=0.01)


def test_debug_accumulated_tokens_increments():
    kernel = StreamingKernel(hard_limit=0.3, streaming_debug=True)
    tokens = ["x", "y", "z"]
    session = kernel.stream_tokens(iter(tokens), _constant_coherence(0.8))
    for i, snap in enumerate(session.debug_log):
        assert snap["accumulated_tokens"] == i + 1


def test_debug_on_halt_event():
    kernel = StreamingKernel(hard_limit=0.5, streaming_debug=True)
    scores = iter([0.8, 0.3])

    def cb(token):
        return next(scores)

    session = kernel.stream_tokens(iter(["ok", "bad"]), cb)
    assert session.halted
    assert len(session.debug_log) == 2
    assert session.debug_log[1]["coherence"] == pytest.approx(0.3)


@pytest.mark.parametrize("debug", [True, False])
def test_debug_toggle(debug):
    kernel = StreamingKernel(hard_limit=0.3, streaming_debug=debug)
    session = kernel.stream_tokens(iter(["a", "b"]), _constant_coherence(0.8))
    if debug:
        assert len(session.debug_log) == 2
    else:
        assert session.debug_log == []


@pytest.mark.parametrize("n_tokens", [1, 3, 5, 10])
def test_debug_log_length_matches_tokens(n_tokens):
    kernel = StreamingKernel(hard_limit=0.1, streaming_debug=True)
    tokens = [f"t{i}" for i in range(n_tokens)]
    session = kernel.stream_tokens(iter(tokens), _constant_coherence(0.9))
    assert len(session.debug_log) == n_tokens


class TestStreamingDebugPerformance:
    """Document streaming debug pipeline performance."""

    def test_debug_snapshot_has_required_keys(self):
        kernel = StreamingKernel(hard_limit=0.1, streaming_debug=True)
        session = kernel.stream_tokens(iter(["a"]), _constant_coherence(0.8))
        snap = session.debug_log[0]
        for key in [
            "index",
            "coherence",
            "window_avg",
            "trend_drop",
            "accumulated_tokens",
        ]:
            assert key in snap, f"Missing key: {key}"

    def test_debug_overhead_minimal(self):
        import time

        kernel_debug = StreamingKernel(hard_limit=0.1, streaming_debug=True)
        kernel_plain = StreamingKernel(hard_limit=0.1, streaming_debug=False)
        tokens = [f"t{i}" for i in range(100)]

        t0 = time.perf_counter()
        kernel_plain.stream_tokens(iter(tokens), _constant_coherence(0.9))
        plain_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        kernel_debug.stream_tokens(iter(tokens), _constant_coherence(0.9))
        debug_ms = (time.perf_counter() - t0) * 1000

        # Debug mode should add <500ms absolute overhead (relative
        # comparison is too flaky on shared CI runners where plain_ms
        # can be <3ms and any GC/scheduling jitter dominates)
        overhead = debug_ms - plain_ms
        assert overhead < 500, (
            f"Debug overhead too high: {overhead:.1f}ms "
            f"(debug={debug_ms:.1f}ms, plain={plain_ms:.1f}ms)"
        )
