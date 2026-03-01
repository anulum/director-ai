# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Streaming Debug Mode Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
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
