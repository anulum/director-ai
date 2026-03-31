# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming NLI Failure Propagation Tests (STRONG)
"""Multi-angle tests for NLI model failure during streaming.

Covers: callback RuntimeError propagation, ValueError propagation,
non-numeric return handling, normal halt on low scores, no-halt on
good scores, various halt reasons, callback failure at different
positions, partial output on halt, score history tracking,
and pipeline performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core.streaming import StreamingKernel


@pytest.fixture
def kernel():
    return StreamingKernel(hard_limit=0.1)


@pytest.fixture
def strict_kernel():
    return StreamingKernel(hard_limit=0.5)


# ── Error propagation ────────────────────────────────────────────


class TestErrorPropagation:
    """Callback errors must propagate to caller."""

    def test_runtime_error_propagates(self, kernel):
        call_count = [0]

        def failing_callback(_text):
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("CUDA OOM")
            return 0.9

        with pytest.raises(RuntimeError, match="CUDA OOM"):
            kernel.stream_tokens(iter(["a", "b", "c", "d"]), failing_callback)

    def test_value_error_propagates(self, kernel):
        def bad_callback(_text):
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            kernel.stream_tokens(iter(["a"]), bad_callback)

    def test_type_error_on_non_numeric(self, kernel):
        def bad_return(_text):
            return "not a number"

        with pytest.raises(TypeError):
            kernel.stream_tokens(iter(["a", "b"]), bad_return)

    @pytest.mark.parametrize(
        "exception_cls",
        [
            RuntimeError,
            ValueError,
            TypeError,
        ],
    )
    def test_various_exceptions_propagate(self, kernel, exception_cls):
        def raiser(_text):
            raise exception_cls("test error")

        with pytest.raises(exception_cls):
            kernel.stream_tokens(iter(["a"]), raiser)

    @pytest.mark.parametrize("fail_at", [1, 2, 5])
    def test_failure_at_different_positions(self, kernel, fail_at):
        count = [0]

        def callback(_text):
            count[0] += 1
            if count[0] == fail_at:
                raise RuntimeError(f"fail at {fail_at}")
            return 0.9

        tokens = [f"t{i}" for i in range(10)]
        with pytest.raises(RuntimeError, match=f"fail at {fail_at}"):
            kernel.stream_tokens(iter(tokens), callback)


# ── Normal halt behaviour ────────────────────────────────────────


class TestNormalHalt:
    """Streaming kernel must halt correctly on low coherence scores."""

    def test_halt_sets_flag(self, strict_kernel):
        session = strict_kernel.stream_tokens(iter(["a", "b", "c"]), lambda _: 0.3)
        assert session.halted
        assert session.halt_reason
        assert "hard_limit" in session.halt_reason

    def test_no_halt_on_good_scores(self, kernel):
        session = kernel.stream_tokens(iter(["a", "b", "c"]), lambda _: 0.9)
        assert not session.halted
        assert len(session.tokens) == 3

    @pytest.mark.parametrize(
        "score,should_halt",
        [
            (0.01, True),
            (0.05, True),
            (0.09, True),
            (0.5, False),
            (0.9, False),
            (1.0, False),
        ],
    )
    def test_halt_boundary(self, score, should_halt):
        kernel = StreamingKernel(hard_limit=0.1)
        session = kernel.stream_tokens(iter(["a", "b", "c"]), lambda _: score)
        assert session.halted == should_halt

    def test_partial_output_on_halt(self, strict_kernel):
        session = strict_kernel.stream_tokens(
            iter(["a", "b", "c", "d", "e"]), lambda _: 0.3
        )
        assert session.halted
        # Should have processed some tokens before halt
        assert len(session.tokens) >= 1

    def test_coherence_history_tracked(self, kernel):
        session = kernel.stream_tokens(iter(["a", "b", "c"]), lambda _: 0.8)
        assert len(session.coherence_history) == 3


# ── Pipeline performance ─────────────────────────────────────────


class TestStreamingPerformance:
    """Document streaming kernel performance characteristics."""

    def test_session_has_output(self, kernel):
        session = kernel.stream_tokens(iter(["hello", "world"]), lambda _: 0.9)
        assert isinstance(session.output, str)
        assert len(session.output) > 0

    def test_session_has_token_count(self, kernel):
        session = kernel.stream_tokens(iter(["a", "b", "c"]), lambda _: 0.9)
        assert session.token_count == 3

    def test_session_has_avg_coherence(self, kernel):
        session = kernel.stream_tokens(iter(["a", "b"]), lambda _: 0.8)
        assert abs(session.avg_coherence - 0.8) < 0.01

    def test_session_has_duration(self, kernel):
        session = kernel.stream_tokens(iter(["a"]), lambda _: 0.9)
        assert session.duration_ms >= 0
