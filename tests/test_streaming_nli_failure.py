# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Streaming NLI Failure Propagation Tests

"""Tests for NLI model failure during streaming."""

from __future__ import annotations

import pytest

from director_ai.core.streaming import StreamingKernel


class TestNLIFailurePropagation:
    def test_callback_runtime_error_propagates(self):
        kernel = StreamingKernel(hard_limit=0.1)

        call_count = [0]

        def failing_callback(_text):
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("CUDA OOM")
            return 0.9

        with pytest.raises(RuntimeError, match="CUDA OOM"):
            kernel.stream_tokens(iter(["a", "b", "c", "d"]), failing_callback)

    def test_callback_value_error_propagates(self):
        kernel = StreamingKernel(hard_limit=0.1)

        def bad_callback(_text):
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            kernel.stream_tokens(iter(["a"]), bad_callback)

    def test_callback_returns_non_numeric(self):
        kernel = StreamingKernel(hard_limit=0.1)

        def bad_return(_text):
            return "not a number"

        with pytest.raises(TypeError):
            kernel.stream_tokens(iter(["a", "b"]), bad_return)

    def test_normal_halt_sets_halted_flag(self):
        kernel = StreamingKernel(hard_limit=0.5)
        session = kernel.stream_tokens(iter(["a", "b", "c"]), lambda _: 0.3)
        assert session.halted
        assert session.halt_reason
        assert "hard_limit" in session.halt_reason

    def test_no_halt_on_good_scores(self):
        kernel = StreamingKernel(hard_limit=0.1)
        session = kernel.stream_tokens(iter(["a", "b", "c"]), lambda _: 0.9)
        assert not session.halted
        assert len(session.tokens) == 3
