"""Unit tests for SafetyKernel (kernel.py)."""

from __future__ import annotations

import pytest

from director_ai.core.kernel import SafetyKernel


class TestSafetyKernelConstruction:
    def test_default_hard_limit(self):
        k = SafetyKernel()
        assert k.hard_limit == 0.5

    def test_custom_hard_limit(self):
        k = SafetyKernel(hard_limit=0.3)
        assert k.hard_limit == 0.3

    def test_starts_active(self):
        k = SafetyKernel()
        assert k.is_active is True


class TestStreamOutput:
    def test_pass_all_tokens(self):
        k = SafetyKernel(hard_limit=0.5)
        tokens = ["Hello", " ", "world"]
        result = k.stream_output(iter(tokens), lambda _t: 0.9)
        assert result == "Hello world"

    def test_halt_on_low_coherence(self):
        k = SafetyKernel(hard_limit=0.5)
        tokens = ["ok", "ok", "bad"]
        scores = iter([0.8, 0.7, 0.3])
        result = k.stream_output(iter(tokens), lambda _t: next(scores))
        assert "KERNEL INTERRUPT" in result

    def test_halt_deactivates_kernel(self):
        k = SafetyKernel(hard_limit=0.5)
        k.stream_output(iter(["x"]), lambda _t: 0.1)
        assert k.is_active is False

    def test_empty_generator(self):
        k = SafetyKernel(hard_limit=0.5)
        result = k.stream_output(iter([]), lambda _t: 0.9)
        assert result == ""

    def test_single_token_pass(self):
        k = SafetyKernel(hard_limit=0.5)
        result = k.stream_output(iter(["hi"]), lambda _t: 0.8)
        assert result == "hi"

    def test_boundary_score_equal_limit_passes(self):
        k = SafetyKernel(hard_limit=0.5)
        result = k.stream_output(iter(["a", "b"]), lambda _t: 0.5)
        assert result == "ab"

    def test_boundary_score_just_below_halts(self):
        k = SafetyKernel(hard_limit=0.5)
        result = k.stream_output(iter(["a"]), lambda _t: 0.4999)
        assert "KERNEL INTERRUPT" in result


class TestEmergencyStop:
    def test_emergency_stop_sets_inactive(self):
        k = SafetyKernel()
        k.emergency_stop()
        assert k.is_active is False

    def test_emergency_stop_idempotent(self):
        k = SafetyKernel()
        k.emergency_stop()
        k.emergency_stop()
        assert k.is_active is False


class TestOnHaltCallback:
    def test_on_halt_called_with_score(self):
        received = []
        k = SafetyKernel(hard_limit=0.5, on_halt=lambda s: received.append(s))
        k.stream_output(iter(["x"]), lambda _t: 0.2)
        assert len(received) == 1
        assert received[0] == 0.2

    def test_on_halt_not_called_when_pass(self):
        received = []
        k = SafetyKernel(hard_limit=0.5, on_halt=lambda s: received.append(s))
        k.stream_output(iter(["x"]), lambda _t: 0.9)
        assert len(received) == 0

    def test_on_halt_exception_propagates(self):
        def boom(score):
            raise ValueError("halt handler failed")

        k = SafetyKernel(hard_limit=0.5, on_halt=boom)
        with pytest.raises(ValueError, match="halt handler failed"):
            k.stream_output(iter(["x"]), lambda _t: 0.1)


class TestTimeouts:
    def test_total_timeout_halts(self):
        import time

        def slow_gen():
            for tok in ["a", "b", "c", "d", "e"]:
                time.sleep(0.05)
                yield tok

        k = SafetyKernel(hard_limit=0.1, total_timeout=0.1)
        result = k.stream_output(slow_gen(), lambda _t: 0.9)
        assert "TOTAL TIMEOUT" in result
        assert k.is_active is False

    def test_token_timeout_halts(self):
        import time

        def slow_callback(_t):
            time.sleep(0.15)
            return 0.9

        k = SafetyKernel(hard_limit=0.1, token_timeout=0.05)
        result = k.stream_output(iter(["a"]), slow_callback)
        assert "TOKEN TIMEOUT" in result
        assert k.is_active is False

    def test_total_timeout_disabled_by_default(self):
        k = SafetyKernel()
        assert k.total_timeout == 0.0
        result = k.stream_output(iter(["a", "b"]), lambda _t: 0.9)
        assert result == "ab"

    def test_token_timeout_disabled_by_default(self):
        k = SafetyKernel()
        assert k.token_timeout == 0.0
        result = k.stream_output(iter(["a"]), lambda _t: 0.9)
        assert result == "a"

    def test_coherence_halt_takes_priority_over_timeout(self):
        k = SafetyKernel(hard_limit=0.5, total_timeout=10.0)
        result = k.stream_output(iter(["x"]), lambda _t: 0.1)
        assert "COHERENCE LIMIT" in result
