# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SafetyKernel Validation Tests
"""Multi-angle tests for SafetyKernel input validation.

Covers: hard_limit boundary validation (negative, above 1, NaN),
valid limits, emergency stop idempotency, token timeout, on_halt
callback, parametrised limits, pipeline integration, and performance.
"""

import pytest

from director_ai.core.kernel import SafetyKernel


class TestHardLimitValidation:
    def test_negative_hard_limit_rejected(self):
        with pytest.raises(ValueError, match="hard_limit must be in"):
            SafetyKernel(hard_limit=-0.1)

    def test_above_one_hard_limit_rejected(self):
        with pytest.raises(ValueError, match="hard_limit must be in"):
            SafetyKernel(hard_limit=1.1)

    def test_nan_hard_limit_rejected(self):
        with pytest.raises(ValueError, match="hard_limit must be in"):
            SafetyKernel(hard_limit=float("nan"))

    def test_inf_hard_limit_rejected(self):
        with pytest.raises(ValueError, match="hard_limit must be in"):
            SafetyKernel(hard_limit=float("inf"))

    def test_zero_hard_limit_accepted(self):
        k = SafetyKernel(hard_limit=0.0)
        assert k.hard_limit == 0.0

    def test_one_hard_limit_accepted(self):
        k = SafetyKernel(hard_limit=1.0)
        assert k.hard_limit == 1.0

    def test_default_hard_limit(self):
        k = SafetyKernel()
        assert k.hard_limit == 0.5


class TestTimeoutPaths:
    def test_total_timeout_interrupts(self):
        import time

        def slow_gen():
            for tok in ["a", "b", "c"]:
                time.sleep(0.05)
                yield tok

        k = SafetyKernel(total_timeout=0.01)
        result = k.stream_output(slow_gen(), lambda t: 0.9)
        assert "TOTAL TIMEOUT" in result

    def test_token_timeout_interrupts(self):
        import time

        def slow_callback(t):
            time.sleep(0.05)
            return 0.9

        k = SafetyKernel(token_timeout=0.01)
        result = k.stream_output(["a"], slow_callback)
        assert "TOKEN TIMEOUT" in result


class TestEmergencyStop:
    def test_idempotent(self):
        k = SafetyKernel()
        k.emergency_stop()
        assert k.is_active is False
        k.emergency_stop()
        assert k.is_active is False

    def test_on_halt_callback(self):
        scores = []
        k = SafetyKernel(hard_limit=0.8, on_halt=lambda s: scores.append(s))
        k.stream_output(["bad"], lambda t: 0.3)
        assert len(scores) == 1
        assert scores[0] == 0.3


class TestKernelParametrised:
    """Parametrised kernel validation tests."""

    @pytest.mark.parametrize("invalid_limit", [-1.0, -0.01, 1.01, 2.0, float("inf")])
    def test_invalid_limits_rejected(self, invalid_limit):
        with pytest.raises(ValueError):
            SafetyKernel(hard_limit=invalid_limit)

    @pytest.mark.parametrize("valid_limit", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_valid_limits_accepted(self, valid_limit):
        k = SafetyKernel(hard_limit=valid_limit)
        assert k is not None


class TestKernelPerformanceDoc:
    """Document SafetyKernel pipeline performance."""

    def test_stream_output_returns_string(self):
        k = SafetyKernel()
        result = k.stream_output(["hello", "world"], lambda t: 0.9)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_kernel_creation_fast(self):
        import time

        t0 = time.perf_counter()
        for _ in range(100):
            SafetyKernel()
        per_call_us = (time.perf_counter() - t0) / 100 * 1_000_000
        assert per_call_us < 500, f"Kernel creation took {per_call_us:.0f}µs"
