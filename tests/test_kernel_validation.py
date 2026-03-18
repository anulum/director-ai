# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” SafetyKernel Validation Tests

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
