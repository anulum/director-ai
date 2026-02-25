# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Tests for GPU-Accelerated UPDE Stepper
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pytest

from director_ai.research.physics.gpu_upde import (
    TorchUPDEConfig,
    TorchUPDEStepper,
    _resolve_device,
)
from director_ai.research.physics.l16_mechanistic import UPDEState
from director_ai.research.physics.scpn_params import load_omega_n


@pytest.mark.physics
class TestTorchUPDEStepper:
    """Tests for the GPU/CPU UPDE stepper."""

    @pytest.fixture
    def stepper(self):
        return TorchUPDEStepper(config=TorchUPDEConfig(device="cpu"))

    @pytest.fixture
    def state(self):
        np.random.seed(42)
        return UPDEState(theta=np.random.uniform(0, 2 * np.pi, 16))

    def test_step_produces_new_state(self, stepper, state):
        new = stepper.step(state)
        assert isinstance(new, UPDEState)
        assert new.step_count == 1
        assert not np.allclose(state.theta, new.theta)

    def test_step_n_advances_n_steps(self, stepper, state):
        new = stepper.step_n(state, 10)
        assert new.step_count == 10
        assert new.t == pytest.approx(0.1, abs=1e-10)

    def test_order_parameter_computed(self, stepper, state):
        new = stepper.step(state)
        assert 0.0 <= new.R_global <= 1.0

    def test_cpu_fallback_always_works(self):
        """CPU fallback works even without CUDA."""
        stepper = TorchUPDEStepper(config=TorchUPDEConfig(device="cpu"))
        state = UPDEState(theta=np.zeros(16))
        new = stepper.step(state)
        assert isinstance(new, UPDEState)

    def test_custom_omega_and_knm(self):
        omega = np.ones(4)
        knm = np.ones((4, 4)) * 0.1
        np.fill_diagonal(knm, 0.0)
        stepper = TorchUPDEStepper(
            omega=omega,
            knm=knm,
            config=TorchUPDEConfig(device="cpu"),
        )
        assert stepper.N == 4
        state = UPDEState(theta=np.zeros(4))
        new = stepper.step(state)
        assert new.theta.shape == (4,)

    def test_no_noise_deterministic(self):
        """With noise=0 and field=0, same initial → same result."""
        cfg = TorchUPDEConfig(noise_amplitude=0.0, field_pressure=0.0, device="cpu")
        s1 = TorchUPDEStepper(config=cfg)
        s2 = TorchUPDEStepper(config=cfg)
        state = UPDEState(theta=np.ones(16))
        r1 = s1.step(state)
        r2 = s2.step(state)
        np.testing.assert_allclose(r1.theta, r2.theta, atol=1e-12)

    def test_strong_coupling_increases_r(self, state):
        """Strong coupling should increase synchrony."""
        from director_ai.research.physics.scpn_params import build_knm_matrix

        K = build_knm_matrix() * 5.0  # noqa: N806
        stepper = TorchUPDEStepper(
            knm=K,
            config=TorchUPDEConfig(noise_amplitude=0.01, device="cpu"),
        )
        R_init = float(np.abs(np.mean(np.exp(1j * state.theta))))  # noqa: N806
        s = state
        for _ in range(200):
            s = stepper.step(s)
        assert s.R_global > R_init or s.R_global > 0.5

    def test_synchronized_phases_stable(self):
        """Equal phases + no noise → coupling vanishes."""
        cfg = TorchUPDEConfig(noise_amplitude=0.0, field_pressure=0.0, device="cpu")
        stepper = TorchUPDEStepper(config=cfg)
        theta_init = np.ones(16)
        state = UPDEState(theta=theta_init.copy())
        new = stepper.step(state)
        expected = theta_init + load_omega_n() * cfg.dt
        np.testing.assert_allclose(new.theta, expected, atol=1e-10)

    def test_resolve_device_auto(self):
        """Auto resolves without error."""
        dev = _resolve_device("auto")
        assert dev in ("cuda", "mps", "cpu")

    def test_resolve_device_explicit(self):
        assert _resolve_device("cpu") == "cpu"

    def test_is_gpu_property(self, stepper):
        """CPU stepper reports is_gpu=False."""
        assert stepper.is_gpu is False
