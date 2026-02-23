# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — GPU-Accelerated UPDE Stepper (PyTorch Backend)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
GPU-accelerated UPDE integrator using PyTorch.

Provides ``TorchUPDEStepper`` — drop-in replacement for ``UPDEStepper``
that runs the Kuramoto coupling on GPU (CUDA/MPS) with automatic CPU
fallback.

Usage::

    stepper = TorchUPDEStepper(device="cuda")
    state = UPDEState(theta=np.random.uniform(0, 2*np.pi, 16))
    state = stepper.step(state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .l16_mechanistic import UPDEState
from .scpn_params import build_knm_matrix, load_omega_n

logger = logging.getLogger("DirectorAI.GPU_UPDE")

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class TorchUPDEConfig:
    """Configuration for the GPU UPDE stepper."""

    dt: float = 0.01
    field_pressure: float = 0.1
    noise_amplitude: float = 0.05
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device."""
    if not _TORCH_AVAILABLE:
        return "cpu"
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchUPDEStepper:
    """GPU-accelerated UPDE integrator using PyTorch.

    Drop-in replacement for ``UPDEStepper`` that offloads the Kuramoto
    coupling computation to GPU when available.

    Parameters
    ----------
    omega : ndarray (N,) — natural frequencies.
    knm : ndarray (N, N) — coupling matrix.
    config : TorchUPDEConfig — stepper configuration.
    """

    def __init__(
        self,
        omega: np.ndarray | None = None,
        knm: np.ndarray | None = None,
        config: TorchUPDEConfig | None = None,
    ) -> None:
        self.cfg = config or TorchUPDEConfig()
        omega_np = omega if omega is not None else load_omega_n()
        knm_np = knm if knm is not None else build_knm_matrix()
        self.N = len(omega_np)
        self.dt = self.cfg.dt
        self.noise_amplitude = self.cfg.noise_amplitude
        self.field_pressure = self.cfg.field_pressure

        self.device_name = _resolve_device(self.cfg.device)
        self._use_torch = _TORCH_AVAILABLE and self.device_name != "cpu"

        if self._use_torch:
            self._device = torch.device(self.device_name)
            self._omega = torch.tensor(
                omega_np, dtype=torch.float64, device=self._device
            )
            self._knm = torch.tensor(knm_np, dtype=torch.float64, device=self._device)
            logger.info("TorchUPDEStepper using device: %s", self.device_name)
        else:
            self._omega_np = omega_np
            self._knm_np = knm_np
            self._phase_diff = np.zeros((self.N, self.N), dtype=np.float64)
            self._sin_diff = np.zeros((self.N, self.N), dtype=np.float64)
            if _TORCH_AVAILABLE:
                logger.info("TorchUPDEStepper falling back to CPU (NumPy).")
            else:
                logger.info("PyTorch not available — using NumPy CPU backend.")

    @property
    def is_gpu(self) -> bool:
        """Whether computation is running on GPU."""
        return self._use_torch and self.device_name != "cpu"

    def step(self, state: UPDEState) -> UPDEState:
        """Advance state by one timestep."""
        if self._use_torch:
            return self._step_torch(state)
        return self._step_numpy(state)

    def step_n(self, state: UPDEState, n: int) -> UPDEState:
        """Advance state by n timesteps (batched on GPU)."""
        if self._use_torch:
            return self._step_n_torch(state, n)
        for _ in range(n):
            state = self._step_numpy(state)
        return state

    def _step_torch(self, state: UPDEState) -> UPDEState:
        """Single step using PyTorch tensors."""
        theta_t = torch.tensor(state.theta, dtype=torch.float64, device=self._device)

        # Phase difference matrix: [n, m] = θ_m - θ_n
        phase_diff = theta_t.unsqueeze(0) - theta_t.unsqueeze(1)
        sin_diff = torch.sin(phase_diff)

        # Kuramoto coupling
        coupling = torch.sum(self._knm * sin_diff, dim=1)

        # External field
        field_term = self.field_pressure * torch.cos(theta_t)

        # Noise
        noise = (
            self.noise_amplitude
            * (self.dt**0.5)
            * torch.randn(self.N, dtype=torch.float64, device=self._device)
        )

        # Euler-Maruyama step
        dtheta = self._omega + coupling + field_term
        theta_new = theta_t + dtheta * self.dt + noise

        theta_new = theta_new % (2.0 * torch.pi)

        # Transfer back to NumPy
        theta_np = theta_new.cpu().numpy()

        new_state = UPDEState(
            theta=theta_np,
            t=state.t + self.dt,
            step_count=state.step_count + 1,
        )
        new_state.compute_order_parameter()
        return new_state

    def _step_n_torch(self, state: UPDEState, n: int) -> UPDEState:
        """N steps entirely on GPU (single transfer)."""
        theta_t = torch.tensor(state.theta, dtype=torch.float64, device=self._device)

        t = state.t
        for _ in range(n):
            phase_diff = theta_t.unsqueeze(0) - theta_t.unsqueeze(1)
            sin_diff = torch.sin(phase_diff)
            coupling = torch.sum(self._knm * sin_diff, dim=1)
            field_term = self.field_pressure * torch.cos(theta_t)
            noise = (
                self.noise_amplitude
                * (self.dt**0.5)
                * torch.randn(self.N, dtype=torch.float64, device=self._device)
            )
            dtheta = self._omega + coupling + field_term
            theta_t = theta_t + dtheta * self.dt + noise
            theta_t = theta_t % (2.0 * torch.pi)
            t += self.dt

        theta_np = theta_t.cpu().numpy()
        new_state = UPDEState(
            theta=theta_np,
            t=t,
            step_count=state.step_count + n,
        )
        new_state.compute_order_parameter()
        return new_state

    def _step_numpy(self, state: UPDEState) -> UPDEState:
        """CPU fallback using NumPy (same as UPDEStepper)."""
        theta = state.theta
        if not np.all(np.isfinite(theta)):
            raise ValueError(
                "TorchUPDEStepper._step_numpy: input theta contains NaN or Inf"
            )

        np.subtract(theta[np.newaxis, :], theta[:, np.newaxis], out=self._phase_diff)
        np.sin(self._phase_diff, out=self._sin_diff)
        coupling = np.sum(self._knm_np * self._sin_diff, axis=1)

        field_term = self.field_pressure * np.cos(theta)
        noise = self.noise_amplitude * np.sqrt(self.dt) * np.random.randn(self.N)

        dtheta = self._omega_np + coupling + field_term
        theta_new = theta + dtheta * self.dt + noise

        theta_new = np.mod(theta_new, 2.0 * np.pi)

        new_state = UPDEState(
            theta=theta_new,
            t=state.t + self.dt,
            step_count=state.step_count + 1,
        )
        new_state.compute_order_parameter()
        return new_state
