# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Halt Monitor

"""Halt monitor and safety kernel: the coherence-gated token streaming core."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

__all__ = ["HaltMonitor", "HaltOutput", "SafetyKernel"]

_HaltReason = Literal[
    "",
    "kernel_inactive",
    "total_timeout",
    "callback_timeout",
    "token_timeout",
    "hard_limit",
]


@dataclass(frozen=True, slots=True)
class HaltOutput:
    """The output, stop reason and last score observed by the monitor."""

    output: str
    halt_reason: _HaltReason = ""
    last_score: float | None = None

    @property
    def halted(self) -> bool:
        """Return whether this call stopped before completing its output."""
        return bool(self.halt_reason)


class HaltMonitor:
    """Threshold-based halt gate for the token output stream.

    Checks each token's coherence score against ``hard_limit``.
    Halts the stream if the score drops below the floor.

    Parameters
    ----------
    hard_limit : float — coherence floor (default 0.5).
    on_halt : callable | None — invoked with score on halt.
    token_timeout : float — max seconds per token (0 = disabled).
    total_timeout : float — max seconds for entire stream (0 = disabled).

    """

    def __init__(
        self,
        hard_limit: float = 0.5,
        on_halt: Callable[[float], None] | None = None,
        token_timeout: float = 0.0,
        total_timeout: float = 0.0,
    ) -> None:
        """Configure the coherence floor, halt callback and optional timeouts.

        Parameters
        ----------
        hard_limit : float
            Inclusive lower passing score in the interval [0, 1].
        on_halt : callable, optional
            Receives the score that triggered a hard-limit halt.
        token_timeout : float
            Maximum callback duration in seconds; zero disables the limit.
        total_timeout : float
            Maximum stream duration in seconds; zero disables the limit.

        Raises
        ------
        ValueError
            If the coherence floor is outside [0, 1].
        """
        if not (0.0 <= hard_limit <= 1.0):
            raise ValueError(f"hard_limit must be in [0, 1], got {hard_limit}")
        self.hard_limit = hard_limit
        self.on_halt = on_halt
        self.token_timeout = token_timeout
        self.total_timeout = total_timeout
        self.logger = logging.getLogger("DirectorAI.Kernel")
        self._active = threading.Event()
        self._active.set()

    def stream_output(
        self,
        token_generator: Iterable[str],
        coherence_callback: Callable[[str], float],
    ) -> str:
        """Return the assembled output or the original halt message."""
        return self.stream_output_result(token_generator, coherence_callback).output

    def stream_output_result(
        self,
        token_generator: Iterable[str],
        coherence_callback: Callable[[str], float],
    ) -> HaltOutput:
        """Emit output under monitoring and return this call's verdict.

        Parameters
        ----------
        token_generator : iterable of str
            Tokens in generation order.
        coherence_callback : callable
            Receives the accumulated output including the proposed token and
            returns a coherence score. Nonfinite results become zero before
            comparison or delivery to the halt callback.

        Returns
        -------
        HaltOutput
            Assembled output, or the halt message with its reason. The
            score is absent if this call had no successful score check.
            An inactive gate refuses before scoring. Use ``reactivate``
            to re-arm it.
            Positive token and total timeouts are measured in seconds.
        """
        output_buffer: list[str] = []
        stream_start = time.monotonic()
        last_score = None

        for token in token_generator:
            if not self.is_active:
                return HaltOutput(
                    "[HALT: KERNEL INACTIVE]", "kernel_inactive", last_score
                )
            if self.total_timeout > 0:
                elapsed = time.monotonic() - stream_start
                if elapsed > self.total_timeout:
                    self.emergency_stop()
                    return HaltOutput(
                        "[HALT: TOTAL TIMEOUT EXCEEDED]", "total_timeout", last_score
                    )

            token_start = time.monotonic()
            accumulated = "".join(output_buffer) + token
            try:
                current_score = coherence_callback(accumulated)
            except (TimeoutError, OSError):
                self.emergency_stop()
                return HaltOutput(
                    "[HALT: CALLBACK TIMEOUT]", "callback_timeout", last_score
                )
            current_score = current_score if math.isfinite(current_score) else 0.0
            last_score = current_score
            token_elapsed = time.monotonic() - token_start

            if self.token_timeout > 0 and token_elapsed > self.token_timeout:
                self.emergency_stop()
                return HaltOutput(
                    "[HALT: TOKEN TIMEOUT EXCEEDED]", "token_timeout", current_score
                )

            if current_score < self.hard_limit:
                self.emergency_stop()
                if self.on_halt:
                    self.on_halt(current_score)
                return HaltOutput(
                    "[HALT: COHERENCE BELOW THRESHOLD]", "hard_limit", current_score
                )

            output_buffer.append(token)

        return HaltOutput("".join(output_buffer), last_score=last_score)

    def emergency_stop(self) -> None:
        """Halt the output stream."""
        self.logger.critical(">>> HALT MONITOR: INFERENCE HALTED <<<")
        self._active.clear()

    @property
    def is_active(self) -> bool:
        """Report whether the monitor is armed (not emergency-stopped)."""
        return self._active.is_set()

    def reactivate(self) -> None:
        """Re-arm the monitor after an emergency stop."""
        self._active.set()
        self.logger.info("Halt monitor reactivated")


SafetyKernel = HaltMonitor
