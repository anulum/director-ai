# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Halt Monitor

import logging
import threading
import time

__all__ = ["HaltMonitor", "SafetyKernel"]


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
        on_halt=None,
        token_timeout: float = 0.0,
        total_timeout: float = 0.0,
    ):
        if not (0.0 <= hard_limit <= 1.0):
            raise ValueError(f"hard_limit must be in [0, 1], got {hard_limit}")
        self.hard_limit = hard_limit
        self.on_halt = on_halt
        self.token_timeout = token_timeout
        self.total_timeout = total_timeout
        self.logger = logging.getLogger("DirectorAI.Kernel")
        self._active = threading.Event()
        self._active.set()

    def stream_output(self, token_generator, coherence_callback):
        """Emit output tokens while monitoring coherence in real-time.

        Returns assembled output string, or an interrupt message if halted.
        Respects token_timeout and total_timeout when > 0.
        """
        output_buffer: list[str] = []
        stream_start = time.monotonic()

        for token in token_generator:
            if self.total_timeout > 0:
                elapsed = time.monotonic() - stream_start
                if elapsed > self.total_timeout:
                    self.emergency_stop()
                    return "[HALT: TOTAL TIMEOUT EXCEEDED]"

            token_start = time.monotonic()
            accumulated = "".join(output_buffer) + token
            current_score = coherence_callback(accumulated)
            token_elapsed = time.monotonic() - token_start

            if self.token_timeout > 0 and token_elapsed > self.token_timeout:
                self.emergency_stop()
                return "[HALT: TOKEN TIMEOUT EXCEEDED]"

            if current_score < self.hard_limit:
                self.emergency_stop()
                if self.on_halt:
                    self.on_halt(current_score)
                return "[HALT: COHERENCE BELOW THRESHOLD]"

            output_buffer.append(token)

        return "".join(output_buffer)

    def emergency_stop(self):
        """Halt the output stream."""
        self.logger.critical(">>> HALT MONITOR: INFERENCE HALTED <<<")
        self._active.clear()

    @property
    def is_active(self) -> bool:
        return self._active.is_set()

    def reactivate(self):
        """Re-arm the monitor after an emergency stop."""
        self._active.set()
        self.logger.info("Halt monitor reactivated")


SafetyKernel = HaltMonitor
