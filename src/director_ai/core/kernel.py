# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Safety Kernel (Hardware Interlock)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging
import time


class SafetyKernel:
    """
    Hardware-level safety interlock for the output stream.

    Sits between the model output buffer and the network interface.
    Monitors the coherence score in real-time and has the physical
    authority to sever the token stream if coherence drops below the
    hard safety limit.

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
        self.hard_limit = hard_limit
        self.on_halt = on_halt
        self.token_timeout = token_timeout
        self.total_timeout = total_timeout
        self.logger = logging.getLogger("DirectorAI.Kernel")
        self.is_active = True

    def stream_output(self, token_generator, coherence_callback):
        """
        Emit output tokens while monitoring coherence in real-time.

        Returns assembled output string, or an interrupt message if halted.
        Respects token_timeout and total_timeout when > 0.
        """
        output_buffer = []
        stream_start = time.monotonic()

        for token in token_generator:
            if self.total_timeout > 0:
                elapsed = time.monotonic() - stream_start
                if elapsed > self.total_timeout:
                    self.emergency_stop()
                    return "[KERNEL INTERRUPT: TOTAL TIMEOUT EXCEEDED]"

            token_start = time.monotonic()
            current_score = coherence_callback(token)
            token_elapsed = time.monotonic() - token_start

            if self.token_timeout > 0 and token_elapsed > self.token_timeout:
                self.emergency_stop()
                return "[KERNEL INTERRUPT: TOKEN TIMEOUT EXCEEDED]"

            if current_score < self.hard_limit:
                self.emergency_stop()
                if self.on_halt:
                    self.on_halt(current_score)
                return "[KERNEL INTERRUPT: COHERENCE LIMIT EXCEEDED]"

            output_buffer.append(token)

        return "".join(output_buffer)

    def emergency_stop(self):
        """Physically halt the inference engine."""
        self.logger.critical(">>> SAFETY KERNEL ACTIVATED: INFERENCE HALTED <<<")
        self.is_active = False
