# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Safety Kernel (Output Gate)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging


class SafetyKernel:
    """
    Software safety gate for the output stream.

    Sits between the model output buffer and the network interface.
    Monitors the coherence score in real-time and has the
    authority to sever the token stream if coherence drops below the
    hard safety limit.
    """

    def __init__(self, hard_limit: float = 0.5):
        self.hard_limit = hard_limit
        self.logger = logging.getLogger("DirectorAI.Kernel")
        self.is_active = True

    def stream_output(self, token_generator, coherence_callback):
        """
        Emit output tokens while monitoring coherence in real-time.

        Args:
            token_generator: Iterator yielding tokens (words).
            coherence_callback: Callable(token) -> current coherence score.

        Returns:
            Assembled output string, or an interrupt message if halted.
        """
        output_buffer = []

        for token in token_generator:
            if not isinstance(token, str):
                token = str(token)

            try:
                current_score = coherence_callback(token)
                current_score = float(current_score)
            except Exception as exc:
                self.logger.error(
                    "Coherence callback raised %s — score=0", exc
                )
                current_score = 0.0

            if current_score < self.hard_limit:
                self.emergency_stop()
                return "[KERNEL INTERRUPT: COHERENCE LIMIT EXCEEDED]"

            output_buffer.append(token)

        return "".join(output_buffer)

    def emergency_stop(self):
        """Halt the inference engine."""
        self.logger.critical(">>> SAFETY KERNEL ACTIVATED: INFERENCE HALTED <<<")
        self.is_active = False

    def reactivate(self):
        """Re-enable the kernel after a previous halt."""
        self.is_active = True
        self.logger.info("Safety kernel reactivated")
