# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Feedback loop detection for EU AI Act Article 15(4) compliance.

Article 15(4) requires: "eliminate or reduce the risk of possibly
biased outputs influencing input for future operations (feedback loops)."

Detects when an AI system's previous outputs reappear as inputs,
creating potential self-reinforcement cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["FeedbackLoopAlert", "FeedbackLoopDetector"]


def _trigram_set(text: str) -> set[str]:
    """Extract character trigrams for fuzzy matching."""
    text = text.lower().strip()
    if len(text) < 3:
        return {text}
    return {text[i : i + 3] for i in range(len(text) - 2)}


def _jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class FeedbackLoopAlert:
    """Alert when a feedback loop is detected."""

    input_text: str
    matched_output: str
    similarity: float  # 0-1, how closely the input matches a previous output
    output_timestamp: float  # when the matched output was generated
    severity: str  # "low", "medium", "high"


@dataclass
class FeedbackLoopDetector:
    """Detect when AI outputs feed back into inputs.

    Maintains a buffer of recent outputs. When a new input arrives,
    checks if it contains substantial overlap with any previous output.

    Parameters
    ----------
    similarity_threshold : float
        Minimum trigram Jaccard similarity to trigger an alert (default 0.5).
    max_buffer_size : int
        Maximum number of recent outputs to track (default 1000).
    min_text_length : int
        Minimum text length to consider for matching (default 20 chars).
    """

    similarity_threshold: float = 0.5
    max_buffer_size: int = 1000
    min_text_length: int = 20
    _outputs: list[tuple[str, float, set[str]]] = field(
        default_factory=list, repr=False
    )

    def record_output(self, text: str, timestamp: float) -> None:
        """Record an AI-generated output for future matching."""
        if len(text) < self.min_text_length:
            return
        trigrams = _trigram_set(text)
        self._outputs.append((text, timestamp, trigrams))
        if len(self._outputs) > self.max_buffer_size:
            self._outputs.pop(0)

    def check_input(self, text: str) -> FeedbackLoopAlert | None:
        """Check if a new input matches any previous output.

        Returns an alert if the input appears to contain text from a
        previous AI output, or None if no match is found.
        """
        if len(text) < self.min_text_length:
            return None

        input_trigrams = _trigram_set(text)
        best_sim = 0.0
        best_output = ""
        best_ts = 0.0

        for output_text, ts, output_trigrams in self._outputs:
            sim = _jaccard_similarity(input_trigrams, output_trigrams)
            if sim > best_sim:
                best_sim = sim
                best_output = output_text
                best_ts = ts

        if best_sim < self.similarity_threshold:
            return None

        if best_sim > 0.8:
            severity = "high"
        elif best_sim > 0.6:
            severity = "medium"
        else:
            severity = "low"

        return FeedbackLoopAlert(
            input_text=text,
            matched_output=best_output,
            similarity=best_sim,
            output_timestamp=best_ts,
            severity=severity,
        )

    def check_and_record(
        self, input_text: str, output_text: str, timestamp: float
    ) -> FeedbackLoopAlert | None:
        """Check input for feedback loops, then record the output.

        Convenience method for use in a scoring pipeline.
        """
        alert = self.check_input(input_text)
        self.record_output(output_text, timestamp)
        return alert

    @property
    def buffer_size(self) -> int:
        """Number of outputs currently tracked."""
        return len(self._outputs)
