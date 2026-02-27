# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Exception Hierarchy
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Structured exception hierarchy for Director-Class AI.

All library-specific exceptions descend from ``DirectorAIError`` so
callers can catch the entire family with a single except clause.
"""


class DirectorAIError(Exception):
    """Base exception for all Director-Class AI errors."""


class CoherenceError(DirectorAIError):
    """Raised when the coherence pipeline encounters a fatal scoring error."""


class KernelHaltError(DirectorAIError):
    """Raised when the safety kernel triggers an emergency halt."""


class GeneratorError(DirectorAIError):
    """Raised when candidate generation fails unrecoverably."""


class ValidationError(DirectorAIError):
    """Raised for invalid inputs (prompts, parameters, configs)."""


class DependencyError(DirectorAIError):
    """Raised when an optional dependency is missing or broken."""


class PhysicsError(DirectorAIError):
    """Raised when the research physics modules encounter a fatal error."""


class NumericalError(PhysicsError):
    """Raised when a numerical computation produces NaN/Inf."""


class HallucinationError(DirectorAIError):
    """Raised when a guarded LLM response fails coherence scoring."""

    def __init__(self, query: str, response: str, score):
        self.query = query
        self.response = response
        self.score = score
        super().__init__(
            f"Hallucination detected (coherence={score.score:.3f}): {response[:100]}"
        )
