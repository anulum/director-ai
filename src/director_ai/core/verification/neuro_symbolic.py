# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Neuro-Symbolic Verification

"""Fusion layer for neural scores and symbolic verifiers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..formal_verification import ReasoningStep, ReasoningVerdict, ReasoningVerifier
from .numeric_verifier import NumericVerificationResult, verify_numeric

__all__ = [
    "NeuroSymbolicVerificationResult",
    "NeuroSymbolicVerifier",
    "NeuroSymbolicVerifierInput",
]


def _unit_score(value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("neural_score must be finite and in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class NeuroSymbolicVerifierInput:
    """Input bundle for neural + symbolic verification."""

    text: str
    neural_score: float
    symbolic_steps: Sequence[ReasoningStep] = ()
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        if not str(self.text).strip():
            raise ValueError("text must be non-empty")
        _unit_score(self.neural_score)
        object.__setattr__(self, "symbolic_steps", tuple(self.symbolic_steps))


@dataclass(frozen=True)
class NeuroSymbolicVerificationResult:
    """Combined verification decision."""

    decision: str
    neural_score: float
    neural_accept_threshold: float
    reasons: tuple[str, ...]
    numeric_result: NumericVerificationResult | None = None
    symbolic_verdict: ReasoningVerdict | None = None
    evidence_ref: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in {"allow", "warn", "reject"}:
            raise ValueError(f"unsupported decision {self.decision!r}")
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "neural_score": self.neural_score,
            "neural_accept_threshold": self.neural_accept_threshold,
            "reasons": list(self.reasons),
            "evidence_ref": self.evidence_ref,
            "text": self.text if include_text else None,
            "numeric": _numeric_to_dict(self.numeric_result),
            "symbolic": _symbolic_to_dict(self.symbolic_verdict),
            "metadata": dict(self.metadata),
        }


class NeuroSymbolicVerifier:
    """Apply numeric and formal verifiers alongside a neural score.

    Neural scores are useful for broad semantic plausibility. Symbolic checks
    are decisive where the claim is formally checkable: numeric consistency,
    dates, probabilities, and caller-supplied logical formulas.
    """

    def __init__(
        self,
        *,
        neural_accept_threshold: float = 0.6,
        reasoning_verifier: ReasoningVerifier | None = None,
        run_numeric: bool = True,
    ) -> None:
        self.neural_accept_threshold = _unit_score(neural_accept_threshold)
        self._reasoning_verifier = reasoning_verifier or ReasoningVerifier()
        self._run_numeric = bool(run_numeric)

    def verify(
        self,
        item: NeuroSymbolicVerifierInput,
    ) -> NeuroSymbolicVerificationResult:
        reasons: list[str] = []
        numeric_result = verify_numeric(item.text) if self._run_numeric else None
        if numeric_result is not None and not numeric_result.valid:
            reasons.append("numeric_inconsistency")

        symbolic_verdict = None
        if item.symbolic_steps:
            symbolic_verdict = self._reasoning_verifier.verify(item.symbolic_steps)
            if symbolic_verdict.contradictory:
                reasons.append("symbolic_contradiction")

        if item.neural_score < self.neural_accept_threshold:
            reasons.append("neural_score_below_threshold")

        decisive = {"numeric_inconsistency", "symbolic_contradiction"}
        if any(reason in decisive for reason in reasons):
            decision = "reject"
        elif reasons:
            decision = "warn"
        else:
            decision = "allow"

        return NeuroSymbolicVerificationResult(
            decision=decision,
            neural_score=item.neural_score,
            neural_accept_threshold=self.neural_accept_threshold,
            reasons=tuple(reasons),
            numeric_result=numeric_result,
            symbolic_verdict=symbolic_verdict,
            evidence_ref=item.evidence_ref,
            text=item.text,
            metadata={
                "numeric_checked": self._run_numeric,
                "symbolic_step_count": len(item.symbolic_steps),
            },
        )


def _numeric_to_dict(result: NumericVerificationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "claims_found": result.claims_found,
        "valid": result.valid,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "issues": [
            {
                "issue_type": issue.issue_type,
                "description": issue.description,
                "severity": issue.severity,
            }
            for issue in result.issues
        ],
    }


def _symbolic_to_dict(verdict: ReasoningVerdict | None) -> dict[str, Any] | None:
    if verdict is None:
        return None
    return {
        "consistent": verdict.consistent,
        "contradictory": verdict.contradictory,
        "step_count": verdict.step_count,
        "backend": verdict.backend,
        "model": verdict.model,
    }
