# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — reviewed defence update pipeline

"""Reviewed promotion gate across self-evolution and adversarial mining."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from director_ai.core.continual_adversarial import EvolveReport
from director_ai.core.self_evolving import GuardLoopProposal

from .registry import Defense, DefenseRegistry, DefenseSnapshot


@dataclass(frozen=True)
class DefenseUpdateReport:
    """Tenant-safe report for one reviewed defence promotion."""

    snapshot: DefenseSnapshot
    proposal_id: str
    suite_version: int
    adversarial_case_count: int
    promoted: bool
    metadata: dict[str, str] = field(default_factory=dict)


class DefenseUpdatePipeline:
    """Promote a candidate defence only after review and adversarial gates.

    The pipeline is deliberately narrow: it never trains a model, mines
    failures, or mutates proposals. It joins the already-reviewed
    ``GuardLoopProposal`` with a ``ContinualEngine`` report and then performs one
    atomic registry promotion if every gate passes.
    """

    def __init__(
        self,
        *,
        registry: DefenseRegistry,
        min_adversarial_cases: int = 1,
        min_holdout_improvement: float = 0.0,
    ) -> None:
        if min_adversarial_cases <= 0:
            raise ValueError("min_adversarial_cases must be positive")
        if not math.isfinite(min_holdout_improvement) or min_holdout_improvement < 0.0:
            raise ValueError("min_holdout_improvement must be finite and non-negative")
        self._registry = registry
        self._min_adversarial_cases = min_adversarial_cases
        self._min_holdout_improvement = min_holdout_improvement

    def review_and_promote(
        self,
        *,
        proposal: GuardLoopProposal,
        evolve_report: EvolveReport,
        defense: Defense,
        version: int,
        label: str,
        baseline_score: float,
        candidate_score: float,
    ) -> DefenseUpdateReport:
        """Validate review, adversarial evidence, and holdout score, then promote."""
        _require_approved_proposal(proposal)
        _validate_score("baseline_score", baseline_score)
        _validate_score("candidate_score", candidate_score)
        if version <= 0:
            raise ValueError("version must be positive")
        if not label.strip():
            raise ValueError("label is required")
        if evolve_report.adversarial_case_count < self._min_adversarial_cases:
            raise ValueError(
                "adversarial_case_count below promotion gate: "
                f"{evolve_report.adversarial_case_count} < "
                f"{self._min_adversarial_cases}"
            )
        holdout_delta = candidate_score - baseline_score
        if holdout_delta < self._min_holdout_improvement:
            raise ValueError(
                "candidate holdout score does not clear the promotion gate: "
                f"{holdout_delta:.6f} < {self._min_holdout_improvement:.6f}"
            )
        metadata = _promotion_metadata(
            proposal=proposal,
            evolve_report=evolve_report,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            holdout_delta=holdout_delta,
        )
        snapshot = self._registry.promote(
            defense=defense,
            version=version,
            label=label,
            metadata=metadata,
        )
        return DefenseUpdateReport(
            snapshot=snapshot,
            proposal_id=proposal.proposal_id,
            suite_version=evolve_report.version.version,
            adversarial_case_count=evolve_report.adversarial_case_count,
            promoted=True,
            metadata=dict(metadata),
        )


def _require_approved_proposal(proposal: GuardLoopProposal) -> None:
    if proposal.guard_decision.decision != "allow":
        raise PermissionError("proposal guard decision must be allow")
    if not proposal.approved or not proposal.approval_id.strip():
        raise PermissionError("proposal must be approved before promotion")


def _promotion_metadata(
    *,
    proposal: GuardLoopProposal,
    evolve_report: EvolveReport,
    baseline_score: float,
    candidate_score: float,
    holdout_delta: float,
) -> dict[str, str]:
    return {
        "proposal_id": proposal.proposal_id,
        "proposal_type": proposal.proposal_type,
        "approval_id": proposal.approval_id,
        "manifest_id": proposal.manifest.manifest_id,
        "manifest_event_count": str(proposal.manifest.event_count),
        "rollback_id": proposal.rollback_id,
        "suite_version": str(evolve_report.version.version),
        "adversarial_case_count": str(evolve_report.adversarial_case_count),
        "mined_pattern_count": str(evolve_report.mined_pattern_count),
        "promotion_reason": evolve_report.promotion_reason,
        "baseline_score": f"{baseline_score:.6f}",
        "candidate_score": f"{candidate_score:.6f}",
        "holdout_delta": f"{holdout_delta:.6f}",
    }


def _validate_score(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
