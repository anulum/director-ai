# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Self-Improving Guard Loop Gate

"""Reviewed-feedback proposal gate for self-improving guardrail updates."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

from director_ai.core.guard_control import GuardDecision, RiskEnvelope
from director_ai.core.safety_event import SafetyEvent

from .feedback import FeedbackEvent, FeedbackStore

__all__ = [
    "GuardLoopProposal",
    "ReviewedFeedbackManifest",
    "SelfImprovingGuardLoop",
]

ProposalType = Literal["calibration_update", "lora_training_job"]


@dataclass(frozen=True)
class ReviewedFeedbackManifest:
    """Tenant-safe manifest for reviewed feedback used by improvement proposals."""

    manifest_id: str
    source_ref: str
    event_count: int
    label_counts: Mapping[str, int]
    reviewer_ids: Sequence[str]
    event_refs: Sequence[str]

    def __post_init__(self) -> None:
        if not self.manifest_id.strip():
            raise ValueError("manifest_id is required")
        if not self.source_ref.strip():
            raise ValueError("source_ref is required")
        if self.event_count <= 0:
            raise ValueError("event_count must be positive")
        object.__setattr__(self, "label_counts", dict(self.label_counts))
        object.__setattr__(self, "reviewer_ids", tuple(map(str, self.reviewer_ids)))
        object.__setattr__(self, "event_refs", tuple(map(str, self.event_refs)))

    def to_dict(self) -> dict[str, Any]:
        """Serialise without prompt, response, or private evidence text."""
        return {
            "manifest_id": self.manifest_id,
            "source_ref": self.source_ref,
            "event_count": self.event_count,
            "label_counts": dict(self.label_counts),
            "reviewer_ids": list(self.reviewer_ids),
            "event_refs": list(self.event_refs),
        }


@dataclass(frozen=True)
class GuardLoopProposal:
    """Calibration or training proposal that cannot apply itself."""

    proposal_id: str
    proposal_type: ProposalType
    manifest: ReviewedFeedbackManifest
    rollback_id: str
    guard_decision: GuardDecision
    payload: Mapping[str, Any]
    submitted: bool = False
    promotion_status: str = "proposed"
    approved: bool = False
    approval_id: str = ""

    def __post_init__(self) -> None:
        if not self.proposal_id.strip():
            raise ValueError("proposal_id is required")
        if self.proposal_type not in {"calibration_update", "lora_training_job"}:
            raise ValueError(f"unsupported proposal_type {self.proposal_type!r}")
        if not self.rollback_id.strip():
            raise ValueError("rollback_id is required")
        if self.submitted:
            raise ValueError("guard-loop proposals must not submit jobs directly")
        if self.promotion_status != "proposed":
            raise ValueError("promotion_status must remain 'proposed'")
        if self.approved and not self.approval_id.strip():
            raise ValueError("approval_id is required for approved proposals")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        """Return a tenant-safe audit payload."""
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "manifest": self.manifest.to_dict(),
            "rollback_id": self.rollback_id,
            "guard_decision": self.guard_decision.to_dict(),
            "payload": dict(self.payload),
            "submitted": self.submitted,
            "promotion_status": self.promotion_status,
            "approved": self.approved,
            "approval_id": self.approval_id,
        }

    def to_safety_event(
        self,
        *,
        hook_id: str,
        hook_scope: str = "agent",
        request_id: str = "",
        tenant_id: str = "",
        latency_ms: float | None = None,
    ) -> SafetyEvent:
        """Convert the proposal guard decision to a tenant-safe event."""
        return self.guard_decision.to_safety_event(
            hook_id=hook_id,
            hook_scope=hook_scope,
            request_id=request_id,
            tenant_id=tenant_id,
            latency_ms=latency_ms,
        )


class SelfImprovingGuardLoop:
    """Create reviewed-feedback proposals without self-applying updates."""

    def __init__(
        self,
        *,
        store: FeedbackStore,
        risk_envelope: RiskEnvelope,
        policy_id: str,
    ) -> None:
        if not policy_id.strip():
            raise ValueError("policy_id is required")
        self._store = store
        self._risk_envelope = risk_envelope
        self._policy_id = policy_id

    def build_manifest(self, *, source_ref: str) -> ReviewedFeedbackManifest:
        """Build a tenant-safe manifest from reviewed feedback events."""
        if not source_ref.strip():
            raise ValueError("source_ref is required")
        events = tuple(self._store.iter_all())
        if not events:
            raise ValueError("feedback store must contain reviewed events")
        event_refs: list[str] = []
        reviewer_ids: set[str] = set()
        labels: Counter[str] = Counter()
        for index, event in enumerate(events):
            event_id = _required_metadata(event, "event_id", index)
            reviewer_id = _required_metadata(event, "reviewer_id", index)
            event_refs.append(event_id)
            reviewer_ids.add(reviewer_id)
            labels[event.label] += 1
        return ReviewedFeedbackManifest(
            manifest_id=f"reviewed-feedback-{uuid4()}",
            source_ref=source_ref,
            event_count=len(events),
            label_counts=dict(sorted(labels.items())),
            reviewer_ids=tuple(sorted(reviewer_ids)),
            event_refs=tuple(event_refs),
        )

    def propose_calibration_update(
        self,
        *,
        source_ref: str,
        current_threshold: float,
        candidate_threshold: float,
        confidence_low: float,
        confidence_high: float,
        rollback_id: str,
        min_feedback: int = 32,
        max_interval_width: float = 0.1,
    ) -> GuardLoopProposal:
        """Propose a calibrate-only threshold update for human approval."""
        _validate_unit_interval("current_threshold", current_threshold)
        _validate_unit_interval("candidate_threshold", candidate_threshold)
        _validate_interval(confidence_low, confidence_high)
        _validate_positive_int("min_feedback", min_feedback)
        _validate_unit_interval("max_interval_width", max_interval_width)
        manifest = self.build_manifest(source_ref=source_ref)
        interval_width = confidence_high - confidence_low
        if manifest.event_count < min_feedback:
            decision = "warn"
            reason = "self_improvement_insufficient_feedback"
            explanation = "Reviewed feedback is insufficient for promotion."
            risk_score = 1.0
        elif interval_width > max_interval_width:
            decision = "warn"
            reason = "self_improvement_interval_too_wide"
            explanation = "Calibration interval is too wide for promotion."
            risk_score = min(1.0, interval_width)
        else:
            decision = "allow"
            reason = "self_improvement_calibration_ready"
            explanation = "Reviewed feedback supports a calibration proposal."
            risk_score = abs(candidate_threshold - current_threshold)
        return self._proposal(
            proposal_type="calibration_update",
            manifest=manifest,
            rollback_id=rollback_id,
            decision=decision,
            reason=reason,
            explanation=explanation,
            risk_score=risk_score,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            payload={
                "current_threshold": current_threshold,
                "candidate_threshold": candidate_threshold,
                "calibration_size": manifest.event_count,
                "max_interval_width": max_interval_width,
            },
        )

    def propose_lora_job(
        self,
        *,
        source_ref: str,
        dataset_uri: str,
        base_model_ref: str,
        rollback_id: str,
        heldout_score: float,
        baseline_score: float,
        min_improvement: float = 0.0,
        min_feedback: int = 32,
    ) -> GuardLoopProposal:
        """Propose, but do not submit, a LoRA training job."""
        _reject_embedded_credentials(dataset_uri)
        if not dataset_uri.strip():
            raise ValueError("dataset_uri is required")
        if not base_model_ref.strip():
            raise ValueError("base_model_ref is required")
        _validate_unit_interval("heldout_score", heldout_score)
        _validate_unit_interval("baseline_score", baseline_score)
        if min_improvement < 0.0 or not math.isfinite(min_improvement):
            raise ValueError("min_improvement must be finite and non-negative")
        _validate_positive_int("min_feedback", min_feedback)
        manifest = self.build_manifest(source_ref=source_ref)
        improvement = heldout_score - baseline_score
        if manifest.event_count < min_feedback:
            decision = "warn"
            reason = "self_improvement_insufficient_feedback"
            explanation = "Reviewed feedback is insufficient for training proposal."
            risk_score = 1.0
        elif improvement < min_improvement:
            decision = "warn"
            reason = "self_improvement_heldout_regression"
            explanation = "Held-out score does not clear the promotion gate."
            risk_score = min(1.0, min_improvement - improvement)
        else:
            decision = "allow"
            reason = "self_improvement_training_ready"
            explanation = "Reviewed feedback supports a training-job proposal."
            risk_score = max(0.0, 1.0 - heldout_score)
        return self._proposal(
            proposal_type="lora_training_job",
            manifest=manifest,
            rollback_id=rollback_id,
            decision=decision,
            reason=reason,
            explanation=explanation,
            risk_score=risk_score,
            confidence_low=min(baseline_score, heldout_score),
            confidence_high=max(baseline_score, heldout_score),
            payload={
                "dataset_uri": dataset_uri,
                "base_model_ref": base_model_ref,
                "heldout_score": heldout_score,
                "baseline_score": baseline_score,
                "min_improvement": min_improvement,
            },
        )

    def approve(
        self,
        proposal: GuardLoopProposal,
        *,
        approval_id: str,
    ) -> GuardLoopProposal:
        """Approve a proposal without applying it."""
        if proposal.guard_decision.decision != "allow":
            raise PermissionError(
                f"cannot approve proposal with decision "
                f"{proposal.guard_decision.decision!r}"
            )
        if not approval_id.strip():
            raise ValueError("approval_id is required")
        return replace(proposal, approved=True, approval_id=approval_id)

    def release(self, proposal: GuardLoopProposal) -> dict[str, Any]:
        """Return proposal payload only after approval; never apply it."""
        if not proposal.approved:
            raise PermissionError("guard-loop proposal is not approved")
        if proposal.guard_decision.decision != "allow":
            raise PermissionError(
                f"cannot release proposal with decision "
                f"{proposal.guard_decision.decision!r}"
            )
        return dict(proposal.payload)

    def _proposal(
        self,
        *,
        proposal_type: ProposalType,
        manifest: ReviewedFeedbackManifest,
        rollback_id: str,
        decision: str,
        reason: str,
        explanation: str,
        risk_score: float,
        confidence_low: float,
        confidence_high: float,
        payload: Mapping[str, Any],
    ) -> GuardLoopProposal:
        guard_decision = GuardDecision(
            decision=decision,
            risk_score=max(0.0, min(1.0, risk_score)),
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            policy_id=self._policy_id,
            reason=reason,
            tenant_safe_explanation=explanation,
            evidence_refs=manifest.event_refs,
            verifier_signals=(),
            risk_envelope=self._risk_envelope,
            attributes={
                "proposal_type": proposal_type,
                "manifest_id": manifest.manifest_id,
            },
        )
        return GuardLoopProposal(
            proposal_id=f"guard-loop-{uuid4()}",
            proposal_type=proposal_type,
            manifest=manifest,
            rollback_id=rollback_id,
            guard_decision=guard_decision,
            payload=payload,
        )


def _required_metadata(event: FeedbackEvent, key: str, index: int) -> str:
    value = event.metadata.get(key, "")
    if not value.strip():
        raise ValueError(f"feedback event {index} is missing {key}")
    return value


def _validate_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _validate_interval(low: float, high: float) -> None:
    _validate_unit_interval("confidence_low", low)
    _validate_unit_interval("confidence_high", high)
    if low > high:
        raise ValueError("confidence_low must be <= confidence_high")


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _reject_embedded_credentials(uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.netloc and "@" in parsed.netloc:
        raise ValueError("dataset_uri must not contain embedded credentials")
