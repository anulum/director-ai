# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — guard decision contracts

"""Tenant-safe decision records shared by advanced guard subsystems."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from director_ai.core.safety_event import SafetyEvent

_DECISIONS = frozenset({"allow", "warn", "halt", "block"})
_ACTION_CATEGORIES = frozenset(
    {
        "text",
        "tool",
        "code",
        "physical",
        "training",
        "inference_steering",
        "multimodal",
        "sustainability",
        "identity",
    }
)
_REVERSIBILITY = frozenset({"reversible", "costly", "irreversible"})
_DOMAINS = frozenset(
    {
        "general",
        "regulated",
        "physical",
        "financial",
        "medical",
        "legal",
        "security",
    }
)
_MODALITIES = frozenset(
    {
        "audio",
        "code",
        "identity",
        "image",
        "physical",
        "policy",
        "sustainability",
        "text",
        "video",
    }
)
_BLOCKED_ATTRIBUTE_PARTS = (
    "credential",
    "image",
    "password",
    "private_key",
    "prompt",
    "raw",
    "secret",
    "sensor",
    "token",
)


@dataclass(frozen=True)
class RiskEnvelope:
    """Risk category and calibrated thresholds for one proposed action."""

    action_category: str
    reversibility: str
    domain: str
    calibrated_threshold: float
    no_go_threshold: float

    def __post_init__(self) -> None:
        if self.action_category not in _ACTION_CATEGORIES:
            raise ValueError(f"unsupported action_category {self.action_category!r}")
        if self.reversibility not in _REVERSIBILITY:
            raise ValueError(f"unsupported reversibility {self.reversibility!r}")
        if self.domain not in _DOMAINS:
            raise ValueError(f"unsupported domain {self.domain!r}")
        _validate_unit_interval("calibrated_threshold", self.calibrated_threshold)
        _validate_unit_interval("no_go_threshold", self.no_go_threshold)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "action_category": self.action_category,
            "reversibility": self.reversibility,
            "domain": self.domain,
            "calibrated_threshold": self.calibrated_threshold,
            "no_go_threshold": self.no_go_threshold,
        }


@dataclass(frozen=True)
class VerifierSignal:
    """One normalized signal emitted by a verifier or policy subsystem."""

    verifier: str
    modality: str
    score: float
    verdict: str
    confidence_low: float
    confidence_high: float
    evidence_refs: Sequence[str] = field(default_factory=tuple)
    latency_ms: float = 0.0
    failure_mode: str = ""

    def __post_init__(self) -> None:
        if not self.verifier.strip():
            raise ValueError("verifier is required")
        if self.modality not in _MODALITIES:
            raise ValueError(f"unsupported modality {self.modality!r}")
        if not self.verdict.strip():
            raise ValueError("verdict is required")
        _validate_unit_interval("score", self.score)
        _validate_confidence_interval(self.confidence_low, self.confidence_high)
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, tenant-safe representation."""
        return {
            "verifier": self.verifier,
            "modality": self.modality,
            "score": self.score,
            "verdict": self.verdict,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "evidence_refs": list(self.evidence_refs),
            "latency_ms": self.latency_ms,
            "failure_mode": self.failure_mode,
        }


@dataclass(frozen=True)
class GuardDecision:
    """Normalized decision shared by future guard-control modules."""

    decision: str
    risk_score: float
    confidence_low: float
    confidence_high: float
    policy_id: str
    reason: str
    tenant_safe_explanation: str
    evidence_refs: Sequence[str]
    verifier_signals: Sequence[VerifierSignal]
    risk_envelope: RiskEnvelope
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in _DECISIONS:
            raise ValueError(f"unsupported decision {self.decision!r}")
        _validate_unit_interval("risk_score", self.risk_score)
        _validate_confidence_interval(self.confidence_low, self.confidence_high)
        if not self.policy_id.strip():
            raise ValueError("policy_id is required")
        if not self.reason.strip():
            raise ValueError("reason is required")
        if not self.tenant_safe_explanation.strip():
            raise ValueError("tenant_safe_explanation is required")
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))
        object.__setattr__(self, "verifier_signals", tuple(self.verifier_signals))
        object.__setattr__(
            self,
            "attributes",
            _tenant_safe_attributes(self.attributes),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise without raw tenant payload fields."""
        return {
            "decision": self.decision,
            "risk_score": self.risk_score,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "policy_id": self.policy_id,
            "reason": self.reason,
            "tenant_safe_explanation": self.tenant_safe_explanation,
            "evidence_refs": list(self.evidence_refs),
            "verifier_signals": [signal.to_dict() for signal in self.verifier_signals],
            "risk_envelope": self.risk_envelope.to_dict(),
            "attributes": dict(self.attributes),
        }

    def to_safety_event(
        self,
        *,
        hook_id: str,
        hook_scope: str,
        request_id: str = "",
        tenant_id: str = "",
        latency_ms: float | None = None,
    ) -> SafetyEvent:
        """Convert this decision into the shared tenant-safe event schema."""
        attrs = {
            "policy_id": self.policy_id,
            "risk_domain": self.risk_envelope.domain,
            "action_category": self.risk_envelope.action_category,
            "reversibility": self.risk_envelope.reversibility,
            **dict(self.attributes),
        }
        return SafetyEvent.from_policy_decision(
            hook_id=hook_id,
            hook_scope=hook_scope,
            policy_decision=self.decision,
            halt_reason=self.reason,
            tenant_safe_explanation=self.tenant_safe_explanation,
            request_id=request_id,
            tenant_id=tenant_id,
            threshold=self.risk_envelope.calibrated_threshold,
            observed_score=self.risk_score,
            latency_ms=latency_ms,
            evidence_refs=tuple(self.evidence_refs),
            attributes=attrs,
        )


def _validate_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _validate_confidence_interval(low: float, high: float) -> None:
    _validate_unit_interval("confidence_low", low)
    _validate_unit_interval("confidence_high", high)
    if low > high:
        raise ValueError("confidence_low must be <= confidence_high")


def _tenant_safe_attributes(attributes: Mapping[str, str]) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in attributes.items():
        key_s = str(key)
        lowered = key_s.lower()
        if any(part in lowered for part in _BLOCKED_ATTRIBUTE_PARTS):
            continue
        safe[key_s] = str(value)
    return safe
