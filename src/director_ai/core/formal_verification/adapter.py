# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Formal Math and Code Verifier Adapter

"""Route formal formulae and generated code into shared guard decisions."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from director_ai.core.guard_control import GuardDecision, RiskEnvelope, VerifierSignal
from director_ai.core.safety_event import SafetyEvent
from director_ai.core.verification.code_verifier import verify_code

from .formula import Formula
from .verifier import ReasoningStep, ReasoningVerifier

__all__ = [
    "FormalCodeVerificationResult",
    "FormalCodeVerifierAdapter",
]

_DEFAULT_FORMAL_VERIFIER = object()


@dataclass(frozen=True)
class FormalCodeVerificationResult:
    """Tenant-safe formal/code verifier result."""

    kind: str
    evidence_ref: str
    signal: VerifierSignal
    guard_decision: GuardDecision
    sandbox: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialise without raw formula or source code text."""
        return {
            "kind": self.kind,
            "evidence_ref": self.evidence_ref,
            "signal": self.signal.to_dict(),
            "guard_decision": self.guard_decision.to_dict(),
            "sandbox": dict(self.sandbox),
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
        """Convert the guard decision to a shared safety event."""
        return self.guard_decision.to_safety_event(
            hook_id=hook_id,
            hook_scope=hook_scope,
            request_id=request_id,
            tenant_id=tenant_id,
            latency_ms=latency_ms,
        )


class FormalCodeVerifierAdapter:
    """Guard-control adapter for formal claims and generated code."""

    def __init__(
        self,
        *,
        formal_verifier: ReasoningVerifier | None | object = _DEFAULT_FORMAL_VERIFIER,
        code_verifier: Callable[..., Any] = verify_code,
        timeout_ms: float = 1000.0,
    ) -> None:
        if timeout_ms <= 0.0:
            raise ValueError("timeout_ms must be positive")
        if formal_verifier is _DEFAULT_FORMAL_VERIFIER:
            self._formal_verifier: ReasoningVerifier | None = ReasoningVerifier()
        elif formal_verifier is None:
            self._formal_verifier = None
        else:
            self._formal_verifier = cast(ReasoningVerifier, formal_verifier)
        self._code_verifier = code_verifier
        self._timeout_ms = timeout_ms

    def verify_formula(
        self,
        *,
        formula: Formula,
        risk_envelope: RiskEnvelope,
        policy_id: str,
        evidence_ref: str,
    ) -> FormalCodeVerificationResult:
        """Verify a formula without treating backend absence as success."""
        if not evidence_ref.strip():
            raise ValueError("evidence_ref is required")
        sandbox = {
            "execution_allowed": False,
            "timeout_ms": self._timeout_ms,
            "backend": "formal",
        }
        if self._formal_verifier is None:
            return self._result(
                kind="formal",
                evidence_ref=evidence_ref,
                verdict="unsupported",
                decision="warn",
                reason="formal_backend_unsupported",
                explanation="Formal backend is unavailable; human review is required.",
                risk_score=1.0,
                confidence_low=0.0,
                confidence_high=1.0,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
                sandbox=sandbox,
            )
        start = time.monotonic()
        try:
            verdict = self._formal_verifier.verify(
                (ReasoningStep(label="target", formula=formula),)
            )
        except Exception:
            return self._result(
                kind="formal",
                evidence_ref=evidence_ref,
                verdict="verifier_failed",
                decision="warn",
                reason="formal_verifier_failed",
                explanation="Formal verifier failed; result is not proof.",
                risk_score=1.0,
                confidence_low=0.0,
                confidence_high=1.0,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
                sandbox=sandbox,
            )
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if elapsed_ms > self._timeout_ms:
            return self._timeout_result(
                kind="formal",
                evidence_ref=evidence_ref,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
                sandbox=sandbox,
            )
        if verdict.contradictory:
            return self._result(
                kind="formal",
                evidence_ref=evidence_ref,
                verdict="contradictory",
                decision="halt",
                reason="formal_contradiction",
                explanation="Formal verifier found a contradiction.",
                risk_score=1.0,
                confidence_low=1.0,
                confidence_high=1.0,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
                sandbox=sandbox,
            )
        return self._result(
            kind="formal",
            evidence_ref=evidence_ref,
            verdict="consistent",
            decision="allow",
            reason="formal_supported",
            explanation="Formal verifier did not find a contradiction.",
            risk_score=0.0,
            confidence_low=0.0,
            confidence_high=0.0,
            risk_envelope=risk_envelope,
            policy_id=policy_id,
            sandbox=sandbox,
        )

    def verify_code(
        self,
        *,
        code: str,
        risk_envelope: RiskEnvelope,
        policy_id: str,
        evidence_ref: str,
        language: str = "python",
        known_modules: set[str] | None = None,
        api_manifest: dict[str, set[str]] | None = None,
    ) -> FormalCodeVerificationResult:
        """Verify code structurally without executing it."""
        if not evidence_ref.strip():
            raise ValueError("evidence_ref is required")
        sandbox = {
            "execution_allowed": False,
            "timeout_ms": self._timeout_ms,
            "language": language,
        }
        start = time.monotonic()
        try:
            result = self._code_verifier(
                code=code,
                language=language,
                known_modules=known_modules,
                api_manifest=api_manifest,
            )
        except Exception:
            return self._result(
                kind="code",
                evidence_ref=evidence_ref,
                verdict="verifier_failed",
                decision="warn",
                reason="code_verifier_failed",
                explanation="Code verifier failed; result is not proof.",
                risk_score=1.0,
                confidence_low=0.0,
                confidence_high=1.0,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
                sandbox=sandbox,
            )
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if elapsed_ms > self._timeout_ms:
            return self._timeout_result(
                kind="code",
                evidence_ref=evidence_ref,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
                sandbox=sandbox,
            )
        error_count = int(getattr(result, "error_count", 0))
        syntax_valid = bool(getattr(result, "syntax_valid", True))
        if error_count or not syntax_valid:
            return self._result(
                kind="code",
                evidence_ref=evidence_ref,
                verdict="invalid",
                decision="halt",
                reason="code_verifier_rejected",
                explanation="Code verifier found structural or API errors.",
                risk_score=1.0,
                confidence_low=1.0,
                confidence_high=1.0,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
                sandbox=sandbox,
            )
        return self._result(
            kind="code",
            evidence_ref=evidence_ref,
            verdict="valid",
            decision="allow",
            reason="code_verifier_supported",
            explanation="Code verifier found no structural or API errors.",
            risk_score=0.0,
            confidence_low=0.0,
            confidence_high=0.0,
            risk_envelope=risk_envelope,
            policy_id=policy_id,
            sandbox=sandbox,
        )

    def _timeout_result(
        self,
        *,
        kind: str,
        evidence_ref: str,
        risk_envelope: RiskEnvelope,
        policy_id: str,
        sandbox: Mapping[str, Any],
    ) -> FormalCodeVerificationResult:
        return self._result(
            kind=kind,
            evidence_ref=evidence_ref,
            verdict="timeout",
            decision="warn",
            reason=f"{kind}_verifier_timeout",
            explanation="Verifier exceeded the configured time budget.",
            risk_score=1.0,
            confidence_low=0.0,
            confidence_high=1.0,
            risk_envelope=risk_envelope,
            policy_id=policy_id,
            sandbox=sandbox,
        )

    def _result(
        self,
        *,
        kind: str,
        evidence_ref: str,
        verdict: str,
        decision: str,
        reason: str,
        explanation: str,
        risk_score: float,
        confidence_low: float,
        confidence_high: float,
        risk_envelope: RiskEnvelope,
        policy_id: str,
        sandbox: Mapping[str, Any],
    ) -> FormalCodeVerificationResult:
        signal = VerifierSignal(
            verifier=f"{kind}.verifier",
            modality="code",
            score=risk_score,
            verdict=verdict,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            evidence_refs=(evidence_ref,),
        )
        guard_decision = GuardDecision(
            decision=decision,
            risk_score=risk_score,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            policy_id=policy_id,
            reason=reason,
            tenant_safe_explanation=explanation,
            evidence_refs=(evidence_ref,),
            verifier_signals=(signal,),
            risk_envelope=risk_envelope,
            attributes={
                "verifier_kind": kind,
                "execution_allowed": "false",
            },
        )
        return FormalCodeVerificationResult(
            kind=kind,
            evidence_ref=evidence_ref,
            signal=signal,
            guard_decision=guard_decision,
            sandbox=sandbox,
        )
