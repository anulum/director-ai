# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Multimodal Guard Decision Adapter

"""Convert multimodal checks into shared guard decisions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from director_ai.core.guard_control import GuardDecision, RiskEnvelope, VerifierSignal
from director_ai.core.safety_event import SafetyEvent

from .claim import MultimodalClaim
from .guard import MultimodalGuard, TemporalConsistencyGuard

Modality = Literal["image", "audio", "video"]
_MODALITIES = frozenset({"image", "audio", "video"})

__all__ = [
    "MultimodalCheckRequest",
    "MultimodalCheckResult",
    "MultimodalVerifierAdapter",
]


@dataclass(frozen=True)
class MultimodalCheckRequest:
    """Input envelope for opt-in multimodal verification."""

    modality: Modality
    claim_text: str
    media_ref: str
    image_bytes: bytes = b""
    transcript_text: str = ""
    frame_similarities: Sequence[float] = ()

    def __post_init__(self) -> None:
        if self.modality not in _MODALITIES:
            raise ValueError(f"unsupported modality {self.modality!r}")
        if not self.claim_text.strip():
            raise ValueError("claim_text is required")
        if not self.media_ref.strip():
            raise ValueError("media_ref is required")
        object.__setattr__(
            self,
            "frame_similarities",
            tuple(float(value) for value in self.frame_similarities),
        )


@dataclass(frozen=True)
class MultimodalCheckResult:
    """Tenant-safe multimodal verification result."""

    request: MultimodalCheckRequest
    signal: VerifierSignal
    guard_decision: GuardDecision

    def to_dict(self) -> dict[str, Any]:
        """Serialise without raw media, transcript, or claim text."""
        return {
            "modality": self.request.modality,
            "media_ref": self.request.media_ref,
            "signal": self.signal.to_dict(),
            "guard_decision": self.guard_decision.to_dict(),
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
        """Convert the decision into the shared tenant-safe event schema."""
        return self.guard_decision.to_safety_event(
            hook_id=hook_id,
            hook_scope=hook_scope,
            request_id=request_id,
            tenant_id=tenant_id,
            latency_ms=latency_ms,
        )


class MultimodalVerifierAdapter:
    """Opt-in adapter from modality-specific checks to guard decisions."""

    def __init__(
        self,
        *,
        image_guard: MultimodalGuard | Any | None = None,
        audio_score_fn: Callable[[str, str], float] | None = None,
        enabled_modalities: Sequence[Modality] = (),
        benchmarked_modalities: Sequence[Modality] = (),
        temporal_alpha: float = 0.5,
        temporal_floor: float = 0.2,
    ) -> None:
        self._image_guard = image_guard
        self._audio_score = audio_score_fn
        self._enabled = frozenset(enabled_modalities)
        self._benchmarked = frozenset(benchmarked_modalities)
        self._temporal_alpha = temporal_alpha
        self._temporal_floor = temporal_floor
        if not self._enabled:
            raise ValueError("at least one enabled modality is required")
        unsupported = self._enabled - _MODALITIES
        if unsupported:
            raise ValueError(f"unsupported enabled modalities {sorted(unsupported)}")
        extra_benchmarked = self._benchmarked - self._enabled
        if extra_benchmarked:
            raise ValueError(
                "benchmarked modalities must be enabled; got "
                f"{sorted(extra_benchmarked)}"
            )

    def check(
        self,
        request: MultimodalCheckRequest,
        *,
        risk_envelope: RiskEnvelope,
        policy_id: str,
    ) -> MultimodalCheckResult:
        """Run the modality check and return a shared guard decision."""
        if request.modality not in self._enabled:
            raise ValueError(f"modality {request.modality!r} is not enabled")
        score, verdict, evidence_refs = self._score(request)
        if request.modality not in self._benchmarked:
            decision = "warn"
            reason = "multimodal_unbenchmarked"
            explanation = "Modality is enabled but lacks benchmark evidence."
        elif verdict in {"hallucinated", "temporal_inconsistent"}:
            decision = "halt"
            reason = "multimodal_hallucinated"
            explanation = "Multimodal verifier found conflicting evidence."
        elif verdict == "uncertain":
            decision = "warn"
            reason = "multimodal_uncertain"
            explanation = "Multimodal verifier evidence is uncertain."
        else:
            decision = "allow"
            reason = "multimodal_supported"
            explanation = "Multimodal verifier evidence supports the claim."
        risk_score = 1.0 - score
        signal = VerifierSignal(
            verifier=f"multimodal.{request.modality}",
            modality=request.modality,
            score=risk_score,
            verdict=verdict,
            confidence_low=max(0.0, min(score, risk_score)),
            confidence_high=min(1.0, max(score, risk_score)),
            evidence_refs=evidence_refs,
        )
        guard_decision = GuardDecision(
            decision=decision,
            risk_score=risk_score,
            confidence_low=signal.confidence_low,
            confidence_high=signal.confidence_high,
            policy_id=policy_id,
            reason=reason,
            tenant_safe_explanation=explanation,
            evidence_refs=evidence_refs,
            verifier_signals=(signal,),
            risk_envelope=risk_envelope,
            attributes={
                "modality": request.modality,
                "media_ref": request.media_ref,
                "benchmarked": str(request.modality in self._benchmarked).lower(),
            },
        )
        return MultimodalCheckResult(
            request=request,
            signal=signal,
            guard_decision=guard_decision,
        )

    def _score(
        self, request: MultimodalCheckRequest
    ) -> tuple[float, str, tuple[str, ...]]:
        if request.modality == "image":
            if self._image_guard is None:
                raise ValueError("image modality requires image_guard")
            verdict = self._image_guard.check(
                MultimodalClaim(
                    image_bytes=request.image_bytes,
                    text_claim=request.claim_text,
                )
            )
            return (
                _unit(float(verdict.similarity)),
                str(verdict.label),
                (request.media_ref,),
            )
        if request.modality == "audio":
            if self._audio_score is None:
                raise ValueError("audio modality requires audio_score_fn")
            if not request.transcript_text.strip():
                raise ValueError("audio modality requires transcript_text")
            score = _unit(
                float(self._audio_score(request.transcript_text, request.claim_text))
            )
            verdict = (
                "consistent"
                if score >= 0.75
                else "uncertain"
                if score >= 0.4
                else "hallucinated"
            )
            return (score, verdict, (request.media_ref,))
        if request.modality == "video":
            if not request.frame_similarities:
                raise ValueError("video modality requires frame_similarities")
            temporal = TemporalConsistencyGuard(
                alpha=self._temporal_alpha,
                consistency_floor=self._temporal_floor,
            )
            halt_frame = -1
            for index, similarity in enumerate(request.frame_similarities):
                if temporal.update(_unit(similarity)):
                    halt_frame = index
            score = _unit(temporal.ema if temporal.ema is not None else 0.0)
            if halt_frame >= 0:
                return (
                    score,
                    "temporal_inconsistent",
                    (f"{request.media_ref}#frame:{halt_frame}",),
                )
            verdict = "consistent" if score >= 0.75 else "uncertain"
            return (score, verdict, (request.media_ref,))
        raise ValueError(f"unsupported modality {request.modality!r}")


def _unit(value: float) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("score must be finite and in [0, 1]")
    return value
