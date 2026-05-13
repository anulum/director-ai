# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - inference-server hook adapters

"""Pre-sampling hook boundary for inference server integrations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from director_ai.core.safety_event import SafetyEvent
from director_ai.core.trajectory import PreHaltSteeringDecision

InferenceServerName = Literal["vllm", "tgi", "llama_cpp"]
ScoreFn = Callable[[str], float]

_SERVERS: frozenset[str] = frozenset({"vllm", "tgi", "llama_cpp"})

__all__ = [
    "InferenceHookDecision",
    "InferenceHookRequest",
    "InferenceServerHook",
    "InferenceServerHookPolicy",
    "InferenceServerName",
    "build_inference_server_hook",
]


@dataclass(frozen=True)
class InferenceHookRequest:
    """Candidate token and stream context supplied by an inference server."""

    server: InferenceServerName
    accumulated_text: str
    candidate_token: str
    token_id: int | None = None
    request_id: str = ""
    tenant_id: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_server(self.server)
        if self.token_id is not None and self.token_id < 0:
            raise ValueError("token_id must be non-negative")
        object.__setattr__(
            self,
            "metadata",
            {str(key): str(value) for key, value in self.metadata.items()},
        )

    @property
    def candidate_text(self) -> str:
        """Return the text that would exist if the token were accepted."""
        return f"{self.accumulated_text}{self.candidate_token}"


@dataclass(frozen=True)
class InferenceServerHookPolicy:
    """Threshold and masking policy for pre-sampling decisions."""

    hard_limit: float = 0.4
    block_token_id: int | None = None
    block_logit: float = -1.0e9
    steering_bias_logit: float = -5.0
    halt_reason: str = "coherence_below_threshold"
    tenant_safe_explanation: str = "Candidate token rejected before sampling."

    def __post_init__(self) -> None:
        if not math.isfinite(self.hard_limit):
            raise ValueError("hard_limit must be finite")
        if self.hard_limit < 0.0 or self.hard_limit > 1.0:
            raise ValueError("hard_limit must be in [0, 1]")
        if self.block_token_id is not None and self.block_token_id < 0:
            raise ValueError("block_token_id must be non-negative")
        if not math.isfinite(self.block_logit):
            raise ValueError("block_logit must be finite")
        if not math.isfinite(self.steering_bias_logit):
            raise ValueError("steering_bias_logit must be finite")
        if self.steering_bias_logit >= 0.0:
            raise ValueError("steering_bias_logit must be negative")
        if not self.halt_reason.strip():
            raise ValueError("halt_reason is required")
        if not self.tenant_safe_explanation.strip():
            raise ValueError("tenant_safe_explanation is required")


@dataclass(frozen=True)
class InferenceHookDecision:
    """Decision returned to the inference-server adapter layer."""

    allow: bool
    score: float
    reason: str
    adjusted_logits: tuple[float, ...] | None = None
    blocked_token_ids: tuple[int, ...] = ()
    safety_event: SafetyEvent | None = None
    server_payload: dict[str, object] = field(default_factory=dict)


class InferenceServerHook:
    """Common pre-sampling hook used by vLLM, TGI, and llama.cpp adapters."""

    def __init__(
        self,
        server: InferenceServerName,
        score_fn: ScoreFn,
        policy: InferenceServerHookPolicy | None = None,
    ) -> None:
        _validate_server(server)
        self.server = server
        self.score_fn = score_fn
        self.policy = policy or InferenceServerHookPolicy()

    def check(
        self,
        request: InferenceHookRequest,
        logits: Sequence[float] | None = None,
    ) -> InferenceHookDecision:
        """Score a candidate token and return a server-neutral action."""
        if request.server != self.server:
            raise ValueError(
                f"request server {request.server!r} does not match hook {self.server!r}"
            )

        score = _clamp_unit(float(self.score_fn(request.candidate_text)))
        if score >= self.policy.hard_limit:
            return InferenceHookDecision(
                allow=True,
                score=score,
                reason="allow",
                adjusted_logits=tuple(logits) if logits is not None else None,
                server_payload=_allow_payload(request.server, score),
            )

        token_id = self.policy.block_token_id
        if token_id is None:
            token_id = request.token_id
        blocked = (token_id,) if token_id is not None else ()
        adjusted = _mask_logits(logits, token_id, self.policy.block_logit)
        event = SafetyEvent.from_policy_decision(
            hook_id=f"inference_server.{request.server}",
            hook_scope="inference_server",
            policy_decision="halt",
            halt_reason=self.policy.halt_reason,
            tenant_safe_explanation=self.policy.tenant_safe_explanation,
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            threshold=self.policy.hard_limit,
            observed_score=score,
            attributes=_event_attributes(request, token_id),
        )
        return InferenceHookDecision(
            allow=False,
            score=score,
            reason=self.policy.halt_reason,
            adjusted_logits=adjusted,
            blocked_token_ids=blocked,
            safety_event=event,
            server_payload=_halt_payload(
                request.server,
                score,
                token_id,
                self.policy.block_logit,
            ),
        )

    def steer(
        self,
        request: InferenceHookRequest,
        steering_decision: PreHaltSteeringDecision,
        logits: Sequence[float] | None = None,
    ) -> InferenceHookDecision:
        """Apply predictive pre-halt steering at the pre-sampling boundary."""
        if request.server != self.server:
            raise ValueError(
                f"request server {request.server!r} does not match hook {self.server!r}"
            )

        score = _clamp_unit(float(steering_decision.halt_probability))
        if steering_decision.action == "proceed":
            return InferenceHookDecision(
                allow=True,
                score=score,
                reason=steering_decision.reason,
                adjusted_logits=tuple(logits) if logits is not None else None,
                server_payload=_allow_payload(request.server, score),
            )

        token_id = self.policy.block_token_id
        if token_id is None:
            token_id = request.token_id

        if steering_decision.action == "escalate":
            adjusted = _mask_logits(logits, token_id, self.policy.steering_bias_logit)
            event = _steering_event(request, steering_decision, token_id)
            return InferenceHookDecision(
                allow=True,
                score=score,
                reason=steering_decision.reason,
                adjusted_logits=adjusted,
                safety_event=event,
                server_payload=_bias_payload(
                    request.server,
                    score,
                    token_id,
                    self.policy.steering_bias_logit,
                ),
            )

        blocked = (token_id,) if token_id is not None else ()
        adjusted = _mask_logits(logits, token_id, self.policy.block_logit)
        event = _steering_event(request, steering_decision, token_id)
        return InferenceHookDecision(
            allow=False,
            score=score,
            reason=steering_decision.reason,
            adjusted_logits=adjusted,
            blocked_token_ids=blocked,
            safety_event=event,
            server_payload=_halt_payload(
                request.server,
                score,
                token_id,
                self.policy.block_logit,
            ),
        )


def build_inference_server_hook(
    server: InferenceServerName,
    score_fn: ScoreFn,
    *,
    hard_limit: float = 0.4,
    block_token_id: int | None = None,
    block_logit: float = -1.0e9,
) -> InferenceServerHook:
    """Build a pre-sampling hook with the default policy fields."""
    return InferenceServerHook(
        server=server,
        score_fn=score_fn,
        policy=InferenceServerHookPolicy(
            hard_limit=hard_limit,
            block_token_id=block_token_id,
            block_logit=block_logit,
        ),
    )


def _validate_server(server: str) -> None:
    if server not in _SERVERS:
        names = ", ".join(sorted(_SERVERS))
        raise ValueError(f"unsupported inference server {server!r}; expected {names}")


def _clamp_unit(value: float) -> float:
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return 1.0 if value > 0.0 else 0.0
    return max(0.0, min(1.0, value))


def _mask_logits(
    logits: Sequence[float] | None,
    token_id: int | None,
    block_logit: float,
) -> tuple[float, ...] | None:
    if logits is None:
        return None
    adjusted = [float(value) for value in logits]
    if token_id is not None and token_id < len(adjusted):
        adjusted[token_id] = block_logit
    return tuple(adjusted)


def _event_attributes(
    request: InferenceHookRequest,
    token_id: int | None,
) -> dict[str, str]:
    attributes: dict[str, str] = {"server": str(request.server)}
    if token_id is not None:
        attributes["token_id"] = str(token_id)
    attributes.update(request.metadata)
    return attributes


def _steering_event(
    request: InferenceHookRequest,
    steering_decision: PreHaltSteeringDecision,
    token_id: int | None,
) -> SafetyEvent:
    guard = steering_decision.guard_decision
    attributes = {
        "policy_id": guard.policy_id,
        "risk_domain": guard.risk_envelope.domain,
        "action_category": guard.risk_envelope.action_category,
        "reversibility": guard.risk_envelope.reversibility,
        **dict(guard.attributes),
        **_event_attributes(request, token_id),
    }
    return SafetyEvent.from_policy_decision(
        hook_id=f"inference_server.{request.server}.prehalt",
        hook_scope="inference_server",
        policy_decision=guard.decision,
        halt_reason=steering_decision.reason,
        tenant_safe_explanation=guard.tenant_safe_explanation,
        request_id=request.request_id,
        tenant_id=request.tenant_id,
        threshold=guard.risk_envelope.calibrated_threshold,
        observed_score=guard.risk_score,
        evidence_refs=steering_decision.evidence_refs,
        attributes=attributes,
    )


def _allow_payload(server: str, score: float) -> dict[str, object]:
    return {
        "server": server,
        "action": "allow",
        "allow": True,
        "score": score,
    }


def _halt_payload(
    server: str,
    score: float,
    token_id: int | None,
    block_logit: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "server": server,
        "allow": False,
        "score": score,
    }
    if server == "vllm":
        payload["action"] = "mask_token"
        payload["token_ids"] = [] if token_id is None else [token_id]
    elif server == "tgi":
        payload["action"] = "filter_next_token"
        payload["token_ids"] = [] if token_id is None else [token_id]
    else:
        payload["action"] = "logit_bias"
        if token_id is not None:
            payload["token_id"] = token_id
            payload["bias"] = block_logit
    return payload


def _bias_payload(
    server: str,
    score: float,
    token_id: int | None,
    bias_logit: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "server": server,
        "allow": True,
        "score": score,
    }
    token_biases = {} if token_id is None else {token_id: bias_logit}
    if server == "vllm":
        payload["action"] = "bias_token"
        payload["token_biases"] = token_biases
    elif server == "tgi":
        payload["action"] = "bias_next_token"
        payload["token_biases"] = token_biases
    else:
        payload["action"] = "logit_bias"
        if token_id is not None:
            payload["token_id"] = token_id
            payload["bias"] = bias_logit
    return payload
