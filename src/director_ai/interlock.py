# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — standalone interlock kernel

"""Standalone halt/interlock kernel for caller-owned scorers."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from director_ai.core.safety_event import SafetyEvent

__all__ = [
    "InterlockDecision",
    "InterlockKernel",
    "InterlockPolicy",
]


@dataclass(frozen=True)
class InterlockPolicy:
    """Policy thresholds for the standalone interlock kernel."""

    hard_limit: float = 0.5
    window_size: int = 4
    window_threshold: float = 0.5
    trend_window: int = 0
    trend_threshold: float = 0.2
    warn_only: bool = False
    hook_id: str = "interlock.kernel"
    hook_scope: str = "streaming"
    policy_id: str = "policy.interlock.default"
    tenant_safe_explanation: str = "Interlock policy stopped or flagged the stream."

    def __post_init__(self) -> None:
        """Validate that the interlock policy thresholds are unit-interval values."""
        _unit("hard_limit", self.hard_limit)
        _unit("window_threshold", self.window_threshold)
        _unit("trend_threshold", self.trend_threshold)
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.trend_window < 0:
            raise ValueError("trend_window must be >= 0")
        if not self.hook_id.strip():
            raise ValueError("hook_id is required")
        if not self.policy_id.strip():
            raise ValueError("policy_id is required")
        if not self.tenant_safe_explanation.strip():
            raise ValueError("tenant_safe_explanation is required")


@dataclass(frozen=True)
class InterlockDecision:
    """Result of applying the interlock to a token sequence."""

    decision: str
    output: str
    scores: tuple[float, ...]
    halt_index: int = -1
    halt_reason: str = ""
    halt_event: SafetyEvent | None = None
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the decision label and freeze the score tuple."""
        if self.decision not in {"allow", "warn", "halt"}:
            raise ValueError(f"unsupported decision {self.decision!r}")
        object.__setattr__(self, "scores", tuple(float(score) for score in self.scores))
        for score in self.scores:
            _unit("score", score)
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result without rejected token text."""
        return {
            "decision": self.decision,
            "output": self.output,
            "scores": list(self.scores),
            "halt_index": self.halt_index,
            "halt_reason": self.halt_reason,
            "evidence_refs": list(self.evidence_refs),
            "halt_event": (
                self.halt_event.to_dict() if self.halt_event is not None else None
            ),
        }


class InterlockKernel:
    """Dependency-light halt gate for streams scored by caller code."""

    def __init__(self, policy: InterlockPolicy | None = None) -> None:
        self.policy = policy or InterlockPolicy()

    def run(
        self,
        tokens: Iterable[str],
        *,
        scorer: Callable[[str], float | Any],
        request_id: str = "",
        tenant_id: str = "",
    ) -> InterlockDecision:
        """Score candidate output and stop before admitting unsafe tokens."""
        accepted: list[str] = []
        scores: list[float] = []
        warning: tuple[int, str, float] | None = None

        for index, token in enumerate(tokens):
            token_text = str(token)
            candidate = "".join(accepted) + token_text
            score = _coerce_score(scorer(candidate))
            scores.append(score)
            reason = self._halt_reason(scores)
            if reason:
                evidence_refs = (f"interlock://token/{index}",)
                if self.policy.warn_only:
                    warning = (index, reason, score)
                    accepted.append(token_text)
                    continue
                return self._decision(
                    decision="halt",
                    output="".join(accepted),
                    scores=scores,
                    halt_index=index,
                    halt_reason=reason,
                    observed_score=score,
                    evidence_refs=evidence_refs,
                    request_id=request_id,
                    tenant_id=tenant_id,
                )
            accepted.append(token_text)

        if warning is not None:
            index, reason, score = warning
            evidence_refs = (f"interlock://token/{index}",)
            return self._decision(
                decision="warn",
                output="".join(accepted),
                scores=scores,
                halt_index=index,
                halt_reason=reason,
                observed_score=score,
                evidence_refs=evidence_refs,
                request_id=request_id,
                tenant_id=tenant_id,
            )
        return InterlockDecision(
            decision="allow",
            output="".join(accepted),
            scores=tuple(scores),
        )

    def _halt_reason(self, scores: Sequence[float]) -> str:
        current = scores[-1]
        if current < self.policy.hard_limit:
            return "hard_limit"
        if len(scores) >= self.policy.window_size:
            window = scores[-self.policy.window_size :]
            if sum(window) / len(window) < self.policy.window_threshold:
                return "window_average"
        if self.policy.trend_window > 0 and len(scores) > self.policy.trend_window:
            previous = scores[-self.policy.trend_window - 1]
            if previous - current > self.policy.trend_threshold:
                return "downward_trend"
        return ""

    def _decision(
        self,
        *,
        decision: str,
        output: str,
        scores: list[float],
        halt_index: int,
        halt_reason: str,
        observed_score: float,
        evidence_refs: tuple[str, ...],
        request_id: str,
        tenant_id: str,
    ) -> InterlockDecision:
        event = SafetyEvent.from_policy_decision(
            hook_id=self.policy.hook_id,
            hook_scope=self.policy.hook_scope,
            policy_decision=decision,
            halt_reason=halt_reason,
            tenant_safe_explanation=self.policy.tenant_safe_explanation,
            request_id=request_id,
            tenant_id=tenant_id,
            threshold=self.policy.hard_limit,
            observed_score=observed_score,
            evidence_refs=evidence_refs,
            attributes={
                "policy_id": self.policy.policy_id,
                "window_size": str(self.policy.window_size),
                "warn_only": str(self.policy.warn_only).lower(),
            },
        )
        return InterlockDecision(
            decision=decision,
            output=output,
            scores=tuple(scores),
            halt_index=halt_index,
            halt_reason=halt_reason,
            halt_event=event,
            evidence_refs=evidence_refs,
        )


def _coerce_score(result: float | Any) -> float:
    if hasattr(result, "score"):
        result = result.score
    score = float(result)
    return _unit("score", score)


def _unit(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value
