# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Execution-ring authorisation gate

"""Gate an agent action on the human factors its risk ring demands.

:class:`ExecutionRingGate` classifies an action into an :class:`ExecutionRing`
and allows it only when every factor that ring requires has been collected. A
read passes unconditionally; a delete waits out a cooling period; an exfiltration
needs two operators and a CISO notification. The decision is side-effect-free and
tenant-safe — it names the ring and the missing factors, never the action payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .authorization import (
    AuthorizationEvidence,
    AuthorizationFactor,
    satisfied_factors,
)
from .rings import RING_REQUIRED_FACTORS, ExecutionRing, classify_operation

__all__ = ["ExecutionRingGate", "RingDecision"]

_DEFAULT_COOLING_SECONDS = 86_400.0  # 24 hours


@dataclass(frozen=True)
class RingDecision:
    """The outcome of gating one action against its ring requirements."""

    ring: ExecutionRing
    allowed: bool
    required: frozenset[AuthorizationFactor]
    satisfied: frozenset[AuthorizationFactor]
    missing: frozenset[AuthorizationFactor] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a tenant-safe JSON dict (factors only, no payload)."""
        return {
            "ring": self.ring.name.lower(),
            "allowed": self.allowed,
            "required": sorted(str(f) for f in self.required),
            "satisfied": sorted(str(f) for f in self.satisfied),
            "missing": sorted(str(f) for f in self.missing),
        }


class ExecutionRingGate:
    """Allow an action only when its ring's authorisation factors are present."""

    def __init__(self, *, cooling_period_seconds: float = _DEFAULT_COOLING_SECONDS):
        if cooling_period_seconds < 0:
            raise ValueError("cooling_period_seconds must be non-negative")
        self._cooling_period_seconds = cooling_period_seconds

    @property
    def cooling_period_seconds(self) -> float:
        """The mandatory delay (seconds) a cooling-period factor must exceed."""
        return self._cooling_period_seconds

    def evaluate(
        self,
        ring: ExecutionRing,
        evidence: AuthorizationEvidence | None = None,
    ) -> RingDecision:
        """Decide whether ``ring`` is authorised by ``evidence``."""
        evidence = evidence or AuthorizationEvidence()
        required = RING_REQUIRED_FACTORS[ring]
        satisfied = satisfied_factors(
            evidence, cooling_period_seconds=self._cooling_period_seconds
        )
        missing = frozenset(required - satisfied)
        return RingDecision(
            ring=ring,
            allowed=not missing,
            required=required,
            satisfied=frozenset(satisfied & required),
            missing=missing,
        )

    def authorize(
        self,
        operation: str,
        evidence: AuthorizationEvidence | None = None,
    ) -> RingDecision:
        """Classify ``operation`` into a ring, then :meth:`evaluate` it."""
        return self.evaluate(classify_operation(operation), evidence)
