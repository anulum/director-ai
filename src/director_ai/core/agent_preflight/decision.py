# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Agent preflight decision

"""The decision an agent preflight hook returns.

Every hook resolves to one :class:`PreflightDecision` whose ``decision`` is from
a closed vocabulary and whose ``reason`` is a tenant-safe code, so a host can act
on it (allow / warn / block / escalate to a human) and log it without leaking the
tool arguments or answer text it inspected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DECISIONS", "PreflightDecision"]

DECISIONS = frozenset({"allow", "warn", "block", "escalate"})


@dataclass(frozen=True)
class PreflightDecision:
    """Outcome of one preflight hook.

    Parameters
    ----------
    hook:
        The hook point, e.g. ``"before_tool_call"``.
    decision:
        One of ``allow`` / ``warn`` / ``block`` / ``escalate``.
    reason:
        Tenant-safe code explaining the decision, e.g. ``"unknown_tool"``.
    evidence_refs:
        Identifiers of the evidence the decision relied on, if any.
    metadata:
        Tenant-safe extra fields (e.g. a reversibility score), all stringified.
    """

    hook: str
    decision: str
    reason: str
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalise a tenant-safe preflight decision."""
        if not self.hook.strip():
            raise ValueError("hook is required")
        if self.decision not in DECISIONS:
            raise ValueError(f"unsupported decision {self.decision!r}")
        if not self.reason.strip():
            raise ValueError("reason is required")
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(
            self,
            "metadata",
            {str(k): str(v) for k, v in self.metadata.items()},
        )

    @property
    def allowed(self) -> bool:
        """Whether the action may proceed without intervention."""
        return self.decision == "allow"

    @property
    def blocked(self) -> bool:
        """Whether the action must not proceed (block or escalate)."""
        return self.decision in ("block", "escalate")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe, tenant-safe dict."""
        return {
            "hook": self.hook,
            "decision": self.decision,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def allow(
        cls,
        hook: str,
        *,
        reason: str = "ok",
        evidence_refs: Sequence[str] = (),
        metadata: dict[str, str] | None = None,
    ) -> PreflightDecision:
        """Build an ``allow`` decision."""
        return cls(
            hook=hook,
            decision="allow",
            reason=reason,
            evidence_refs=tuple(evidence_refs),
            metadata=metadata or {},
        )
