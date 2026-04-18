# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — containment guard (attestor + detector compose)

"""Single entry-point that composes :class:`ContainmentAttestor`
with :class:`BreakoutDetector`.

The gateway calls :meth:`ContainmentGuard.check` before every
outbound action; the returned :class:`ContainmentVerdict` tells
the caller whether to ``allow`` the action, ``warn`` (log and
continue), or ``block`` (refuse + raise alarm). The guard never
raises on untrusted input — callers always get a structured
verdict so they can log every branch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .anchor import ContainmentAttestor, RealityAnchor
from .detector import BreakoutDetector, BreakoutFinding
from .scope import ContainmentScope, scope_allows_real_effects


@dataclass(frozen=True)
class ContainmentVerdict:
    """Aggregate answer from :meth:`ContainmentGuard.check`.

    ``decision`` is one of ``"allow"``, ``"warn"`` or ``"block"``.
    ``findings`` lists every :class:`BreakoutFinding` the detector
    raised (empty on clean events). ``anchor_reason`` is filled
    only when anchor verification failed, so auditors can
    distinguish cryptographic rejections from behavioural ones.
    """

    decision: str
    findings: tuple[BreakoutFinding, ...]
    anchor_reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


@dataclass
class ContainmentGuard:
    """Orchestrator combining anchor verification and breakout scan.

    The decision matrix is explicit:

    * Anchor fails to verify → ``"block"`` regardless of event
      content. A tampered anchor is the single clearest breakout
      indicator; we never warn on it.
    * Anchor verifies and scope == ``"production"`` → skip the
      production-target check but still run the anti-anchor
      phrase scan (prompt injection is scope-independent).
    * Anchor verifies and scope is one of the rehearsal scopes →
      run every detector, any high-severity finding is
      ``"block"``, a medium-severity one is ``"warn"``.
    """

    attestor: ContainmentAttestor
    detector: BreakoutDetector

    def check(
        self,
        event: Mapping[str, object],
        anchor: RealityAnchor,
        claimed_scope: ContainmentScope | None = None,
    ) -> ContainmentVerdict:
        """Return the verdict for a single proposed action."""
        verification = self.attestor.verify(anchor)
        if not verification.valid:
            return ContainmentVerdict(
                decision="block",
                findings=(),
                anchor_reason=verification.reason,
            )

        findings = tuple(
            self.detector.scan(
                event,
                anchored_scope=anchor.scope,
                claimed_scope=claimed_scope,
            ),
        )

        decision = _decide(anchor.scope, findings)
        return ContainmentVerdict(decision=decision, findings=findings)


def _decide(
    scope: ContainmentScope,
    findings: tuple[BreakoutFinding, ...],
) -> str:
    """Collapse a list of findings into ``allow``/``warn``/``block``.

    Production scope treats production-host findings as expected
    (they are the agent's legitimate target) and downgrades them.
    All other high-severity findings remain blocking.
    """
    if not findings:
        return "allow"

    severities: set[str] = set()
    for finding in findings:
        if scope_allows_real_effects(scope) and finding.category == "production_target":
            # Sanctioned real traffic — don't block on it, but the
            # other detectors (anti-anchor injection, scope
            # mismatch) still apply.
            continue
        severities.add(finding.severity)

    if "high" in severities:
        return "block"
    if "medium" in severities:
        return "warn"
    return "allow"
