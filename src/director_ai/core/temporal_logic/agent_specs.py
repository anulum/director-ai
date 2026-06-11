# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Agent Trajectory Safety Specifications
"""Built-in LTL safety specifications for LLM agent trajectories.

This is the agent-domain application of the LTL monitor: it names the atomic
propositions an agent step can raise, expresses the EU AI Act Article 15
"continuous monitoring" obligations as LTL formulas, and runs them together over
a trajectory with :class:`TrajectorySafetyMonitor`.

The built-in specifications are:

* ``tool_calls_are_verified`` — ``G(tool_call → F verification_passed)``
* ``handoff_is_coherence_checked`` — ``G(handoff → X coherence_check)``
* ``no_output_after_injection`` — ``G(¬(injection_detected ∧ output_emitted))``
* ``fact_claims_are_grounded`` — ``G(fact_claim → F evidence_retrieved)``
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .formula import F, G, X, and_, atom, implies, not_
from .monitor import LTLMonitor, Verdict

# ── Atomic propositions an agent step can raise ──────────────────────────────

TOOL_CALL = "tool_call"
VERIFICATION_PASSED = "verification_passed"
HANDOFF = "handoff"
COHERENCE_CHECK = "coherence_check"
INJECTION_DETECTED = "injection_detected"
OUTPUT_EMITTED = "output_emitted"
FACT_CLAIM = "fact_claim"
EVIDENCE_RETRIEVED = "evidence_retrieved"


def default_agent_safety_specs() -> dict[str, object]:
    """Return the built-in named LTL agent-safety specifications.

    A fresh mapping is returned on each call so callers can extend it without
    mutating shared state.
    """
    return {
        "tool_calls_are_verified": G(
            implies(atom(TOOL_CALL), F(atom(VERIFICATION_PASSED))),
        ),
        "handoff_is_coherence_checked": G(
            implies(atom(HANDOFF), X(atom(COHERENCE_CHECK))),
        ),
        "no_output_after_injection": G(
            not_(and_(atom(INJECTION_DETECTED), atom(OUTPUT_EMITTED))),
        ),
        "fact_claims_are_grounded": G(
            implies(atom(FACT_CLAIM), F(atom(EVIDENCE_RETRIEVED))),
        ),
    }


@dataclass(frozen=True)
class StepObservation:
    """The propositions raised by a single agent trajectory step.

    Convert domain events (a tool invocation, a verifier verdict, a handoff, an
    injection-detector hit, an emitted token, a fact claim, a retrieval) into the
    atomic propositions the LTL specifications reason over.
    """

    tool_call: bool = False
    verification_passed: bool = False
    handoff: bool = False
    coherence_check: bool = False
    injection_detected: bool = False
    output_emitted: bool = False
    fact_claim: bool = False
    evidence_retrieved: bool = False

    def propositions(self) -> frozenset[str]:
        """Return the set of true atomic-proposition names for this step."""
        mapping = {
            TOOL_CALL: self.tool_call,
            VERIFICATION_PASSED: self.verification_passed,
            HANDOFF: self.handoff,
            COHERENCE_CHECK: self.coherence_check,
            INJECTION_DETECTED: self.injection_detected,
            OUTPUT_EMITTED: self.output_emitted,
            FACT_CLAIM: self.fact_claim,
            EVIDENCE_RETRIEVED: self.evidence_retrieved,
        }
        return frozenset(name for name, present in mapping.items() if present)


@dataclass(frozen=True)
class SpecStatus:
    """The runtime status of one named specification."""

    name: str
    verdict: Verdict
    formula: str
    violated_at_step: int | None

    def to_dict(self) -> dict[str, object]:
        """Tenant-safe serialisable view (no trajectory payload, formula only)."""
        return {
            "name": self.name,
            "verdict": self.verdict.value,
            "formula": self.formula,
            "violated_at_step": self.violated_at_step,
        }


class TrajectorySafetyMonitor:
    """Run a set of named LTL safety specs over one agent trajectory.

    Feed one :class:`StepObservation` (or an explicit set of proposition names)
    per trajectory step; query :meth:`report` at any time or :meth:`finalize`
    when the trajectory ends to resolve pending eventualities.
    """

    def __init__(self, specs: dict[str, object] | None = None) -> None:
        source = default_agent_safety_specs() if specs is None else specs
        if not source:
            raise ValueError("at least one specification is required")
        self._monitors: dict[str, LTLMonitor] = {
            name: LTLMonitor(formula, name=name)  # type: ignore[arg-type]
            for name, formula in source.items()
        }
        self._violated_at: dict[str, int | None] = dict.fromkeys(source, None)
        self._steps = 0

    @property
    def steps(self) -> int:
        """Number of trajectory steps observed."""
        return self._steps

    def observe(self, step: StepObservation | Iterable[str]) -> None:
        """Advance every monitor by one trajectory step."""
        state = (
            step.propositions()
            if isinstance(step, StepObservation)
            else frozenset(step)
        )
        self._steps += 1
        for name, monitor in self._monitors.items():
            verdict = monitor.push(state)
            if verdict is Verdict.VIOLATED and self._violated_at[name] is None:
                self._violated_at[name] = self._steps

    def finalize(self) -> dict[str, object]:
        """Resolve pending obligations and return the final report."""
        for name, monitor in self._monitors.items():
            verdict = monitor.finalize()
            if verdict is Verdict.VIOLATED and self._violated_at[name] is None:
                self._violated_at[name] = self._steps
        return self.report()

    def report(self) -> dict[str, object]:
        """Return a tenant-safe report of every spec's current verdict.

        ``verdict`` is the worst spec verdict (violated > inconclusive >
        satisfied). The report carries only specification formulas and step
        indices — never trajectory content.
        """
        statuses = [
            SpecStatus(
                name=name,
                verdict=monitor.verdict,
                formula=str(monitor.residual)
                if monitor.verdict is Verdict.INCONCLUSIVE
                else str(monitor.initial),
                violated_at_step=self._violated_at[name],
            )
            for name, monitor in self._monitors.items()
        ]
        verdicts = {status.verdict for status in statuses}
        if Verdict.VIOLATED in verdicts:
            overall = Verdict.VIOLATED
        elif Verdict.INCONCLUSIVE in verdicts:
            overall = Verdict.INCONCLUSIVE
        else:
            overall = Verdict.SATISFIED
        return {
            "verdict": overall.value,
            "steps": self._steps,
            "specs": [status.to_dict() for status in statuses],
            "violations": [
                status.name for status in statuses if status.verdict is Verdict.VIOLATED
            ],
        }
