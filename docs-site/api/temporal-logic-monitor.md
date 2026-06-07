# Temporal-Logic Trajectory Monitor

Runtime Linear Temporal Logic (LTL) monitoring of agent trajectories: check
formal safety specifications over the whole sequence of steps, not just one step
in isolation. This is the formal reading of the EU AI Act Article 15 requirement
for "continuous monitoring" of high-risk systems.

## Why temporal logic?

A loop detector tells you a single step is circular. It cannot express
properties *over time* such as "every tool call is eventually verified" or "no
output is emitted after an injection is detected". Those are temporal properties,
and LTL is the standard formalism for them. The monitor evaluates each property
incrementally as the trajectory unfolds and returns a three-valued verdict, so a
violation is reported the moment it becomes unavoidable.

## Built-in agent-safety specifications

| Name | LTL | Meaning |
|------|-----|---------|
| `tool_calls_are_verified` | `G(tool_call → F verification_passed)` | Every tool call is eventually verified |
| `handoff_is_coherence_checked` | `G(handoff → X coherence_check)` | A handoff is coherence-checked on the next step |
| `no_output_after_injection` | `G(¬(injection_detected ∧ output_emitted))` | Never emit output in a step where injection was detected |
| `fact_claims_are_grounded` | `G(fact_claim → F evidence_retrieved)` | Every fact claim is eventually grounded in evidence |

`G` = always, `F` = eventually, `X` = next, `→` = implies.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.temporal_logic import StepObservation

guard = ProductionGuard(DirectorConfig())
monitor = guard.trajectory_monitor()

# Feed one observation per agent step. Map your agent's events onto the
# StepObservation flags (a tool invocation, a verifier verdict, a handoff, an
# injection-detector hit, an emitted token, a fact claim, a retrieval).
monitor.observe(StepObservation(tool_call=True))
monitor.observe(StepObservation(verification_passed=True))
monitor.observe(StepObservation(fact_claim=True))
monitor.observe(StepObservation(evidence_retrieved=True))

# When the trajectory ends, resolve any pending obligations.
report = monitor.finalize()
print(report["verdict"])      # "satisfied" | "violated" | "inconclusive"
print(report["violations"])   # e.g. ["tool_calls_are_verified"]
```

## Verdicts

The verdict is three-valued and definitive verdicts latch (a later step cannot
flip them):

- **`violated`** — a safety property was broken; reported immediately.
- **`satisfied`** — the property can no longer fail.
- **`inconclusive`** — the property still depends on future steps.

`report()` returns the worst verdict across all specifications. The overall
report is tenant-safe: it carries only specification formulas and step indices,
never trajectory content.

```json
{
  "verdict": "violated",
  "steps": 3,
  "specs": [
    {"name": "no_output_after_injection",
     "verdict": "violated",
     "formula": "G ¬(injection_detected ∧ output_emitted)",
     "violated_at_step": 2}
  ],
  "violations": ["no_output_after_injection"]
}
```

## Finite-trace end semantics

`finalize()` resolves obligations that are still pending when the trajectory
ends, using strong-eventuality semantics:

- a never-violated `Always` (`G`) property is **satisfied**;
- an undischarged `Eventually` (`F`) or `Until` is a **violation** — e.g. a tool
  call that was never verified by the end of the session.

Call `report()` mid-trajectory for the live verdict; call `finalize()` once at
the end to decide pending eventualities.

## Custom specifications

Build your own specifications from the formula algebra and pass them in. Each
spec is a named LTL formula; `StepObservation` (or a raw set of proposition
names) drives them all in lock-step.

```python
from director_ai.core.temporal_logic import (
    TrajectorySafetyMonitor,
    G,
    atom,
    not_,
)

specs = {"never_calls_external_api": G(not_(atom("external_api_call")))}
monitor = TrajectorySafetyMonitor(specs)
monitor.observe({"external_api_call"})
assert monitor.report()["violations"] == ["never_calls_external_api"]
```

The formula constructors — `atom`, `not_`, `and_`, `or_`, `implies`, `G`, `F`,
`X`, `U` — are canonical and self-simplifying, which keeps the progressed formula
bounded no matter how long the trajectory runs.

## Notes

- Monitoring is pure control logic (per-step Boolean formula progression), not a
  numeric hot path, so there is no separate accelerated backend; the residual
  formula is kept bounded by construction.
- Each `guard.trajectory_monitor()` call returns an independent monitor, so
  concurrent trajectories never share state.
