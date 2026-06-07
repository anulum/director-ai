# Embodied-AI Robot Command Guard

Verify an LLM-planned robot command sequence **before it executes**. Where the
per-action [cyber-physical grounding hook](cyber-physical.md) checks one action,
this guard checks a whole plan and adds **temporal** safety properties a single
action cannot express — bounded per-step displacement (no teleport jumps) and a
bounded total path length.

It is **warn-only by default**, matching the project's posture that physical hooks
stay advisory until an explicit high-risk flag is set. With
`high_risk_enabled=True`, an unsafe plan is blocked before any action runs, naming
the violated constraint and the offending step.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.cyber_physical import (
    PhysicalAction, Vec3, AABB, WorkspaceConstraint, VelocityConstraint,
)

env = WorkspaceConstraint(
    name="cell", envelope=AABB(min_corner=Vec3(0, 0, 0), max_corner=Vec3(1, 1, 1))
)
speed = VelocityConstraint(name="vmax", max_velocity=1.0)

guard = ProductionGuard(DirectorConfig()).robot_command_guard(
    [env, speed],
    high_risk_enabled=True,        # block unsafe plans (vs warn-only default)
    max_step_displacement=0.5,     # reject teleport-like jumps
    max_path_length=5.0,           # bound the total trajectory
)

plan = [
    PhysicalAction(actuator_id="arm", target_position=Vec3(0.1, 0.1, 0.1)),
    PhysicalAction(actuator_id="arm", target_position=Vec3(5, 5, 5), velocity_magnitude=9.0),
]
verdict = guard.verify_plan(plan)
print(verdict.blocked)            # True — step 1 leaves the cell and is too fast
for v in verdict.violations:
    print(v.step_index, v.constraint, v.reason)
```

`verify_plan()` returns a `PlanVerdict`:

| Field | Meaning |
|-------|---------|
| `blocked` | The plan must not run (only set when `high_risk_enabled`). |
| `warn_only` | Violations exist but the plan is not blocked (advisory mode). |
| `safe` | No violation was raised. |
| `violations` | `StepViolation`s (`step_index`, `constraint`, `reason`). |
| `step_count` | Number of actions in the plan. |

`to_dict()` is tenant-safe — constraint names, step indices, and reasons only.

## What is checked

- **Per-action constraints** — every action is evaluated against the supplied
  `PhysicalConstraint`s (workspace envelope, spatial obstacles, velocity, torque).
- **Temporal properties** — `max_step_displacement` bounds the distance between
  consecutive targets; `max_path_length` bounds the cumulative trajectory
  (reported once when first exceeded).

## Notes

- Composes with the per-action `GroundingHook` (which also enforces tenant
  budgets); this guard adds the plan-level and temporal layer for LLM planners.
- The `model` argument is only required by constraints that use it (e.g. spatial
  collision); workspace/velocity/temporal checks need no model.
- Default warn-only posture keeps it safe to enable in observation mode before a
  real high-risk deployment opts in.
