# Cyber-Physical Grounding

`director_ai.core.cyber_physical` screens proposed physical actions
(end-effector targets, actuator commands) against geometric, kinematic
and actuator limits before the action executes. Dependency-free by
default; lazy-loaded adapters for ROS 2 / MuJoCo / CARLA when those
stacks are present.

The base install does not install simulator SDKs. Use
`pip install director-ai[physical]` for the pinned MuJoCo Python
wheel. ROS 2 (`rclpy`) and CARLA are installed through their vendor
or distribution channels, then run behind the same isolated physical
runtime boundary.

Dependency boundary:

- Install `[physical]` with `uv sync --locked --extra physical` when working
  from the repository.
- Keep ROS 2 and CARLA in vendor-managed images or venvs; do not add them to
  the base package.
- Pin simulator assets, robot description files, and driver images alongside
  the adapter version.
- Run adapters behind a local action gateway with CPU, memory, and request
  limits.
- Keep `physical_action_mode="warn"` until the deployment has an external stop
  path and calibrated tenant budgets.

## Quick start

```python
from director_ai.core.cyber_physical import (
    AABB, GroundingHook, JointChain, PhysicalAction,
    SimpleKinematicModel, Vec3, WorkspaceConstraint,
)

chain = JointChain(base=Vec3(0, 0, 0), link_lengths=(1.0, 1.0))
model = SimpleKinematicModel(chain=chain)
workspace = AABB(min_corner=Vec3(-5, -5, -5), max_corner=Vec3(5, 5, 5))

hook = GroundingHook(
    model=model,
    constraints=(WorkspaceConstraint(name="cell", envelope=workspace),),
)

action = PhysicalAction(
    actuator_id="arm",
    target_position=Vec3(1.0, 0.0, 0.0),
    velocity_magnitude=0.1,
    torque_magnitude=0.5,
)
verdict = hook.evaluate(action)
if verdict.allowed:
    robot.execute(action)
else:
    for v in verdict.violations:
        log.warning("%s rejected: %s", v.constraint, v.reason)
```

## Primitives

### `Vec3`
Immutable 3-D point / vector; component-wise `+` / `-` / scalar `*`,
`dot`, `norm`, `distance`, `clamp`. Rejects non-finite components on
construction.

### `AABB`
Axis-aligned bounding box with `contains(point)`, `intersects(other)`,
`expand(margin)` and a `centre` property.

### `Sphere`
Solid sphere with `contains(point)`, `intersects(other)`,
`intersects_aabb(box)`. Uses the closest-point-on-box test; a negative
radius is rejected.

## Kinematics

### `JointChain`
Planar revolute chain — base position and a tuple of link lengths.
`reach` returns `sum(link_lengths)`.

### `SimpleKinematicModel`
Dependency-free `KinematicModel` implementation:

- Forward kinematics for arbitrary planar chains via cumulative angle
  walk.
- Analytical two-link IK from Spong's *Robot Modeling and Control*
  (Ch. 5), both `elbow_up` and `elbow_down` branches. Longer chains
  raise `UnsupportedKinematicsError` — the operator plugs in a numerical
  solver or capability-specific `KinematicModel` adapter for those.
- Collision against `AABB` and `Sphere` obstacles, honouring the
  model's `collision_margin` and `end_effector_radius`.

## Actions & constraints

### `PhysicalAction`
Frozen dataclass with `actuator_id`, `target_position`,
`velocity_magnitude`, `torque_magnitude` and optional `joint_angles`.
Construction rejects empty actuator ids and negative velocities /
torques.

### `PhysicalConstraint` Protocol
Read-only `name` property + `evaluate(action, model) -> str | None`
returning either `None` (pass) or a short reason string (fail). The
`model` argument may be `None` for pure action checks; constraints that need
kinematics must return a failure reason instead of raising.

Four concrete implementations ship:

| Class | Checks |
|---|---|
| `SpatialConstraint` | End-effector must stay outside a given set of AABB / sphere obstacles. |
| `WorkspaceConstraint` | End-effector must stay inside an AABB envelope. |
| `VelocityConstraint` | Action velocity ≤ `max_velocity`. |
| `TorqueConstraint` | Action torque ≤ `max_torque`. |

`SpatialConstraint` needs a kinematic model for collision checks. If a caller
omits the model, it reports a violation named after the spatial constraint; in a
high-risk `RobotCommandGuard` deployment that blocks the plan before any action
runs.

## `GroundingHook`

Orchestrator. `evaluate(action)` returns a `GroundingVerdict` listing
*every* failing constraint (no short-circuit) plus optional
reachability rejection if the kinematic model's `inverse()` returns
`None` for the target. Actions carrying an explicit `joint_angles`
vector skip the reachability check, since the joint vector already
fixes the posture.

Each verdict carries `safety_event`, a tenant-safe `SafetyEvent` with
the `cyber_physical.grounding` hook id, policy decision, constraint
references, and a bounded operator explanation.

## Heavy-dependency adapters

All three are constructed via a `from_*` classmethod that imports the
heavy dependency lazily, so the subpackage imports cleanly on
deployments that do not install them.

- `Ros2Adapter.from_ros2(node)` — rclpy node-driven, collision from
  caller-supplied obstacles. ROS 2 stays outside PyPI and must be
  installed through the target distribution in the isolated physical
  runtime. Forward / inverse kinematics are deployment-specific and
  raise `UnsupportedKinematicsError` until configured with FK / IK callables
  backed by robot-state services such as MoveIt FK / IK.
- `MuJoCoAdapter.from_mjcf(path)` — mjcf model loading,
  `mj_forward` + `site_xpos` for forward kinematics, live `data.ncon`
  contact count for collisions. Inverse kinematics raises
  `UnsupportedKinematicsError` unless an operator-supplied numerical solver
  is configured. Install via `director-ai[physical]`.
- `CarlaAdapter.from_carla(port=2000)` — vehicle-class scenarios with
  spawn-filtered collision. Install CARLA from the simulator vendor
  into the isolated physical runtime. Joint-space FK / IK are intentionally
  unsupported; use CARLA vehicle controls or route planning for motion.

## CoherenceAgent wiring

Pass a configured hook to the agent:

```python
from director_ai.core.agent import CoherenceAgent

agent = CoherenceAgent(grounding_hook=hook)
verdict = agent.verify_physical_action(action)
```

Calling `verify_physical_action` with no hook configured raises
`RuntimeError` so callers cannot silently skip the check.
`CoherenceAgent` treats physical failures as warnings by default.
Blocking requires both `physical_action_mode="block"` and
`allow_physical_action_blocking=True`.

## Tenant budgets

Use `TenantPhysicalBudget` on `GroundingHook` to cap expensive physical
checks per tenant:

```python
from director_ai.core.cyber_physical import (
    PhysicalBudgetLimits,
    TenantPhysicalBudget,
)

budget = TenantPhysicalBudget(
    PhysicalBudgetLimits(
        window_seconds=60.0,
        max_action_validations=120,
        max_inverse_kinematics=30,
        max_simulation_checks=60,
        max_sensor_fusion=60,
    )
)
hook = GroundingHook(model=model, constraints=constraints, budget=budget)
```

The hook consumes one action-validation unit for every proposed action,
one inverse-kinematics unit before calling `model.inverse`, and one
simulation-check unit before spatial constraints call `model.collides_with`.
`PhysicalGroundingEvaluator` additionally consumes one sensor-fusion unit
before comparing perception and simulator snapshots.
Budget exhaustion blocks before the expensive operation is called. This
budget block is not converted into warn-only mode, so it still prevents
tenant payloads from exhausting physical runtimes when
`CoherenceAgent` uses the default advisory physical policy.
All four per-counter limits and `window_seconds` must be positive; a zero
limit is rejected at configuration time instead of creating a deny-all
budget that only fails on the first runtime consume.

## Closed-loop evaluator

`PhysicalGroundingEvaluator` compares pre-action perception state,
pre-action simulator state, the planned `PhysicalAction`, and optional
post-action perception/simulator snapshots. It is a policy-control layer
over `GroundingHook`, not a replacement for kinematic checks.

```python
from director_ai.core.cyber_physical import (
    PhysicalGroundingEvaluator,
    SensorStateSnapshot,
)
from director_ai.core.guard_control import RiskEnvelope

evaluator = PhysicalGroundingEvaluator(
    grounding_hook=hook,
    high_risk_physical_deployment=False,
    state_tolerance_m=0.05,
    budget=budget,
)

pre_perception = SensorStateSnapshot(
    snapshot_ref="sensor://pre-1",
    sensor_id="camera-1",
    adapter_id="vision.cell",
    timestamp=1.0,
    end_effector_position=Vec3(0.0, 0.0, 0.0),
)
pre_simulation = SensorStateSnapshot(
    snapshot_ref="sim://pre-1",
    sensor_id="simulator-1",
    adapter_id="sim.cell",
    timestamp=1.0,
    end_effector_position=Vec3(0.01, 0.0, 0.0),
)

result = evaluator.evaluate(
    action=action,
    risk_envelope=RiskEnvelope(
        action_category="physical",
        reversibility="reversible",
        domain="physical",
        calibrated_threshold=0.6,
        no_go_threshold=0.8,
    ),
    pre_perception=pre_perception,
    pre_simulation=pre_simulation,
    tenant_id="tenant-a",
)
```

Production semantics:

- unsupported sensors/adapters produce explicit `unsupported` violations
  rather than silent passes
- perception/simulator mismatch produces `warn` by default
- mismatch becomes `block` only when `high_risk_physical_deployment=True`
- irreversible actions go through no-go policy and require human review
- post-action checks compare both perception and simulation with the planned
  target and with each other
- `SensorStateSnapshot` serialises references and coordinates only; it does
  not serialise raw frames, point clouds, prompts, credentials, or driver
  payloads

## Live physical grounding loop

`PhysicalGroundingLoop` wraps the evaluator for live deployments. It first runs
a pre-action sensor-fusion check, executes the supplied action callback only
when the pre-check allows, then re-runs the evaluator with post-action
perception and simulation snapshots.

High-risk physical deployments require both post-action suppliers. Missing
post-action sensor fusion returns a blocking evaluation with reason
`post_action_sensor_fusion_required`; the action callback is not invoked.

```python
from director_ai.core.cyber_physical import PhysicalGroundingLoop

loop = PhysicalGroundingLoop(
    evaluator=evaluator,
    execute_action=lambda action: robot_driver.move_to(action.target_position),
)

result = loop.run(
    action=action,
    risk_envelope=risk_envelope,
    pre_perception=pre_perception,
    pre_simulation=pre_simulation,
    post_perception=lambda: camera_adapter.snapshot(),
    post_simulation=lambda: simulator_adapter.snapshot(),
    tenant_id="tenant-a",
)

if result.final_evaluation.decision.decision == "block":
    raise RuntimeError(result.final_evaluation.reason)
```

The loop deliberately stores only `SensorStateSnapshot` references and numeric
coordinates. Raw camera frames, simulator dumps, driver payloads, credentials,
and private sensor packets remain outside Director telemetry.

::: director_ai.core.cyber_physical.closed_loop.SensorStateSnapshot

::: director_ai.core.cyber_physical.closed_loop.PhysicalGroundingEvaluator

::: director_ai.core.cyber_physical.closed_loop.PhysicalGroundingLoop

::: director_ai.core.cyber_physical.closed_loop.PhysicalGroundingLoopResult

::: director_ai.core.cyber_physical.closed_loop.PhysicalGroundingEvaluation

::: director_ai.core.cyber_physical.closed_loop.PhysicalGroundingViolation

## Rust acceleration

When `backfire_kernel` is installed, `AABB.contains`, `Sphere.contains`
/ `intersects` / `intersects_aabb` and the two-link IK dispatch to
the `rust_*` FFI functions. Bit-exact on geometry, < 1e-12 ULP on
the trigonometric IK path. No change on the caller side.
