# Cyber-Physical Grounding

`director_ai.core.cyber_physical` screens proposed physical actions
(end-effector targets, actuator commands) against geometric, kinematic
and actuator limits before the action executes. Dependency-free by
default; lazy-loaded adapters for ROS 2 / MuJoCo / CARLA when those
stacks are present.

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
  raise `NotImplementedError` — the operator plugs in a numerical
  solver (e.g. a MuJoCo-backed `KinematicModel` subclass) for those.
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
returning either `None` (pass) or a short reason string (fail).

Four concrete implementations ship:

| Class | Checks |
|---|---|
| `SpatialConstraint` | End-effector must stay outside a given set of AABB / sphere obstacles. |
| `WorkspaceConstraint` | End-effector must stay inside an AABB envelope. |
| `VelocityConstraint` | Action velocity ≤ `max_velocity`. |
| `TorqueConstraint` | Action torque ≤ `max_torque`. |

## `GroundingHook`

Orchestrator. `evaluate(action)` returns a `GroundingVerdict` listing
*every* failing constraint (no short-circuit) plus optional
reachability rejection if the kinematic model's `inverse()` returns
`None` for the target. Actions carrying an explicit `joint_angles`
vector skip the reachability check, since the joint vector already
fixes the posture.

## Heavy-dependency adapters

All three are constructed via a `from_*` classmethod that imports the
heavy dependency lazily — the subpackage imports cleanly on
deployments that do not install them.

- `Ros2Adapter.from_ros2(node)` — rclpy node-driven, collision from
  caller-supplied obstacles. Forward / inverse kinematics are
  deployment-specific and raise `NotImplementedError`.
- `MuJoCoAdapter.from_mjcf(path)` — mjcf model loading,
  `mj_forward` + `site_xpos` for forward kinematics, live `data.ncon`
  contact count for collisions.
- `CarlaAdapter.from_carla(port=2000)` — vehicle-class scenarios with
  spawn-filtered collision.

## CoherenceAgent wiring

Pass a configured hook to the agent:

```python
from director_ai.core.agent import CoherenceAgent

agent = CoherenceAgent(grounding_hook=hook)
verdict = agent.verify_physical_action(action)
```

Calling `verify_physical_action` with no hook configured raises
`RuntimeError` so callers cannot silently skip the check.

## Rust acceleration

When `backfire_kernel` is installed, `AABB.contains`, `Sphere.contains`
/ `intersects` / `intersects_aabb` and the two-link IK dispatch to
the `rust_*` FFI functions. Bit-exact on geometry, < 1e-12 ULP on
the trigonometric IK path. No change on the caller side.
