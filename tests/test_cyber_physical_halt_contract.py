# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - cyber-physical halt contract tests

"""Cyber-physical verdicts feeding HaltMonitor."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from director_ai.core.cyber_physical import (
    AABB,
    GroundingHook,
    GroundingVerdict,
    JointChain,
    PhysicalAction,
    SimpleKinematicModel,
    TorqueConstraint,
    Vec3,
    VelocityConstraint,
    WorkspaceConstraint,
)
from director_ai.core.runtime.kernel import HaltMonitor

_HALT_TEXT = "[HALT: COHERENCE BELOW THRESHOLD]"

_COORDS = st.floats(
    min_value=-2.5,
    max_value=2.5,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_MAGNITUDES = st.floats(
    min_value=0.0,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _hook() -> GroundingHook:
    return GroundingHook(
        model=SimpleKinematicModel(
            chain=JointChain(
                base=Vec3(0.0, 0.0, 0.0),
                link_lengths=(1.0, 1.0),
            ),
        ),
        constraints=(
            WorkspaceConstraint(
                name="bench_workspace",
                envelope=AABB(
                    min_corner=Vec3(-1.0, -1.0, -0.1),
                    max_corner=Vec3(1.0, 1.0, 0.1),
                ),
            ),
            VelocityConstraint(name="bench_velocity", max_velocity=1.0),
            TorqueConstraint(name="bench_torque", max_torque=2.0),
        ),
    )


def _verdict_and_score(action: PhysicalAction) -> tuple[GroundingVerdict, float]:
    verdict = _hook().evaluate(action)
    assert verdict.safety_event is not None
    return verdict, verdict.safety_event.observed_score


def _monitor_result(score: float) -> tuple[str, bool]:
    monitor = HaltMonitor(hard_limit=0.5)
    result = monitor.stream_output(iter(("MOVE",)), lambda _: score)
    return result, monitor.is_active


@given(x=_COORDS, y=_COORDS, velocity=_MAGNITUDES, torque=_MAGNITUDES)
@settings(max_examples=80, deadline=None)
def test_grounding_score_controls_halt_monitor(
    x: float,
    y: float,
    velocity: float,
    torque: float,
) -> None:
    action = PhysicalAction(
        actuator_id="arm",
        target_position=Vec3(x, y, 0.0),
        velocity_magnitude=velocity,
        torque_magnitude=torque,
    )
    verdict, score = _verdict_and_score(action)
    result, active = _monitor_result(score)

    if verdict.allowed:
        assert result == "MOVE"
        assert active
    else:
        assert result == _HALT_TEXT
        assert not active


async def _run_action(action: PhysicalAction) -> tuple[bool, str, bool]:
    verdict, score = await asyncio.to_thread(_verdict_and_score, action)
    result, active = await asyncio.to_thread(_monitor_result, score)
    return verdict.allowed, result, active


@pytest.mark.asyncio
async def test_parallel_physical_actions_keep_separate_halt_state() -> None:
    allowed_action = PhysicalAction(
        actuator_id="arm_ok",
        target_position=Vec3(0.5, 0.0, 0.0),
        velocity_magnitude=0.1,
        torque_magnitude=0.1,
    )
    blocked_action = PhysicalAction(
        actuator_id="arm_block",
        target_position=Vec3(1.5, 0.0, 0.0),
        velocity_magnitude=3.0,
        torque_magnitude=0.1,
    )

    allowed, blocked = await asyncio.gather(
        _run_action(allowed_action),
        _run_action(blocked_action),
    )

    assert allowed == (True, "MOVE", True)
    assert blocked == (False, _HALT_TEXT, False)


@pytest.mark.asyncio
async def test_cancelled_physical_action_never_reaches_halt_monitor() -> None:
    monitor = HaltMonitor(hard_limit=0.5)
    action = PhysicalAction(
        actuator_id="arm_pending",
        target_position=Vec3(0.5, 0.0, 0.0),
        velocity_magnitude=0.1,
        torque_magnitude=0.1,
    )
    gate = asyncio.Event()
    calls = 0

    async def evaluate_after_gate() -> str:
        nonlocal calls
        await gate.wait()
        calls += 1
        verdict, score = _verdict_and_score(action)
        assert verdict.allowed
        return monitor.stream_output(iter(("MOVE",)), lambda _: score)

    task = asyncio.create_task(evaluate_after_gate())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == 0
    assert monitor.is_active
