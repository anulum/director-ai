# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Physical Grounding Closed Loop Tests
"""Module-specific tests for cyber-physical closed-loop grounding."""

from __future__ import annotations

import pytest

import director_ai.core.cyber_physical.closed_loop as closed_loop
from director_ai.core.cyber_physical import (
    AABB,
    GroundingHook,
    GroundingVerdict,
    JointChain,
    PhysicalAction,
    PhysicalBudgetLimits,
    PhysicalGroundingEvaluator,
    PhysicalGroundingLoop,
    PhysicalGroundingViolation,
    SensorStateSnapshot,
    SimpleKinematicModel,
    TenantPhysicalBudget,
    Vec3,
    Violation,
    WorkspaceConstraint,
)
from director_ai.core.guard_control import RiskEnvelope


def test_sensor_state_snapshot_validates_required_fields() -> None:
    base = {
        "snapshot_ref": "sensor://pre",
        "sensor_id": "camera-1",
        "adapter_id": "vision.cell",
        "timestamp": 1.0,
        "end_effector_position": Vec3(0.0, 0.0, 0.0),
    }

    cases = [
        ({"snapshot_ref": " "}, "snapshot_ref"),
        ({"sensor_id": " "}, "sensor_id"),
        ({"adapter_id": " "}, "adapter_id"),
        ({"timestamp": -0.1}, "timestamp"),
        ({"confidence": 1.1}, "confidence"),
    ]
    for override, message in cases:
        with pytest.raises(ValueError, match=message):
            SensorStateSnapshot(**(base | override))


def test_sensor_state_snapshot_exports_metadata_without_payloads() -> None:
    snapshot = _snapshot("sensor://pre", 0.25, confidence=0.75)

    assert snapshot.to_dict() == {
        "snapshot_ref": "sensor://pre",
        "sensor_id": "camera-1",
        "adapter_id": "vision.cell",
        "timestamp": 1.0,
        "position_x": 0.25,
        "position_y": 0.0,
        "position_z": 0.0,
        "confidence": 0.75,
        "supported": True,
        "status_detail": "",
    }


def test_physical_grounding_violation_validates_required_fields() -> None:
    base = {
        "stage": "pre_action",
        "status": "mismatch",
        "constraint": "workspace",
        "reason": "outside cell",
        "evidence_refs": ("ref",),
    }

    for override, message in [
        ({"stage": " "}, "stage"),
        ({"constraint": " "}, "constraint"),
        ({"reason": " "}, "reason"),
    ]:
        with pytest.raises(ValueError, match=message):
            PhysicalGroundingViolation(**(base | override))


def test_physical_grounding_violation_normalizes_evidence_refs() -> None:
    violation = PhysicalGroundingViolation(
        stage="pre_action",
        status="mismatch",
        constraint="workspace",
        reason="outside cell",
        evidence_refs=(123, "ref"),
    )

    assert violation.evidence_refs == ("123", "ref")


def test_evaluator_rejects_negative_state_tolerance() -> None:
    with pytest.raises(ValueError, match="state_tolerance_m"):
        PhysicalGroundingEvaluator(
            grounding_hook=_hook(),
            state_tolerance_m=-0.01,
        )


def test_pre_action_grounding_hook_violations_are_exposed() -> None:
    action = _action(1.0)
    evaluator = PhysicalGroundingEvaluator(
        grounding_hook=_RejectingHook(
            GroundingVerdict(
                action=action,
                allowed=False,
                violations=(Violation("workspace", "outside guarded cell"),),
            )
        ),
    )

    verdict = evaluator.evaluate(
        action=action,
        risk_envelope=_risk(),
        pre_perception=_snapshot("sensor://pre", 0.0),
        pre_simulation=_snapshot("sim://pre", 0.0),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "warn"
    assert verdict.violations[0].constraint == "workspace"
    assert verdict.violations[0].evidence_refs == ("physical:workspace",)


def test_evaluate_requires_complete_post_action_snapshot_pair() -> None:
    evaluator = PhysicalGroundingEvaluator(grounding_hook=_hook())

    verdict = evaluator.evaluate(
        action=_action(1.0),
        risk_envelope=_risk(),
        pre_perception=_snapshot("sensor://pre", 0.0),
        pre_simulation=_snapshot("sim://pre", 0.0),
        post_perception=_snapshot("sensor://post", 1.0),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "warn"
    assert verdict.reason == "physical_sensor_unsupported"
    assert verdict.violations[0].constraint == "post_action_snapshot_pair"


def test_budget_available_path_allows_consistent_evaluation() -> None:
    budget = TenantPhysicalBudget(
        PhysicalBudgetLimits(
            max_action_validations=10,
            max_inverse_kinematics=10,
            max_simulation_checks=10,
            max_sensor_fusion=1,
        )
    )
    evaluator = PhysicalGroundingEvaluator(grounding_hook=_hook(), budget=budget)

    verdict = evaluator.evaluate(
        action=_action(1.0),
        risk_envelope=_risk(),
        pre_perception=_snapshot("sensor://pre", 0.0),
        pre_simulation=_snapshot("sim://pre", 0.0),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "allow"
    assert budget.snapshot("tenant-a")["sensor_fusion"] == 1


def test_budget_reason_helper_paths_block_and_explain_budget_status() -> None:
    evaluator = PhysicalGroundingEvaluator(grounding_hook=_hook())
    violation = PhysicalGroundingViolation(
        stage="pre_action",
        status="budget",
        constraint="budget:sensor_fusion",
        reason="sensor budget exhausted",
        evidence_refs=("physical_budget:sensor_fusion",),
    )

    assert closed_loop._reason_for((violation,)) == "physical_budget_exceeded"
    assert (
        evaluator._decision_for(
            reason="physical_budget_exceeded",
            risk_envelope=_risk(),
            violations=(violation,),
        )
        == "block"
    )


def test_irreversible_mismatch_blocks_without_no_go_second_pass_allowance() -> None:
    evaluator = PhysicalGroundingEvaluator(grounding_hook=_hook())

    verdict = evaluator.evaluate(
        action=_action(1.0),
        risk_envelope=_risk(reversibility="irreversible"),
        pre_perception=_snapshot("sensor://pre", 0.0),
        pre_simulation=_snapshot("sim://pre", 0.4),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "block"
    assert verdict.reason == "no_go_irreversible_risk"
    assert verdict.requires_human_review is True


def test_loop_without_execute_callback_still_runs_post_action_verification() -> None:
    evaluator = PhysicalGroundingEvaluator(grounding_hook=_hook())
    loop = PhysicalGroundingLoop(evaluator=evaluator)

    result = loop.run(
        action=_action(1.0),
        risk_envelope=_risk(),
        pre_perception=_snapshot("sensor://pre", 0.0),
        pre_simulation=_snapshot("sim://pre", 0.0),
        post_perception=lambda: _snapshot("sensor://post", 1.0),
        post_simulation=lambda: _snapshot("sim://post", 1.0),
        tenant_id="tenant-a",
    )

    assert result.action_executed is False
    assert result.pre_evaluation.decision.decision == "allow"
    assert result.final_evaluation.post_action_verified is True


class _RejectingHook:
    def __init__(self, verdict: GroundingVerdict) -> None:
        self._verdict = verdict

    def evaluate(self, action: PhysicalAction, *, tenant_id: str = "") -> GroundingVerdict:
        _ = action, tenant_id
        return self._verdict


def _hook() -> GroundingHook:
    return GroundingHook(
        model=SimpleKinematicModel(
            chain=JointChain(base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0))
        ),
        constraints=(
            WorkspaceConstraint(
                name="cell",
                envelope=AABB(
                    min_corner=Vec3(-2.0, -2.0, -0.1),
                    max_corner=Vec3(2.0, 2.0, 0.1),
                ),
            ),
        ),
    )


def _snapshot(
    ref: str,
    x: float,
    *,
    confidence: float = 1.0,
    supported: bool = True,
) -> SensorStateSnapshot:
    return SensorStateSnapshot(
        snapshot_ref=ref,
        sensor_id="camera-1",
        adapter_id="vision.cell",
        timestamp=1.0,
        end_effector_position=Vec3(x, 0.0, 0.0),
        confidence=confidence,
        supported=supported,
    )


def _action(x: float) -> PhysicalAction:
    return PhysicalAction(actuator_id="arm", target_position=Vec3(x, 0.0, 0.0))


def _risk(*, reversibility: str = "reversible") -> RiskEnvelope:
    return RiskEnvelope(
        action_category="physical",
        reversibility=reversibility,
        domain="physical",
        calibrated_threshold=0.6,
        no_go_threshold=0.8,
    )
