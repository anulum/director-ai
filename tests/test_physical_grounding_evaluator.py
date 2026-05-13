# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — closed-loop physical grounding evaluator tests

from __future__ import annotations

from director_ai.core.cyber_physical import (
    AABB,
    GroundingHook,
    JointChain,
    PhysicalAction,
    PhysicalBudgetLimits,
    PhysicalGroundingEvaluator,
    SensorStateSnapshot,
    SimpleKinematicModel,
    TenantPhysicalBudget,
    Vec3,
    VelocityConstraint,
    WorkspaceConstraint,
)
from director_ai.core.guard_control import RiskEnvelope


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
            VelocityConstraint(name="velocity", max_velocity=1.0),
        ),
    )


def _snapshot(
    *,
    ref: str,
    x: float,
    y: float = 0.0,
    supported: bool = True,
    confidence: float = 1.0,
) -> SensorStateSnapshot:
    return SensorStateSnapshot(
        snapshot_ref=ref,
        sensor_id="camera-1",
        adapter_id="vision.cell",
        timestamp=1.0,
        end_effector_position=Vec3(x, y, 0.0),
        confidence=confidence,
        supported=supported,
    )


def _risk(
    *,
    reversibility: str = "reversible",
    domain: str = "physical",
) -> RiskEnvelope:
    return RiskEnvelope(
        action_category="physical",
        reversibility=reversibility,
        domain=domain,
        calibrated_threshold=0.6,
        no_go_threshold=0.8,
    )


def test_closed_loop_allows_consistent_supported_snapshots() -> None:
    evaluator = PhysicalGroundingEvaluator(
        grounding_hook=_hook(),
        high_risk_physical_deployment=True,
        state_tolerance_m=0.05,
    )
    action = PhysicalAction(actuator_id="arm", target_position=Vec3(1.0, 0.0, 0.0))

    verdict = evaluator.evaluate(
        action=action,
        risk_envelope=_risk(),
        pre_perception=_snapshot(ref="sensor://pre", x=0.0),
        pre_simulation=_snapshot(ref="sim://pre", x=0.02),
        post_perception=_snapshot(ref="sensor://post", x=1.0),
        post_simulation=_snapshot(ref="sim://post", x=1.01),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "allow"
    assert verdict.reason == "physical_grounding_consistent"
    assert verdict.pre_action.allowed
    assert verdict.post_action_verified is True
    assert verdict.decision.verifier_signals[0].modality == "physical"


def test_pre_action_perception_simulator_mismatch_warns_by_default() -> None:
    evaluator = PhysicalGroundingEvaluator(
        grounding_hook=_hook(),
        high_risk_physical_deployment=False,
        state_tolerance_m=0.05,
    )

    verdict = evaluator.evaluate(
        action=PhysicalAction(actuator_id="arm", target_position=Vec3(0.5, 0.0, 0.0)),
        risk_envelope=_risk(),
        pre_perception=_snapshot(ref="sensor://pre", x=0.0),
        pre_simulation=_snapshot(ref="sim://pre", x=0.5),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "warn"
    assert verdict.reason == "physical_state_mismatch"
    assert verdict.violations[0].stage == "pre_action"


def test_post_action_mismatch_blocks_when_high_risk_flag_enabled() -> None:
    evaluator = PhysicalGroundingEvaluator(
        grounding_hook=_hook(),
        high_risk_physical_deployment=True,
        state_tolerance_m=0.05,
    )

    verdict = evaluator.evaluate(
        action=PhysicalAction(actuator_id="arm", target_position=Vec3(1.0, 0.0, 0.0)),
        risk_envelope=_risk(),
        pre_perception=_snapshot(ref="sensor://pre", x=0.0),
        pre_simulation=_snapshot(ref="sim://pre", x=0.0),
        post_perception=_snapshot(ref="sensor://post", x=1.0),
        post_simulation=_snapshot(ref="sim://post", x=1.5),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "block"
    assert verdict.reason == "physical_state_mismatch"
    assert verdict.violations[0].stage == "post_action"


def test_unsupported_sensor_reports_unsupported_status() -> None:
    evaluator = PhysicalGroundingEvaluator(grounding_hook=_hook())

    verdict = evaluator.evaluate(
        action=PhysicalAction(actuator_id="arm", target_position=Vec3(0.5, 0.0, 0.0)),
        risk_envelope=_risk(),
        pre_perception=_snapshot(ref="sensor://pre", x=0.0, supported=False),
        pre_simulation=_snapshot(ref="sim://pre", x=0.0),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "warn"
    assert verdict.reason == "physical_sensor_unsupported"
    assert verdict.violations[0].status == "unsupported"


def test_irreversible_action_requires_no_go_review() -> None:
    evaluator = PhysicalGroundingEvaluator(
        grounding_hook=_hook(),
        high_risk_physical_deployment=True,
    )

    verdict = evaluator.evaluate(
        action=PhysicalAction(actuator_id="arm", target_position=Vec3(0.5, 0.0, 0.0)),
        risk_envelope=_risk(reversibility="irreversible"),
        pre_perception=_snapshot(ref="sensor://pre", x=0.0),
        pre_simulation=_snapshot(ref="sim://pre", x=0.0),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "block"
    assert verdict.reason == "no_go_irreversible_risk"
    assert verdict.requires_human_review is True


def test_sensor_fusion_budget_blocks_before_snapshot_comparison() -> None:
    budget = TenantPhysicalBudget(
        PhysicalBudgetLimits(
            max_action_validations=10,
            max_inverse_kinematics=10,
            max_simulation_checks=10,
            max_sensor_fusion=0,
        )
    )
    evaluator = PhysicalGroundingEvaluator(
        grounding_hook=_hook(),
        budget=budget,
    )

    verdict = evaluator.evaluate(
        action=PhysicalAction(actuator_id="arm", target_position=Vec3(0.5, 0.0, 0.0)),
        risk_envelope=_risk(),
        pre_perception=_snapshot(ref="sensor://pre", x=0.0),
        pre_simulation=_snapshot(ref="sim://pre", x=0.0),
        tenant_id="tenant-a",
    )

    assert verdict.decision.decision == "block"
    assert verdict.reason == "physical_budget_exceeded"
    assert verdict.violations[0].constraint == "budget:sensor_fusion"
    assert budget.snapshot("tenant-a")["sensor_fusion"] == 0
