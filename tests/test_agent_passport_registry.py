# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — agent passport registry tests

from __future__ import annotations

from director_ai.core.agent_identity import (
    AgentPassportRegistry,
    PassportSigner,
)
from director_ai.core.guard_control import RiskEnvelope


class _ManualClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _registry(clock: _ManualClock | None = None) -> AgentPassportRegistry:
    return AgentPassportRegistry(
        signer=PassportSigner(
            active_key=b"p" * 32,
            active_key_id="k1",
            default_ttl_seconds=60.0,
            clock=clock or _ManualClock(),
        )
    )


def _risk(
    *,
    action_category: str = "tool",
    domain: str = "regulated",
    reversibility: str = "reversible",
) -> RiskEnvelope:
    return RiskEnvelope(
        action_category=action_category,
        reversibility=reversibility,
        domain=domain,
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    )


def test_registry_issues_and_allows_matching_capability() -> None:
    registry = _registry()
    passport = registry.issue_passport(
        agent_id="tenant-a/worker/tool",
        role="worker",
        tenant_id="tenant-a",
        capabilities=("tool:search",),
    )

    verdict = registry.evaluate_action(
        passport=passport,
        required_capability="tool:search",
        risk_envelope=_risk(),
        event_ref="event://tool-1",
    )

    assert verdict.accepted is True
    assert verdict.guard_decision.decision == "allow"
    assert verdict.guard_decision.verifier_signals[0].modality == "identity"
    assert verdict.guard_decision.attributes["capability"] == "tool:search"


def test_expired_passport_fails_closed() -> None:
    clock = _ManualClock()
    registry = _registry(clock)
    passport = registry.issue_passport(
        agent_id="tenant-a/worker/tool",
        role="worker",
        tenant_id="tenant-a",
        capabilities=("tool:search",),
        ttl_seconds=10.0,
    )
    clock.now = 2_000.0

    verdict = registry.evaluate_action(
        passport=passport,
        required_capability="tool:search",
        risk_envelope=_risk(),
        event_ref="event://tool-2",
    )

    assert verdict.accepted is False
    assert verdict.reason == "passport_invalid"
    assert verdict.guard_decision.decision == "block"
    assert "expired" in verdict.detail


def test_revoked_passport_fails_closed_and_rotation_preserves_old_key() -> None:
    registry = _registry()
    old = registry.issue_passport(
        agent_id="tenant-a/worker/tool",
        role="worker",
        tenant_id="tenant-a",
        capabilities=("tool:search",),
    )
    registry.rotate_signer(new_active_key=b"q" * 32, new_active_key_id="k2")
    registry.revoke(old, reason="operator_rotation")

    blocked = registry.evaluate_action(
        passport=old,
        required_capability="tool:search",
        risk_envelope=_risk(),
        event_ref="event://tool-3",
    )
    fresh = registry.issue_passport(
        agent_id="tenant-a/worker/tool",
        role="worker",
        tenant_id="tenant-a",
        capabilities=("tool:search",),
    )

    assert blocked.accepted is False
    assert blocked.reason == "passport_revoked"
    assert blocked.guard_decision.decision == "block"
    assert fresh.key_id == "k2"


def test_capability_mismatch_blocks_tool_physical_and_training_actions() -> None:
    registry = _registry()
    passport = registry.issue_passport(
        agent_id="tenant-a/worker/text",
        role="worker",
        tenant_id="tenant-a",
        capabilities=("text:review",),
    )

    for category in ("tool", "physical", "training"):
        verdict = registry.evaluate_action(
            passport=passport,
            required_capability=f"{category}:execute",
            risk_envelope=_risk(action_category=category),
            event_ref=f"event://{category}",
        )

        assert verdict.accepted is False
        assert verdict.reason == "capability_mismatch"
        assert verdict.guard_decision.decision == "block"
        assert verdict.guard_decision.attributes["capability"] == f"{category}:execute"


def test_no_go_policy_can_escalate_identity_decision() -> None:
    registry = _registry()
    passport = registry.issue_passport(
        agent_id="tenant-a/worker/physical",
        role="worker",
        tenant_id="tenant-a",
        capabilities=("physical:act",),
    )

    verdict = registry.evaluate_action(
        passport=passport,
        required_capability="physical:act",
        risk_envelope=_risk(action_category="physical", reversibility="irreversible"),
        event_ref="event://physical-1",
    )

    assert verdict.accepted is False
    assert verdict.reason == "no_go_irreversible_risk"
    assert verdict.guard_decision.decision == "block"


def test_event_linked_coherence_history_exports_without_payloads() -> None:
    registry = _registry()
    passport = registry.issue_passport(
        agent_id="tenant-a/worker/tool",
        role="worker",
        tenant_id="tenant-a",
        capabilities=("tool:search",),
    )

    registry.record_coherence(
        agent_id=passport.agent_id,
        event_ref="event://score-1",
        coherence_score=0.91,
        decision="allow",
    )
    registry.record_coherence(
        agent_id=passport.agent_id,
        event_ref="event://score-2",
        coherence_score=0.62,
        decision="warn",
    )
    exported = registry.export_agent(passport.agent_id)
    rendered = repr(exported)

    assert exported["agent_id"] == passport.agent_id
    assert exported["capabilities"] == ["tool:search"]
    assert exported["coherence_history"][0]["event_ref"] == "event://score-1"
    assert exported["coherence_summary"]["count"] == 2
    assert exported["coherence_summary"]["minimum"] == 0.62
    assert "prompt" not in rendered.lower()
    assert "credential" not in rendered.lower()
    assert "secret" not in rendered.lower()
