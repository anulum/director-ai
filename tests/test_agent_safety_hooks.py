# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CoherenceAgent safety-hook wiring tests

"""Verifies the opt-in wiring of cyber-physical grounding,
simulation containment and cross-org passport checks into
:class:`CoherenceAgent`. Default None → zero change to existing
behaviour; configured → each module triggers at the expected
boundary."""

from __future__ import annotations

import pytest

from director_ai.core.agent import CoherenceAgent
from director_ai.core.containment import (
    BreakoutDetector,
    ContainmentAttestor,
    ContainmentGuard,
)
from director_ai.core.cyber_physical import (
    AABB,
    GroundingHook,
    JointChain,
    PhysicalAction,
    SimpleKinematicModel,
    Vec3,
    WorkspaceConstraint,
)
from director_ai.core.zk_attestation import (
    CommitmentBackend,
    CrossOrgPassport,
    MinimumCoherence,
    PassportIssuer,
    PassportVerifier,
)

_KEY = b"x" * 32


def _agent(**extra):
    """Build a CoherenceAgent with the MockGenerator and optional
    safety hooks attached."""
    return CoherenceAgent(**extra)


# --- containment wiring --------------------------------------------


class TestContainmentWiring:
    def _make_guard_and_anchor(
        self, scope="sandbox"
    ):
        attestor = ContainmentAttestor(key=_KEY, issuer="host://unit")
        guard = ContainmentGuard(attestor=attestor, detector=BreakoutDetector())
        anchor = attestor.mint(session_id="sess-1", scope=scope)
        return guard, anchor

    def test_default_no_wiring_preserves_behaviour(self):
        agent = _agent()
        result = agent.process("Paris is the capital of France.")
        # The baseline mock path yields a ReviewResult; containment
        # is None so no extra processing happens.
        assert not result.output.startswith("[CONTAINMENT-BLOCK]")

    def test_mismatched_guard_and_anchor_rejected(self):
        guard, _ = self._make_guard_and_anchor()
        with pytest.raises(ValueError, match="together"):
            _agent(containment_guard=guard, containment_anchor=None)

    def test_clean_output_passes_guard(self):
        guard, anchor = self._make_guard_and_anchor()
        agent = _agent(containment_guard=guard, containment_anchor=anchor)
        result = agent.process("Paris is the capital of France.")
        assert not result.output.startswith("[CONTAINMENT-BLOCK]")

    def test_injection_in_output_blocks(self):
        # Manipulate the guard's detector so any output is blocked —
        # this proves the wiring actually calls the guard.
        guard, anchor = self._make_guard_and_anchor()

        from director_ai.core.containment import BreakoutFinding

        class _AlwaysBlock(BreakoutDetector):
            def scan(self, event, anchored_scope, claimed_scope=None):
                del event, anchored_scope, claimed_scope
                return [
                    BreakoutFinding(
                        category="policy",
                        severity="high",
                        detail="injected test failure",
                    ),
                ]

        guard_always = ContainmentGuard(
            attestor=guard.attestor, detector=_AlwaysBlock()
        )
        agent = _agent(containment_guard=guard_always, containment_anchor=anchor)
        result = agent.process("Paris is the capital of France.")
        assert result.output.startswith("[CONTAINMENT-BLOCK]")
        assert result.halted is True
        assert result.halt_evidence is not None
        assert result.halt_evidence.reason == "containment_block"
        assert "policy:high" in result.halt_evidence.suggested_action

    def test_bad_anchor_blocks(self):
        guard, anchor = self._make_guard_and_anchor()
        # Corrupt the anchor MAC so the guard's first-line check
        # rejects it — the agent must surface that.
        from director_ai.core.containment import RealityAnchor

        tampered = RealityAnchor(
            session_id=anchor.session_id,
            scope=anchor.scope,
            issuer=anchor.issuer,
            created_at=anchor.created_at,
            nonce=anchor.nonce,
            mac="0" * 64,
        )
        agent = _agent(containment_guard=guard, containment_anchor=tampered)
        result = agent.process("hello")
        assert result.output.startswith("[CONTAINMENT-BLOCK]")
        assert result.halt_evidence is not None
        assert "mac_mismatch" in result.halt_evidence.suggested_action


# --- grounding wiring ----------------------------------------------


class TestGroundingWiring:
    def _make_hook(self) -> GroundingHook:
        chain = JointChain(base=Vec3(0, 0, 0), link_lengths=(1.0, 1.0))
        model = SimpleKinematicModel(chain=chain)
        box = AABB(min_corner=Vec3(-5, -5, -5), max_corner=Vec3(5, 5, 5))
        return GroundingHook(
            model=model,
            constraints=(WorkspaceConstraint(name="room", envelope=box),),
        )

    def test_missing_hook_raises(self):
        agent = _agent()
        action = PhysicalAction(
            actuator_id="arm",
            target_position=Vec3(1, 0, 0),
            velocity_magnitude=0.1,
            torque_magnitude=0.5,
        )
        with pytest.raises(RuntimeError, match="grounding_hook"):
            agent.verify_physical_action(action)

    def test_allowed_action_returns_allow(self):
        agent = _agent(grounding_hook=self._make_hook())
        action = PhysicalAction(
            actuator_id="arm",
            target_position=Vec3(1.0, 0.0, 0.0),
            velocity_magnitude=0.1,
            torque_magnitude=0.5,
        )
        verdict = agent.verify_physical_action(action)
        assert verdict.allowed is True

    def test_out_of_workspace_action_rejected(self):
        agent = _agent(grounding_hook=self._make_hook())
        action = PhysicalAction(
            actuator_id="arm",
            target_position=Vec3(1000.0, 0.0, 0.0),
            velocity_magnitude=0.1,
            torque_magnitude=0.5,
        )
        verdict = agent.verify_physical_action(action)
        assert verdict.allowed is False
        assert any(
            v.constraint == "room" for v in verdict.violations
        )


# --- passport wiring -----------------------------------------------


class TestPassportWiring:
    def _setup(self):
        issuer = PassportIssuer(key=_KEY, issuing_org="org://source")
        verifier = PassportVerifier(
            issuer_keys={"org://source": _KEY},
            backends={"commitment": CommitmentBackend(key=_KEY)},
        )
        samples = [
            {"coherence": 0.95, "halted": False, "breakout": False}
            for _ in range(32)
        ]
        return issuer, verifier, samples

    def test_missing_verifier_raises(self):
        agent = _agent()
        # Build a real passport that the default agent can't check.
        issuer, _, samples = self._setup()
        passport = issuer.issue(
            agent_id="a",
            samples=samples,
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
        )
        with pytest.raises(RuntimeError, match="passport_verifier"):
            agent.verify_passport(passport)

    def test_valid_passport_accepted(self):
        issuer, verifier, samples = self._setup()
        agent = _agent(passport_verifier=verifier)
        passport = issuer.issue(
            agent_id="a",
            samples=samples,
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
        )
        verdict = agent.verify_passport(passport)
        assert verdict.accepted is True

    def test_tampered_passport_rejected(self):
        issuer, verifier, samples = self._setup()
        agent = _agent(passport_verifier=verifier)
        passport = issuer.issue(
            agent_id="a",
            samples=samples,
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
        )
        tampered = CrossOrgPassport(
            agent_id=passport.agent_id,
            issuing_org=passport.issuing_org,
            created_at=passport.created_at,
            entries=passport.entries,
            mac="0" * 64,
        )
        verdict = agent.verify_passport(tampered)
        assert verdict.accepted is False
        assert verdict.signature_ok is False
