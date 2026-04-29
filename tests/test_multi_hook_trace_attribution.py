# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multi-hook trace attribution tests

"""Trace attribution tests across streaming, swarm, and physical hooks."""

from __future__ import annotations

import json

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.swarm_guardian import SwarmGuardian
from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel
from director_ai.core.cyber_physical import (
    AABB,
    GroundingHook,
    JointChain,
    PhysicalAction,
    SimpleKinematicModel,
    Vec3,
    WorkspaceConstraint,
)
from director_ai.ui.config_wizard import build_trace_explorer


def _streaming_event() -> dict:
    store = GroundTruthStore()
    store.add("sky", "The sky is blue")
    scorer = CoherenceScorer(threshold=0.5, ground_truth_store=store, use_nli=False)
    kernel = StreamingKernel(hard_limit=0.99)

    session = kernel.stream_tokens(
        iter(["wrong"]),
        lambda _text: 0.1,
        scorer=scorer,
        prompt="sky",
    )

    assert session.halted is True
    event = session.safety_events[0].to_dict()
    assert event["trace_attribution"]["token_offset"] == 0
    return event


def _swarm_event() -> dict:
    guardian = SwarmGuardian()
    guardian.register_agent(
        AgentProfile.for_role(
            "researcher",
            agent_id="researcher-1",
            coherence_threshold=0.9,
        )
    )

    result = guardian.score_handoff(
        "researcher-1",
        "writer-1",
        "unrelated fabricated claim",
        "Paris France capital",
    )

    assert result.should_halt is True
    assert result.safety_event is not None
    return result.safety_event.to_dict()


def _physical_event() -> dict:
    chain = JointChain(base=Vec3(0, 0, 0), link_lengths=(1.0, 1.0))
    model = SimpleKinematicModel(chain=chain)
    room = AABB(min_corner=Vec3(-5, -5, -5), max_corner=Vec3(5, 5, 5))
    hook = GroundingHook(
        model=model,
        constraints=(WorkspaceConstraint(name="room", envelope=room),),
    )
    action = PhysicalAction(
        actuator_id="arm",
        target_position=Vec3(1000.0, 0.0, 0.0),
        velocity_magnitude=0.1,
        torque_magnitude=0.5,
    )

    verdict = hook.evaluate(action)

    assert verdict.allowed is False
    assert verdict.safety_event is not None
    return verdict.safety_event.to_dict()


class TestMultiHookTraceAttribution:
    def test_real_hook_events_keep_attribution_stable(self):
        payload = {
            "safety_events": [
                _streaming_event(),
                _swarm_event(),
                _physical_event(),
            ]
        }

        summary, rows, detail = build_trace_explorer(json.dumps(payload))

        assert "Events: 3" in summary
        assert "Halted: yes" in summary
        assert [row[1] for row in rows] == ["streaming", "swarm", "cyber_physical"]
        assert [row[5] for row in rows] == [
            "streaming.kernel",
            "swarm.guardian.handoff",
            "cyber_physical.grounding",
        ]
        assert [row[3] for row in rows] == ["halt", "halt", "block"]
        assert rows[0][6].startswith("hard_limit")
        assert rows[1][6] == "swarm_handoff_halt"
        assert rows[2][6] == "physical_constraint_violation"
        assert detail["trace_attribution"]["token_offset"] == 0
        assert detail["scopes"] == ["cyber_physical", "streaming", "swarm"]
