# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real production-surface coverage for inter-agent handoff scoring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.handoff_scorer import HandoffScorer
from director_ai.agentic.swarm_guardian import SwarmGuardian
from director_ai.integrations.langgraph_swarm import SwarmGraphBuilder

EdgeFn = Callable[[dict[str, Any]], str]


def test_langgraph_builder_custom_handoff_scorer_controls_edge() -> None:
    """SwarmGraphBuilder should route guarded edges through its custom scorer."""
    guardian = SwarmGuardian(hallucination_threshold=0.5)
    guardian.register_agent(AgentProfile.for_role("researcher", agent_id="r1"))
    guardian.register_agent(AgentProfile.for_role("summariser", agent_id="s1"))
    scorer = HandoffScorer(threshold=0.0)
    builder = SwarmGraphBuilder(guardian, scorer=scorer)
    builder.add_guarded_edge("r1", "s1")

    edge = cast(EdgeFn, builder.get_edges()[0]["edge_fn"])
    result = edge(
        {
            "context": "shared fact context",
            "messages": [{"content": "shared fact extra"}],
        }
    )

    state = guardian.get_agent_state("r1")
    assert result == "__end__"
    assert guardian.is_quarantined("r1") is True
    assert state is not None
    assert state.handoff_count == 1
    assert state.hallucination_count == 1
    assert scorer.history[-1].from_agent == "r1"
    assert scorer.history[-1].to_agent == "s1"
    assert scorer.history[-1].grounded is False


def test_swarm_guardian_registration_and_cleanup_guardrails() -> None:
    """SwarmGuardian should enforce registration caps and clean dependencies."""
    guardian = SwarmGuardian(max_agents=2)
    researcher = AgentProfile.for_role("researcher", agent_id="r1")
    summariser = AgentProfile.for_role("summariser", agent_id="s1")

    assert guardian.register_agent(researcher) == "r1"
    with pytest.raises(ValueError, match="already registered"):
        guardian.register_agent(researcher)

    assert guardian.register_agent(summariser) == "s1"
    with pytest.raises(ValueError, match="Max agents"):
        guardian.register_agent(AgentProfile.for_role("critic", agent_id="c1"))

    result = guardian.score_handoff(
        from_agent="r1",
        to_agent="s1",
        message="shared fact",
        context="shared fact",
    )
    assert result.should_halt is False

    guardian.unregister_agent("s1")

    assert guardian.agent_count == 1
    assert guardian.list_agents() == ["r1"]
    assert guardian.get_agent_state("s1") is None
    assert guardian.quarantined_count == 0
    assert guardian.quarantine_agent("r1", "manual review") == ["r1"]
    assert guardian.quarantined_count == 1


def test_swarm_guardian_handoff_safety_event_guardrails() -> None:
    """SwarmGuardian should emit tenant-safe events for guardrail decisions."""
    guardian = SwarmGuardian(hallucination_threshold=0.4, cascade_halt=False)
    guardian.register_agent(
        AgentProfile(
            agent_id="loose",
            role="researcher",
            coherence_threshold=0.0,
        )
    )
    guardian.register_agent(AgentProfile.for_role("summariser", agent_id="s1"))

    missing = guardian.score_handoff(
        from_agent="missing",
        to_agent="s1",
        message="claim",
        context="source",
    )
    assert missing.safety_event is not None
    assert missing.score == 0.5
    assert missing.should_halt is False
    assert missing.reasons == ["source agent not registered"]
    assert missing.safety_event.policy_decision == "allow"
    assert missing.safety_event.threshold == 0.4

    no_context = guardian.score_handoff(
        from_agent="loose",
        to_agent="s1",
        message="claim",
    )
    assert no_context.safety_event is not None
    assert no_context.score == 0.5
    assert no_context.should_halt is False
    assert no_context.safety_event.tenant_safe_explanation == (
        "Swarm handoff passed the guardian check."
    )

    assert guardian.quarantine_agent("unknown", "missing") == []
    assert guardian.quarantine_agent("loose", "manual review") == ["loose"]
    assert guardian.quarantine_agent("loose", "manual review") == []
    assert guardian.is_quarantined("s1") is False

    quarantined = guardian.score_handoff(
        from_agent="loose",
        to_agent="s1",
        message="claim",
        context="source",
    )
    assert quarantined.safety_event is not None
    assert quarantined.score == 1.0
    assert quarantined.should_halt is True
    assert quarantined.reasons == ["source agent quarantined: manual review"]
    assert quarantined.safety_event.policy_decision == "halt"
    assert quarantined.safety_event.observed_score == 1.0
