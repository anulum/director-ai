<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI — CrewAI integration -->

# CrewAI

```bash
pip install director-ai[crewai]
```

For central Cloud Run review service deployment and Vercel AI SDK client wiring,
use the tracked `deploy/agent-frameworks` templates and the
[Agent Framework Deploy Pack](agent-framework-deploy.md).

## Agent Tool

```python
from director_ai.integrations.crewai import DirectorAITool
from crewai import Agent, Crew, Task

guard_tool = DirectorAITool(
    facts={"refund": "Customers can request a refund within 30 days."},
)

agent = Agent(
    role="Research Analyst",
    tools=[guard_tool],
    goal="Verify all claims before reporting",
    backstory="Checks factual claims against the approved knowledge base.",
)

task = Task(
    description="Answer the refund-policy question and verify the final claim.",
    expected_output="A verified refund-policy answer.",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

## Tool Input Format

The agent sends `"query | claim"` separated by a pipe:

```
What year was the company founded? | The company was founded in 2019.
```

Returns:

```
[REJECTED] Coherence: 0.320 (logical: 0.450, factual: 0.680)
```

## Direct API

```python
result = guard_tool.check(
    "When was the company founded?",
    "The company was founded in 2020.",
)
print(result["approved"])  # True
print(result["score"])  # ~0.85
```

## Swarm handoff scoring

For deployments with the advanced swarm features installed, `CrewGuardian`
wraps a `SwarmGuardian` without importing CrewAI itself. Its optional `scorer`
argument supplies the actual handoff scorer:

```python
from director_ai.agentic import AgentProfile, HandoffScorer, SwarmGuardian
from director_ai.integrations.crewai_swarm import CrewGuardian

guardian = SwarmGuardian()
guardian.register_agent(AgentProfile.for_role("researcher", agent_id="researcher"))
scorer = HandoffScorer(threshold=0.2)
crew_guardian = CrewGuardian(guardian, scorer=scorer)
result = crew_guardian.guard_task_output(
    agent_id="researcher",
    task_output="The refund window is 30 days.",
    context="The refund window is 30 days.",
    next_agent_id="writer",
)
```

A supplied scorer controls the grounding decision and records the handoff in
its history. The guardian still tracks agent state and dependencies, and the
wrapper retains its quarantine policy and counters. Without a supplied scorer,
the registered agent's existing guardian threshold remains in effect. An
already quarantined agent is blocked before scoring. `guard_crew_output` uses
the same scoring path for a final crew result.
