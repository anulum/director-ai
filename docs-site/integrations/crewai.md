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
print(result["score"])     # ~0.85
```
