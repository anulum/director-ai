# CrewAI

```bash
pip install director-ai[crewai]
```

## Agent Tool

```python
from director_ai.integrations.crewai import DirectorAITool
from crewai import Agent

guard_tool = DirectorAITool(
    facts={"company": "Founded in 2020", "revenue": "$10M ARR"},
)

agent = Agent(
    role="Research Analyst",
    tools=[guard_tool],
    goal="Verify all claims before reporting",
)
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
