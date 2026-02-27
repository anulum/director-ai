# LangGraph

```bash
pip install director-ai[langgraph]
```

## Guardrail Node

```python
from director_ai.integrations.langgraph import (
    director_ai_node,
    director_ai_conditional_edge,
)
from langgraph.graph import StateGraph

graph = StateGraph(dict)
graph.add_node("generate", llm_node)
graph.add_node("guardrail", director_ai_node(
    facts={"refund": "within 30 days"},
    on_fail="flag",  # "raise" | "flag" | "rewrite"
))
graph.add_node("retry", retry_node)
graph.add_node("output", output_node)

graph.add_edge("generate", "guardrail")
graph.add_conditional_edges(
    "guardrail",
    director_ai_conditional_edge(
        approved_node="output",
        rejected_node="retry",
    ),
)
```

## State Keys

After the guardrail node runs, these keys are added to state:

| Key | Type | Description |
|-----|------|-------------|
| `director_ai_score` | float | Coherence score |
| `director_ai_approved` | bool | Pass/fail |
| `director_ai_h_logical` | float | Logical divergence |
| `director_ai_h_factual` | float | Factual divergence |
| `director_ai_rewritten` | bool | True if rewrite mode activated |

## Failure Modes

- `on_fail="raise"` — raises `HallucinationError`
- `on_fail="flag"` — sets `director_ai_approved=False`, continues
- `on_fail="rewrite"` — replaces response with KB context
