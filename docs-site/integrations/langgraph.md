# LangGraph

```bash
pip install director-ai[langgraph]
```

For a shared Cloud Run review service plus Vercel client wiring, use the tracked
`deploy/agent-frameworks` templates and the
[Agent Framework Deploy Pack](agent-framework-deploy.md).

## Guardrail Node

```python
from director_ai.integrations.langgraph import (
    director_ai_node,
    director_ai_conditional_edge,
)
from langgraph.graph import END, START, StateGraph

graph = StateGraph(dict)
graph.add_node("generate", llm_node)
graph.add_node("guardrail", director_ai_node(
    facts={"refund": "within 30 days"},
    on_fail="flag",  # "raise" | "flag" | "rewrite"
))
graph.add_node("retry", retry_node)
graph.add_node("output", output_node)

graph.add_edge(START, "generate")
graph.add_edge("generate", "guardrail")
graph.add_conditional_edges(
    "guardrail",
    director_ai_conditional_edge(
        approved_node="output",
        rejected_node="retry",
    ),
    ["output", "retry"],
)
graph.add_edge("output", END)
app = graph.compile()
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

The adapter fails closed during node construction if `on_fail` is not one of
those three modes. Custom `query_key` and `response_key` values must be
non-blank and distinct, so a graph cannot silently read and overwrite the same
state slot.
