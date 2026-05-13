# Causal Attribution Graph

::: director_ai.core.attribution.causal_graph.CausalAttributionGraph

::: director_ai.core.attribution.causal_graph.AttributionNode

::: director_ai.core.attribution.causal_graph.AttributionEdge

::: director_ai.core.attribution.causal_graph.build_causal_attribution_graph

## Operational Model

`build_causal_attribution_graph()` converts existing Director-AI evidence into a
validated directed acyclic graph:

- `CoherenceScore` and `ScoringEvidence` graphs connect retrieved evidence,
  claim support or contradiction, and final score contribution.
- `HaltEvidence` graphs connect retrieved evidence, halt trace coordinates, and
  counterfactual score deltas to the halt decision.
- `to_dict()` redacts raw claim, source, and fact text by default. Pass
  `include_text=True` only inside trusted operator or audit boundaries.

```python
from director_ai.core import build_causal_attribution_graph

graph = build_causal_attribution_graph(score)
for edge in graph.top_contributors(limit=3):
    print(edge.source, edge.relation, edge.weight)
```

The graph is not a speculative causal discovery algorithm. It is an auditable
representation of the causal path Director-AI used during scoring or stream
halt enforcement.
