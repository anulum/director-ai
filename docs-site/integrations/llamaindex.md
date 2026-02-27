# LlamaIndex

```bash
pip install director-ai[llamaindex]
```

## As Node Postprocessor

```python
from director_ai.integrations.llamaindex import DirectorAIPostprocessor

postprocessor = DirectorAIPostprocessor(
    facts={"capital": "Paris is the capital of France."},
    threshold=0.6,
)

query_engine = index.as_query_engine(
    node_postprocessors=[postprocessor],
)
```

## Response Validation

```python
approved, score = postprocessor.validate_response(
    "What is the capital?",
    "Paris is the capital of France.",
)
```
