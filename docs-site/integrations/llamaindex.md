# LlamaIndex

```bash
pip install director-ai
```

The `llamaindex` extra is intentionally empty while upstream
`llama-index-core` pulls an `nltk` release with no patched security path. Install
a safe LlamaIndex SDK version separately when upstream resolves that dependency.

For central Cloud Run review service deployment and Vercel AI SDK client wiring,
use the tracked `deploy/agent-frameworks` templates and the
[Agent Framework Deploy Pack](agent-framework-deploy.md).

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

response = query_engine.query("What is the capital of France?")
```

## Response Validation

```python
approved, score = postprocessor.validate_response(
    "What is the capital?",
    "Paris is the capital of France.",
)
```
