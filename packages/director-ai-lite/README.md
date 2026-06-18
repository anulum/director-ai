# Director-AI Lite

`director-ai-lite` is the small-install front door for Director-AI's streaming
halt surface. It exposes the same `StreamGuard` implementation as
`director_ai.lite`, plus a one-call `guard()` helper.

```bash
pip install director-ai-lite
```

```python
from director_ai_lite import guard

result = guard(
    token_stream,
    facts={"capital": "Paris is the capital of France."},
    prompt="What is the capital of France?",
)

print(result.output)
print(result.halted)
```

The default path is model-free: it uses Director-AI's heuristic scorer, grounded
facts, and streaming kernel without downloading an NLI model. For model-backed
accuracy, install `director-ai-lite[nli]` and pass a configured Director-AI
scorer to `StreamGuard`.

This package intentionally remains a thin distribution wrapper. The canonical
runtime implementation lives in `director_ai.lite` so bug fixes, type behavior,
and halt semantics stay identical between the full and Lite installs.
