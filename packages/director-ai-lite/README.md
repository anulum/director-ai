# Director-AI Lite

`director-ai-lite` is a **standalone, dependency-free** LLM streaming-halt guard:
it stops a token stream *before* a hallucination finishes generating. It installs
with zero heavy dependencies (standard library only) and does **not** require the
full `director-ai` package.

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

print(result.output)       # surviving text (halted tokens removed)
print(result.halted)       # True if the stream was stopped
print(result.halt_reason)  # why it was stopped
```

## How it works

The default path is **model-free**: each accumulated prefix is scored by a
grounding heuristic (content-word overlap against the supplied `facts`) and the
same calibrated coherence combination the full package uses in its no-model path.
The stream hard-halts on the first token whose coherence drops below `threshold`
(default `0.5`). With no `facts`, scoring stays neutral and nothing is halted.

Because the grounding heuristic and the coherence calibration match the full
package, you can upgrade to model-backed (NLI/RAG) scoring without changing the
call site.

## Upgrade to model-backed scoring

```bash
pip install "director-ai-lite[full]"
```

Then pass the full package's scorer to `StreamGuard`:

```python
from director_ai_lite import StreamGuard

guard = StreamGuard(facts=..., scorer=my_nli_scorer)  # any review(prompt, text) scorer
```

## License

Apache-2.0.
