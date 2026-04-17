---
title: Director-AI
emoji: "🛡"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Real-time LLM hallucination guardrail
---

# Director-AI Demo

Real-time LLM hallucination guardrail with NLI + RAG fact-checking and
token-level streaming halt.

## Apps

| Script | What it shows |
| --- | --- |
| `app.py` | Post-hoc scoring, side-by-side raw vs guarded output, pre-baked streaming-halt trace, domain-profile inspector |
| `streaming_halt_live.py` | **Interactive live streaming halt** — tokens appear one by one, the coherence gauge tracks the latest score, the halt banner fires on the same frame the kernel stops the stream |

Run either script locally:

```bash
pip install -e ".[demo]"
python demo/app.py
# or
python demo/streaming_halt_live.py
```

Publishing the live demo to a Hugging Face Space waits for an
explicit CEO approval — the Space deploy is a shared-state action
and is not automated from the repo.

## Tabs in `app.py`

- **Score a Response** — enter facts, a query, and an LLM response;
  see the coherence score with PASS/FAIL verdict.
- **Side-by-Side Comparison** — raw LLM output vs the guardrail
  verdict on the same prompt.
- **Streaming Halt** — pre-baked scenario; produces a markdown
  trace showing per-token coherence and halt reason.
- **Domain Presets** — inspect the eight shipped `DirectorConfig`
  profiles.

The **live** streaming halt lives in its own file because Spaces
typically host a single `app.py`. When publishing, choose which
script to expose as the entry.

[GitHub](https://github.com/anulum/director-ai) |
[PyPI](https://pypi.org/project/director-ai/)
