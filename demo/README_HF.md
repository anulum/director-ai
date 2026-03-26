---
title: Director-AI Guardrail
emoji: "\U0001F6E1"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.7.0"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Real-time LLM hallucination guardrail
---

# Director-AI Demo

Real-time LLM hallucination guardrail with NLI + RAG fact-checking and
token-level streaming halt.

**Tab 1 — Score a Response:** Enter facts, a query, and an LLM response.
See the coherence score with PASS/FAIL verdict and evidence.

**Tab 2 — Side-by-Side Comparison:** Raw LLM vs Director-AI guarded output
with token-level colour coding.

**Tab 3 — Streaming Halt:** Watch three halt mechanisms (hard limit,
sliding window, downward trend) stop hallucination mid-stream.

**Tab 4 — Domain Presets:** Explore 8 built-in profiles (medical, finance,
legal, creative, etc.) with tuned thresholds.

[GitHub](https://github.com/anulum/director-ai) |
[PyPI](https://pypi.org/project/director-ai/) |
[Docs](https://anulum.github.io/director-ai/)
