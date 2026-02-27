# Director-AI — 60-Second Demo Video Script

**Target**: Twitter/X, LinkedIn, HN launch post embed
**Format**: 1920x1080, screen recording + overlay text, no voiceover (caption-only)

---

## Shot 1 — The Problem (0:00–0:08)

**Screen**: Split view — left shows ChatGPT confidently stating a wrong refund policy, right shows a user trusting it.

**Overlay text**:
> LLMs hallucinate. Your users trust them anyway.

---

## Shot 2 — One-Line Install (0:08–0:14)

**Screen**: Terminal, dark theme.

```
$ pip install director-ai
```

**Overlay text**:
> ~1 MB. No GPU required.

---

## Shot 3 — Score a Response (0:14–0:28)

**Screen**: HF Spaces demo (Tab 1 — "Score a Response").

**Actions**:
1. Type fact: "Refunds available within 30 days"
2. Type response: "We offer full refunds within 90 days" (hallucination)
3. Click "Score"
4. Red FAIL badge appears, coherence score ~0.28

**Overlay text**:
> Dual-entropy scoring: NLI + RAG against YOUR knowledge base.

---

## Shot 4 — Streaming Halt (0:28–0:42)

**Screen**: HF Spaces demo (Tab 2 — "Streaming Halt").

**Actions**:
1. Click "Run Streaming Demo"
2. Tokens flow in real-time
3. Mid-stream: red "[HALT]" marker appears, generation stops
4. Camera zooms on the halt point

**Overlay text**:
> Token-by-token oversight. Halts mid-stream — not post-hoc.

---

## Shot 5 — SDK Guard (0:42–0:50)

**Screen**: VS Code, dark theme, `sdk_guard_demo.py` open.

**Highlight** the 2-line usage:
```python
from director_ai import guard
client = guard(OpenAI(), facts={"refund": "within 30 days"})
```

**Overlay text**:
> 2-line SDK wrapper. Works with OpenAI, Anthropic, Ollama.

---

## Shot 6 — Benchmarks (0:50–0:55)

**Screen**: `benchmarks/results/end_to_end_results.md` rendered in GitHub.

**Highlight**: QA precision 81.8%, p95 latency 40ms.

**Overlay text**:
> Honest benchmarks. 300-trace HaluEval eval. Full comparison in repo.

---

## Shot 7 — CTA (0:55–1:00)

**Screen**: GitHub repo landing page.

**Overlay text**:
> pip install director-ai
> github.com/anulum/director-ai
> AGPL-3.0 | Commercial licensing available

---

## Recording Tips

- Use OBS or ScreenFlow at 60fps
- Terminal font: JetBrains Mono or Fira Code, 16pt minimum
- Overlay font: Inter or SF Pro, white on semi-transparent dark pill
- Add subtle typing sound effects for terminal shots
- Background music: lo-fi ambient, -20dB (optional)
- Export: H.264, 1080p, <15 MB for Twitter upload

## Social Media Captions

**Twitter/X**:
> Director-AI — real-time LLM hallucination guardrail.
>
> Token-level streaming halt. NLI + RAG scoring. 2-line SDK wrapper.
>
> pip install director-ai
> github.com/anulum/director-ai

**LinkedIn**:
> Shipped Director-AI: an open-source guardrail that catches LLM
> hallucinations in real-time and halts generation mid-stream.
>
> Not post-hoc review — token-by-token coherence monitoring against
> your own knowledge base.
>
> Try the live demo: huggingface.co/spaces/anulum/director-ai-guardrail
