# Show HN Post — Director-AI v1.2.0

## Title

```
Show HN: Director-AI – Open-source hallucination guardrail that halts LLM output mid-stream
```

## URL

```
https://github.com/anulum/director-ai
```

## Text

```
I built Director-AI because every guardrail I found reviews output after it's already been sent to the user. That's too late. Director-AI monitors coherence token-by-token during streaming and severs the output the moment it starts hallucinating.

How it works:

1. Two independent scorers run on every token prefix:
   - NLI model (DeBERTa / MiniCheck) catches logical contradictions
   - RAG scorer checks against your own knowledge base (ChromaDB or in-memory)

2. Composite score: Coherence = 1 - (0.6 × H_logical + 0.4 × H_factual)

3. Three halt mechanisms fire during streaming:
   - Hard limit: score drops below threshold → immediate kill
   - Sliding window: average of last N scores below threshold → kill
   - Downward trend: sustained coherence decline → kill

Works with any LLM — OpenAI, Anthropic, Ollama, vLLM, llama.cpp. One-liner SDK interceptor:

    from director_ai.integrations import guard
    client = guard(OpenAI(), facts={"policy": "Refunds within 30 days."})
    # Every completion is now scored. Hallucinations raise or get logged.

Also integrates with LangChain, LlamaIndex, LangGraph, Haystack, and CrewAI.

Honest benchmarks (LLM-AggreFact, 29K samples):

    Bespoke-MiniCheck-7B: 77.4% balanced accuracy  (7B params)
    Director-AI (FactCG): 75.8% balanced accuracy   (0.4B params)
    MiniCheck-Flan-T5-L:  75.0% balanced accuracy   (0.8B params)

4th on the leaderboard, within 1.6pp of the top 7B model at 17x fewer params. And none of the above do streaming halt — that's Director-AI only.

What's new in v1.2.0:
- LRU score cache with blake2b hashing (eliminates redundant NLI calls during streaming)
- 8-bit quantized NLI via bitsandbytes (<80ms on GPU)
- LangGraph nodes, Haystack components, CrewAI tools
- Full docs site: https://anulum.github.io/director-ai

374 tests, CI on Python 3.10–3.12, mypy, ruff. AGPL v3 (commercial license available).

Live demo: https://huggingface.co/spaces/anulum/director-ai-guardrail

I'd love feedback on the scoring formula, the halt mechanisms, and whether the NLI + RAG combination actually catches the hallucinations you see in practice.
```

---

## Preemptive FAQ (for comments)

**Q: How accurate is it?**
A: 75.8% balanced accuracy on AggreFact (29K samples), ranking 4th on the leaderboard. Combined with your own KB facts, the system accuracy goes higher. The NLI model is also pluggable — swap in MiniCheck or any future model.

**Q: How does this compare to NeMo Guardrails?**
A: NeMo focuses on topic/intent rails (block certain categories of questions). Director-AI focuses specifically on factual accuracy — does the output contradict known facts or its own logic? Different tools for different problems. They can be stacked.

**Q: What's the latency overhead?**
A: ~575ms per NLI call on CPU (FactCG with source chunking). ~50-80ms on GPU. The score cache eliminates redundant calls during streaming, so the effective overhead per token is much lower.

**Q: Why not just use Claude/GPT-4 as a judge?**
A: Latency and cost. An LLM judge adds 1-3s per check. Director-AI runs a 0.4B parameter model locally at ~575ms CPU / ~50ms GPU. For token-level streaming where you need sub-second decisions, an LLM judge doesn't work.

**Q: AGPL — is this really open source?**
A: Yes, full source, self-host, modify freely. The AGPL copyleft means if you run it as a service, you share your modifications. Commercial license removes that requirement. Standard dual-license model (similar to MongoDB, Grafana).
