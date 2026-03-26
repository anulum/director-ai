# Show HN Post — Director-AI v3.10.1

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
I built Director-AI because every guardrail I found reviews output after it's already been sent to the user. That's too late. Director-AI monitors coherence token-by-token during streaming and halts the output the moment it starts hallucinating.

How it works:

1. Two independent scorers run on the accumulated text:
   - NLI model (FactCG-DeBERTa-v3-Large, 0.4B params) catches logical contradictions
   - RAG scorer checks against your own knowledge base (ChromaDB, FAISS, Qdrant, Pinecone, or in-memory)

2. Composite score: coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)

3. Three halt mechanisms fire during streaming:
   - Hard limit: score drops below threshold -> immediate stop
   - Sliding window: rolling average below threshold -> stop
   - Downward trend: sustained coherence decline -> stop

Works with any LLM. One-liner SDK interceptor:

    from director_ai import guard
    client = guard(OpenAI(), facts={"policy": "Refunds within 30 days."})
    # Every completion is now scored. Hallucinations raise or get logged.

Also integrates with LangChain, LlamaIndex, LangGraph, Haystack, and CrewAI.

Honest benchmarks (LLM-AggreFact, 29K samples):

    Director-AI (FactCG):    75.86% balanced accuracy (0.4B params, $0/call)
    Claude Haiku 4.5:        75.10% balanced accuracy (~20B params, $0.37/1K)
    Claude Sonnet 4.6:       74.25% balanced accuracy (~200B params, $1.40/1K)
    GPT-4o:                  73.46% balanced accuracy (~200B params, $1.16/1K)

A 0.4B model beating all frontier LLMs on the same benchmark at $0/call. Sub-millisecond latency on GPU (0.5 ms/pair on L40S FP16). 14.6 ms/pair on a GTX 1060 with ONNX.

What's in v3.10.1:
- Sentence-level claim verification (VerifiedScorer) with 5 independent signals
- Meta-confidence scoring (how confident is the verdict itself?)
- Cross-turn contradiction tracking in conversations
- JSON, tool call, and code AST verification (stdlib only, zero dependencies)
- Online calibration from production feedback with Wilson confidence intervals
- EU AI Act Article 15 compliance reporting (audit log, drift detection, feedback loop detection)
- Verification gems: numeric arithmetic checking, reasoning chain validation, temporal freshness

3200+ tests, CI on Python 3.11-3.13, 91% coverage, mypy, ruff, bandit. AGPL v3 (commercial license available).

Live demo: https://huggingface.co/spaces/anulum/director-ai-guardrail
Docs: https://anulum.github.io/director-ai
PyPI: pip install director-ai

I'd love feedback on the scoring approach and whether this catches hallucinations you see in practice.
```

---

## Preemptive FAQ (for comments)

**Q: How accurate is it?**
A: 75.86% balanced accuracy on AggreFact (29,320 samples). Beats Claude Haiku 4.5, Claude Sonnet 4.6, GPT-4o, and GPT-4o-mini on the same test set. With your own KB loaded, accuracy improves further (89.1% on Reveal, 86.4% on Lfqa subsets). The NLI model is pluggable.

**Q: How does this compare to NeMo Guardrails?**
A: NeMo focuses on topic/intent rails (block certain categories). Director-AI focuses on factual accuracy — does the output contradict known facts or its own logic? Different tools, stackable. NeMo also doesn't do streaming halt.

**Q: What's the latency overhead?**
A: 0.5 ms/pair on L40S (FP16 batch=32). 14.6 ms/pair on GTX 1060 (ONNX GPU batch). 383 ms/pair on CPU. Heuristic scoring (no GPU): <0.1 ms. Score cache eliminates redundant NLI calls during streaming.

**Q: Why not just use Claude/GPT-4 as a judge?**
A: We benchmarked that. GPT-4o as judge: 73.46% BA at $1.16/1K and ~2.3 seconds per check. Director-AI: 75.86% BA at $0/call and 0.5 ms. For streaming where you need sub-second decisions, an LLM judge doesn't work.

**Q: AGPL — is this really open source?**
A: Full source, self-host, modify freely. AGPL copyleft means if you run it as a networked service, you share modifications. Commercial license removes that. Standard dual-license model (MongoDB, Grafana playbook). Founding member pricing: CHF 29-99/mo.

**Q: Voice AI?**
A: Yes — VoiceGuard class for real-time TTS pipelines. You can't unsay spoken words, so streaming halt is critical for voice. Works with ElevenLabs, OpenAI TTS, any streaming TTS provider.
