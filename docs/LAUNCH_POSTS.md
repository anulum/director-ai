# Director-AI v1.0.0 Launch Posts

Ready to copy-paste. Updated Feb 26 2026.

---

## Twitter/X Thread

**Tweet 1/6:**
Shipped Director-AI v1.0.0 — open-source LLM hallucination guardrail with token-level streaming halt.

Your LLM lies mid-sentence? Director-AI kills the stream before the user sees it.

No other guardrail does this. github.com/anulum/director-ai

**Tweet 2/6:**
How it works:

- NLI model (DeBERTa) catches logical contradictions
- RAG scorer checks against YOUR knowledge base
- Both must pass. Score drops below threshold → output severed mid-token

```python
from director_ai import CoherenceAgent
agent = CoherenceAgent()
result = agent.process("What is our refund policy?")
# result.halted = True if the LLM hallucinated
```

**Tweet 3/6:**
Honest benchmarks (LLM-AggreFact, 29K samples):

Director-AI: 66.2% balanced acc
MiniCheck-DeBERTa: 72.6%
Bespoke-MiniCheck-7B: 77.4%

NLI alone isn't SOTA. The value is the system — streaming halt + custom KB + configurable thresholds. The NLI model is pluggable.

**Tweet 4/6:**
Enterprise-ready out of the box:

- YAML policy engine
- JSONL audit trail
- Multi-tenant KB isolation
- Prompt injection hardening
- LangChain Runnable + LlamaIndex postprocessor
- FastAPI REST server + CLI

**Tweet 5/6:**
Install:
```
pip install director-ai          # basic
pip install director-ai[nli]     # DeBERTa scoring
pip install director-ai[vector]  # ChromaDB knowledge base
pip install director-ai[server]  # REST API
```

Dual-licensed: AGPL v3 (open-source) + commercial from anulum.li

**Tweet 6/6:**
If you're building RAG/chat apps and tired of post-hoc hallucination review — this is the alternative.

Real-time. Token-level. Your data, your rules.

Star it, try it, break it: github.com/anulum/director-ai

---

## Hacker News

**Title:** Director-AI: Open-source LLM hallucination guardrail with token-level streaming halt

**Body:**
I built Director-AI because every hallucination detector I found works post-hoc — score the full output, then decide. That's fine for batch evaluation. It's useless when you're streaming tokens to a user and the LLM starts hallucinating mid-sentence.

Director-AI sits between the LLM and the user. It scores coherence token-by-token using dual-entropy (NLI contradiction detection + RAG fact-checking against your own knowledge base). When coherence drops below threshold, it halts the stream immediately.

Honest numbers: 66.2% balanced accuracy on LLM-AggreFact (29K samples). Not state-of-the-art on the NLI component alone — MiniCheck-7B hits 77.4%. But none of them offer streaming halt, and the NLI model is pluggable. The system value is the real-time gating + custom KB + configurable policy.

Enterprise modules included: YAML policy engine, audit logger, multi-tenant KB isolation, prompt injection hardening, LangChain/LlamaIndex integrations.

AGPL v3 + commercial license. Python 3.10+, Rust kernel under the hood.

github.com/anulum/director-ai

---

## Reddit (r/LangChain or r/LocalLLaMA)

**Title:** Director-AI v1.0.0 — real-time hallucination guardrail with streaming halt (open-source)

**Body:**
Released Director-AI, a hallucination guardrail that works at the token level during streaming — not after the full response is generated.

**What makes it different from NeMo Guardrails / Guardrails AI / SelfCheckGPT:**
- Token-level streaming halt — kills output mid-stream when coherence drops
- Dual scoring: NLI (DeBERTa) + RAG against your own knowledge base
- Both must pass before output reaches the user

**LangChain integration is built-in:**
```python
from director_ai.integrations.langchain import DirectorAIGuard
guard = DirectorAIGuard(threshold=0.6)
chain = my_llm | guard  # drops hallucinated outputs
```

LlamaIndex postprocessor also included.

Enterprise stuff: policy engine, audit trail, multi-tenant KB, prompt injection hardening, REST API, CLI.

Benchmarks are honest — 66.2% on AggreFact. Not the highest NLI score, but no competitor offers streaming halt. The NLI model is swappable.

`pip install director-ai[nli,vector]`

AGPL v3 + commercial license. Repo: github.com/anulum/director-ai

---

## LangChain Discord

Shipped a LangChain-native hallucination guardrail: `director-ai`

```python
from director_ai.integrations.langchain import DirectorAIGuard
guard = DirectorAIGuard(threshold=0.6, use_nli=True)
chain = my_retriever | my_llm | guard
```

It scores every output with NLI + RAG fact-checking and can halt token streams mid-generation. Enterprise modules (policy, audit, multi-tenant) included.

`pip install director-ai[langchain,nli]`

github.com/anulum/director-ai

---

## LlamaIndex Discord

Built a LlamaIndex-native hallucination filter: `director-ai`

```python
from director_ai.integrations.llamaindex import DirectorAIPostprocessor

postprocessor = DirectorAIPostprocessor(threshold=0.6, use_nli=True)
query_engine = index.as_query_engine(node_postprocessors=[postprocessor])
```

Filters out hallucinated nodes before they reach the user. Adds `director_ai_score` metadata to surviving nodes. Also does token-level streaming halt for real-time apps.

`pip install director-ai[llamaindex,nli]`

github.com/anulum/director-ai
