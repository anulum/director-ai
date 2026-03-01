# Reddit Posts — Director-AI v1.2.1

## Why Previous Posts Got Removed (Common Reasons)

1. **Self-promotion ratio**: Reddit enforces ~10% self-promo rule. Your
   account needs organic participation (comments, upvotes on others' work)
   before posting your own project.
2. **Wrong flair/tag**: r/MachineLearning requires `[P]` tag for projects.
   r/LocalLLaMA has no project flair — frame as discussion.
3. **Marketing language**: words like "revolutionary", "game-changing",
   badges, pricing tables → instant removal.
4. **Duplicate posting**: posting the same link to multiple subs within
   hours triggers cross-post spam detection.

## Strategy

- Post to ONE sub first. Wait 24-48h. Then cross-post to the next.
- Comment genuinely on 5-10 other posts in each sub BEFORE posting.
- Frame as "I built X, here's what I learned" not "check out my product".
- Lead with the problem and your honest experience, not features.

---

## Post 1: r/MachineLearning

**Rules**: Must use `[P]` tag. Must be technical. No pure marketing.

### Title

```
[P] I built an open-source guardrail that halts LLM streaming mid-token when coherence drops — here are my honest benchmark numbers
```

### Body

```
I've been working on a hallucination detection system for LLMs and wanted
to share what I've learned, including where it falls short.

The core idea: instead of checking LLM output after it's been sent to the
user, run NLI + fact-checking continuously during streaming and kill the
output if coherence drops. Three independent halt triggers (hard limit,
sliding window, downward trend) so a single good token can't mask a decline.

The scoring is simple:
  Coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)

H_logical comes from DeBERTa-based NLI (contradiction probability).
H_factual comes from retrieval against a user-provided knowledge base.

I evaluated on LLM-AggreFact (29K samples across 11 datasets). Here's
where I have to be honest:

  Bespoke-MiniCheck-7B:  77.4% balanced acc  (7B params)
  Director-AI (FactCG):  75.8%               (0.4B params)
  MiniCheck-Flan-T5-L:   75.0%               (0.8B params)

4th on the AggreFact leaderboard, within 1.6pp of the top 7B model
at 17x fewer params. And none of the models above do streaming halt.

Per-dataset highlights:
- Lfqa (long-form QA): 87.3%. TofuEval-MediaS: 86.2%.
- Summarisation is still a blind spot: AggreFact-CNN 68.8%, ExpertQA 59.1%.
- The heuristic fallback (word overlap when NLI model isn't installed) is
  basically useless in production. I added a strict_mode that returns
  neutral 0.5 instead, so users don't get false confidence.
- Caching NLI calls during streaming matters more than I expected. The same
  prefix gets scored repeatedly as new tokens arrive. LRU cache with blake2b
  keys cut effective latency by ~60%.

The code is AGPL v3 with commercial licensing. 378 tests, Python 3.10+.
There's also a Rust kernel for the streaming interlock (zero unsafe code,
pre-allocated hot path).

Repo: https://github.com/anulum/director-ai

I'd genuinely appreciate feedback on:
1. Is the NLI + RAG combination the right decomposition, or should factual
   checking use a different approach entirely?
2. The 0.6/0.4 weight split was tuned on AggreFact. Has anyone found
   domain-specific tuning to matter significantly for these kinds of systems?
3. Are there better benchmarks than AggreFact for evaluating streaming
   guardrails specifically?
```

---

## Post 2: r/LocalLLaMA

**Rules**: Community-focused. Loves local-first, practical tools. Hates
corporate marketing. Frame as a tool that helps local LLM users.

### Title

```
I made a guardrail that works with Ollama/llama.cpp to catch hallucinations during streaming — open source, runs locally, no API calls needed
```

### Body

```
I run local models (mostly via Ollama) and got tired of outputs that sound
confident but are completely wrong. I couldn't find a guardrail that:
- Works during streaming (not after)
- Runs fully local (no OpenAI API calls to verify)
- Lets me define what "correct" means for my use case

So I built one. It sits between your local model and the output, scores
every token prefix for coherence, and kills the stream if the output goes
off the rails.

The fact-checking runs against your own knowledge base — just a Python dict,
ChromaDB, or any text you feed it. The contradiction detection uses FactCG-DeBERTa-v3-Large
(0.4B params, 75.8% on AggreFact, ~575ms CPU / ~50ms GPU). Everything
local, no external calls.

Quick example with Ollama:

    from director_ai import CoherenceAgent
    agent = CoherenceAgent(llm_api_url="http://localhost:11434/api/generate")
    result = agent.process("What is our refund policy?")

    if result.halted:
        print("Caught hallucination mid-stream")

You can also wrap any OpenAI-compatible endpoint (which Ollama exposes):

    from director_ai import guard
    from openai import OpenAI

    client = guard(
        OpenAI(base_url="http://localhost:11434/v1"),
        facts={"refund": "30-day refund policy, no exceptions."}
    )

I'll be real about limitations:
- 75.8% balanced accuracy on AggreFact puts us 4th on the leaderboard.
  Combined with your own KB facts, it catches significantly more.
- It adds latency. ~575ms per check on CPU (source chunking). ~50-80ms
  on GPU. The caching helps for streaming but it's not zero-cost.
- Summarisation tasks are a weak spot (68.8% on AggreFact-CNN).
  Works much better for factual QA (87.3% on Lfqa).

378 tests, works on Python 3.10+, AGPL v3. No telemetry, no cloud dependency.

https://github.com/anulum/director-ai

Has anyone else tried running hallucination detection locally? Curious what
approaches are working for you.
```

---

## Post 3: r/LangChain

**Rules**: Must be relevant to LangChain ecosystem. Practical, how-to focus.

### Title

```
Open-source hallucination guardrail with LangChain/LangGraph integration — catches bad outputs during streaming, not after
```

### Body

```
I built a hallucination detection guardrail that integrates with LangChain
and LangGraph. The main difference from post-hoc checkers: it monitors
coherence during token streaming and halts output mid-generation if it
starts hallucinating.

LangChain usage (pipe it after any LLM):

    from director_ai.integrations.langchain import DirectorAIGuard

    guard = DirectorAIGuard(
        facts={"refund": "Refunds available within 30 days."},
        threshold=0.6,
        use_nli=True,
    )
    chain = my_llm | guard
    result = chain.invoke({"query": "What is the refund policy?"})

LangGraph usage (add as a node with conditional routing):

    from director_ai.integrations.langgraph import (
        director_ai_node,
        director_ai_conditional_edge,
    )

    node = director_ai_node(
        facts={"policy": "Refunds within 30 days."},
        on_fail="flag",
    )
    edge = director_ai_conditional_edge("output", "retry")

It scores using two signals:
- NLI contradiction detection (DeBERTa, runs locally)
- Fact-checking against whatever knowledge base you provide

The KB can be a simple dict, ChromaDB, or anything implementing the store
interface.

Also supports LlamaIndex, Haystack, and CrewAI if you use those.

pip install director-ai[langchain,nli]

AGPL v3, 378+ tests. 75.8% balanced accuracy on AggreFact (4th on leaderboard).
The real value is combining NLI with your own facts + streaming halt.

https://github.com/anulum/director-ai

Would love to hear if anyone has tried combining this kind of guardrail
with LangGraph retry loops — that's the use case I'm most interested in
optimizing.
```

---

## Post 4: r/artificial (or r/ArtificialIntelligence)

**Rules**: Discussion-oriented. Less technical. Frame around the problem
space, not implementation details.

### Title

```
Most LLM guardrails check output after it's been sent to users. I built one that monitors during streaming and halts mid-generation. Here's what I learned about catching hallucinations in real-time.
```

### Body

```
I spent the last few months building a hallucination detection system for
LLMs and wanted to share some things that surprised me.

The basic problem: every guardrail I could find works post-hoc. The LLM
generates a complete response, then the guardrail checks it. But in
streaming applications, the user is already reading the output while it's
being generated. By the time a post-hoc check runs, the damage is done.

So I built a system that runs two checks on every token prefix during
streaming:
1. Does this output logically contradict itself? (NLI model)
2. Does this output contradict known facts? (retrieval from a knowledge base)

If either check fails badly enough, it kills the stream immediately.

Things I learned:

**Summarisation breaks everything.** NLI models (including mine) are really
bad at catching hallucinations in summarisation tasks. 53% accuracy — barely
better than a coin flip. They're much better at factual QA (80.7%). If
your use case is summarisation, NLI-based guardrails aren't the answer yet.

**Your own facts matter more than the model.** A mediocre NLI model + a
good knowledge base catches way more hallucinations than a great NLI model
alone. The retrieval component does most of the heavy lifting in practice.

**Streaming changes the game.** A single bad token in a 200-token response
is hard to catch post-hoc. But watching the coherence score drop over a
sliding window makes it obvious. The trend matters as much as any individual
score.

**Speed is non-negotiable.** You need sub-second checks for streaming.
That rules out using GPT-4/Claude as a judge (1-3s per call). A 0.4B param
DeBERTa model at 220ms is about the upper bound for usable latency.

The tool is open source: https://github.com/anulum/director-ai

Curious if others have experience with real-time guardrails in production.
What approaches are working for you?
```

---

## Posting Schedule

| Day | Subreddit | Post # |
|-----|-----------|--------|
| Day 1 | r/MachineLearning | Post 1 (most technical, hardest audience) |
| Day 3 | r/LocalLLaMA | Post 2 (practical, local-first angle) |
| Day 5 | r/LangChain | Post 3 (integration-focused) |
| Day 7 | r/artificial | Post 4 (discussion-oriented) |

Between posts: comment genuinely on 3-5 threads per sub. Answer questions.
Help people. Build karma before posting.

## Common Moderator Triggers to Avoid

- "We're excited to announce" → removed
- Pricing tables in the post → removed
- Multiple links in the body → flagged
- Account age < 30 days → automod filter
- Posting same link to 3+ subs in 24h → spam filter
- "Star us on GitHub" / "join our Discord" → removed
- Excessive formatting (headers, bold, badges) → looks like marketing
