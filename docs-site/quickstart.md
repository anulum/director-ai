# Quickstart

## Installation

```bash
pip install director-ai
```

For NLI-enhanced scoring (recommended for production):

```bash
pip install director-ai[nli]         # DeBERTa NLI model
pip install director-ai[minicheck]   # MiniCheck (72.6% balanced acc)
```

## Score a Response

```python
from director_ai import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("capital", "Paris is the capital of France.")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
approved, score = scorer.review(
    "What is the capital of France?",
    "The capital of France is Berlin.",
)

print(f"Approved: {approved}")        # False
print(f"Score: {score.score:.3f}")    # ~0.35
print(f"Evidence: {score.evidence}")  # Retrieved context + NLI details
```

## Guard an SDK Client

```python
from director_ai import guard
from openai import OpenAI

client = guard(
    OpenAI(),
    facts={"refund": "within 30 days"},
    on_fail="raise",  # or "log" or "metadata"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the refund policy?"}],
)
```

## Streaming Halt

```python
from director_ai import StreamingKernel

kernel = StreamingKernel(hard_limit=0.4, window_size=8)

def score_fn(token):
    # Your coherence scoring logic
    return 0.85

session = kernel.stream_tokens(token_generator, score_fn)
if session.halted:
    print(f"Halted at token {session.halt_index}: {session.halt_reason}")
```

## Fallback Modes

```python
from director_ai import CoherenceAgent

# Retrieval fallback: return KB context when all candidates fail
agent = CoherenceAgent(fallback="retrieval")

# Disclaimer fallback: prepend warning to best-rejected candidate
agent = CoherenceAgent(fallback="disclaimer")
```

## Next Steps

- [Scoring guide](guide/scoring.md) — thresholds, weights, soft zones
- [Streaming halt](guide/streaming.md) — halt mechanisms, on_halt callbacks
- [Integrations](integrations/sdk-guard.md) — OpenAI, Anthropic, LangChain, etc.
- [Production deployment](deployment/production.md) — scaling, caching, metrics
