# Quickstart

## Choose Your Path

| Method | Command | Time |
|--------|---------|------|
| **pip install** | `pip install director-ai` | 30 seconds |
| **CLI scaffold** | `director-ai quickstart --profile medical` | 1 minute |
| **Colab notebook** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/quickstart.ipynb) | 5 minutes |
| **Docker** | `docker run -p 8080:8080 ghcr.io/anulum/director-ai:latest` | 2 minutes |
| **HF Spaces** | [Try it live](https://huggingface.co/spaces/anulum/director-ai-guardrail) | 0 minutes |

## Installation

```bash
pip install director-ai
```

For NLI-enhanced scoring (recommended for production):

```bash
pip install director-ai[nli]         # FactCG-DeBERTa-v3-Large (75.8% balanced acc)
pip install director-ai[minicheck]   # MiniCheck alternative (72.6% balanced acc)
```

## CLI Quickstart

Scaffold a working project in one command:

```bash
director-ai quickstart --profile medical
cd director_guard
python guard.py
```

This creates `director_guard/` with `config.yaml`, `facts.txt`, `guard.py`, and `README.md`.

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
- [Domain presets](guide/config.md) — medical, finance, legal, creative profiles
