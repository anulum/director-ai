# Quickstart

## Choose Your Path

| Method | Command | Time |
|--------|---------|------|
| **pip install** | `pip install director-ai` | 30 seconds |
| **CLI scaffold** | `director-ai quickstart --profile medical` | 1 minute |
| **Colab notebook** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/quickstart.ipynb) | 5 minutes |
| **Docker** | `docker build -t director-ai . && docker run -p 8080:8080 director-ai` | 5 minutes |
| **HF Spaces** | [Try demo](https://huggingface.co/spaces/anulum/director-ai-guardrail) (may be sleeping) | 0 minutes |

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

Creates `director_guard/` with `config.yaml`, `facts.txt`, `guard.py`, and `README.md`.

## Score a Response

```python
from director_ai import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("capital", "Paris is the capital of France.")

scorer = CoherenceScorer(threshold=0.3, ground_truth_store=store)

# Correct answer — approved
approved, cs = scorer.review(
    "What is the capital of France?",
    "The capital of France is Paris.",
)
print(f"Approved: {approved}")        # True
print(f"Score: {cs.score:.3f}")       # ~0.44

# Hallucinated answer — rejected
approved, cs = scorer.review(
    "What is the capital of France?",
    "The capital of France is Berlin.",
)
print(f"Approved: {approved}")        # False
print(f"Score: {cs.score:.3f}")       # ~0.02
```

## Guard an SDK Client

=== "OpenAI"

    ```python
    from director_ai import guard
    from openai import OpenAI

    client = guard(
        OpenAI(),
        facts={"refund": "within 30 days"},
        on_fail="raise",
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the refund policy?"}],
    )
    ```

=== "Anthropic"

    ```python
    from director_ai import guard
    import anthropic

    client = guard(
        anthropic.Anthropic(),
        facts={"refund": "within 30 days"},
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What is the refund policy?"}],
    )
    ```

=== "Bedrock"

    ```python
    from director_ai import guard
    import boto3

    bedrock = boto3.client("bedrock-runtime")
    client = guard(bedrock, facts={"refund": "within 30 days"})

    response = client.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[{"role": "user", "content": [{"text": "Refund policy?"}]}],
    )
    ```

=== "Gemini"

    ```python
    from director_ai import guard
    import google.generativeai as genai

    model = genai.GenerativeModel("gemini-1.5-flash")
    client = guard(model, facts={"refund": "within 30 days"})

    response = client.generate_content("What is the refund policy?")
    ```

### Failure Modes

| Mode | Behavior |
|------|----------|
| `on_fail="raise"` | Raises `HallucinationError` (default) |
| `on_fail="log"` | Logs warning, returns response unchanged |
| `on_fail="metadata"` | Stores score in context var for later inspection |

## Streaming Halt

```python
from director_ai import StreamingKernel

kernel = StreamingKernel(hard_limit=0.4, window_size=8)

def score_fn(accumulated_text):
    return 0.85  # your coherence scoring logic on text so far

session = kernel.stream_tokens(token_generator, score_fn)
if session.halted:
    print(f"Halted at token {session.halt_index}: {session.halt_reason}")
```

## Fallback Modes

```python
from director_ai import CoherenceAgent

# Retrieval: return KB context when all candidates fail
agent = CoherenceAgent(fallback="retrieval")

# Disclaimer: prepend warning to best-rejected candidate
agent = CoherenceAgent(fallback="disclaimer")
```

## Batch Scoring

```python
from director_ai import CoherenceScorer

scorer = CoherenceScorer(threshold=0.6, use_nli=True)

items = [
    ("What is 2+2?", "The answer is 4."),
    ("Capital of France?", "Paris is in Germany."),
]
results = scorer.review_batch(items)
for approved, score in results:
    print(f"approved={approved}  score={score.score:.3f}")
```

`review_batch()` batches NLI pairs into 2 GPU forward passes when NLI is available. Dialogue items fall back to sequential scoring.

## Async Usage

```python
import asyncio
from director_ai import CoherenceAgent

agent = CoherenceAgent(use_nli=True)

async def main():
    result = await agent.aprocess("What is the capital of France?")
    print(result)

asyncio.run(main())
```

## Next Steps

- [Scoring guide](guide/scoring.md) — thresholds, weights, NLI backends
- [Streaming halt](guide/streaming.md) — halt mechanisms, `on_halt` callbacks
- [KB ingestion](guide/kb-ingestion.md) — populate your knowledge base
- [Integrations](integrations/sdk-guard.md) — OpenAI, Anthropic, LangChain, and more
- [Production deployment](deployment/production.md) — scaling, caching, monitoring
- [Domain presets](guide/presets.md) — medical, finance, legal, creative profiles
- [Tutorials](tutorials.md) — 16 Jupyter notebooks from basics to production
