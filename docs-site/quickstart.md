<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-AI — Quickstart
-->

# Quickstart

Use this page when you want to run Director-AI, score a response, and protect
one application path quickly. If you are still deciding whether the product fits
your use case, read [Product Overview](guide/product-overview.md) first. If you
are planning a pilot, use [Evaluation Onboarding](guide/onboarding.md) as the
checklist.

## Choose Your First Path

| Goal | Start with | Then read |
|---|---|---|
| Understand the product | [Applications and Market Map](guide/applications-and-market-map.md) | [Product Overview](guide/product-overview.md) |
| Score one answer | `score()` example below | [Scoring](guide/scoring.md) |
| Wrap an existing app | `guard()` example below | [SDK Guard](integrations/sdk-guard.md) |
| Protect a RAG bot | `director-ai[vector]` | [KB Ingestion](guide/kb-ingestion.md) |
| Evaluate a pilot | labelled examples | [Evaluation Onboarding](guide/onboarding.md) |
| Deploy a service | `director-ai quickstart --run` | [Production Guide](deployment/production.md) |
| Prepare a production stack | `director-ai quickstart --profile production` | [Monitoring](deployment/monitoring.md) |

## Recommended Path

Start with the Python service and local Chroma path:

```bash
pip install director-ai[server,vector]
director-ai quickstart --run
director-ai doctor
```

This starts the default proxy on port 8080, the FastAPI service on port 8000,
and local Chroma persistence under `./director_guard/chroma`.

## What You Should See

After the first run you should have:

- a guarded call path that can score a prompt/response pair;
- at least one governed fact loaded inline or through local Chroma;
- a rejection or low score for a deliberately wrong answer;
- a path to inspect the score, evidence, and failure action;
- no secrets printed in logs or notebook output.

If you are evaluating Director-AI for a team, save those five observations in
the pilot evidence packet from [Evaluation Onboarding](guide/onboarding.md).

## Other Entry Points

| Method | Command | Use When |
|--------|---------|----------|
| **CLI scaffold** | `director-ai quickstart --profile medical` | You want editable local files before running services |
| **Base package** | `pip install director-ai` | You only need the in-process Python guard API |
| **Colab notebook** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/quickstart.ipynb) | You want a notebook walkthrough |
| **Docker image** | `docker build -t director-ai . && docker run -p 8080:8080 director-ai` | You are validating the packaged container |
| **HF Spaces** | [Try demo](https://huggingface.co/spaces/anulum/director-ai-guardrail) (may be sleeping) | You only want to inspect the demo |

## Installation

For the smallest in-process install:

```bash
pip install director-ai                # rules engine + heuristic (zero ML, <1ms)
pip install director-ai[embed]         # + embedding scorer (~65% BA, 3ms CPU)
pip install director-ai[nli]           # + FactCG NLI (75.6% BA, 14.6ms GPU) — recommended
pip install director-ai[nli,server]    # + REST API server for production
```

For backend choices beyond the default, use the
[advanced backend matrix](installation.md#advanced-backend-matrix).

## CLI Quickstart

Scaffold a working project in one command:

```bash
director-ai quickstart --profile medical
cd director_guard
python guard.py
```

Creates `director_guard/` with `config.yaml`, `facts.txt`, `guard.py`,
`README.md`, Docker Compose, local Chroma persistence, and an opt-in FactCG
ONNX profile.

Run the default proxy and FastAPI services:

```bash
director-ai quickstart --run
```

For an authenticated production scaffold:

```bash
director-ai quickstart --profile production
cd director_guard
```

Fill `.env` with `DIRECTOR_API_KEY_TENANT_MAP`, `DIRECTOR_PROXY_API_KEYS`,
`DIRECTOR_LLM_API_URL`, `DIRECTOR_UPSTREAM_URL`, `DIRECTOR_KB_HMAC_KEYS`, and
`DIRECTOR_CORS_ORIGINS`, then start the service:

```bash
docker compose up
```

The production scaffold enables NLI, model-backed fail-closed checks, tenant
routing, signed knowledge writes, audit/compliance/feedback stores, JSON logs,
authenticated metrics, rate limiting, and the review queue. To add Prometheus,
write the matching API key to `secrets/director-api-key` and run:

```bash
docker compose --profile monitoring up
```

Run the ONNX service after placing exported model files in
`director_guard/models/factcg-onnx/`:

```bash
cd director_guard
docker compose --profile onnx up director-proxy-onnx
```

See [ONNX Artefacts](deployment/onnx-artefacts.md) for export commands and
CPU/GPU wheel targets.

Rust, Go, Julia, Lean, TensorRT, and WASM paths are optional advanced
runtimes. Start with the Python-only quickstart unless one of those runtimes is
explicitly needed. See [Runtime Boundaries](guide/runtime-boundaries.md).

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

=== "Mistral"

    ```python
    import os

    from director_ai import guard
    from mistralai import Mistral

    client = guard(
        Mistral(api_key=os.environ["MISTRAL_API_KEY"]),
        facts={"refund": "within 30 days"},
    )

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "What is the refund policy?"}],
    )
    ```

=== "Pydantic AI"

    ```python
    from director_ai import guard
    from pydantic_ai import Agent

    agent = guard(
        Agent("openai:gpt-4o-mini"),
        facts={"refund": "within 30 days"},
    )

    result = agent.run_sync("What is the refund policy?")
    print(result.output)
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
- [Tutorials](tutorials.md) — 17 Jupyter notebooks from basics to production
- [Notebook Gallery](notebook-gallery.md) — use-case index across every published notebook
