<p align="center">
  <img src="docs/assets/header.png" width="1280" alt="Director-AI — Real-time LLM Hallucination Guardrail">
</p>

<h1 align="center">Director-AI</h1>

<p align="center">
  <strong>Real-time LLM hallucination guardrail — NLI + RAG fact-checking with token-level streaming halt</strong>
</p>

<p align="center">
  <a href="https://github.com/anulum/director-ai/actions/workflows/ci.yml"><img src="https://github.com/anulum/director-ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/tests-2051_passed-brightgreen.svg" alt="Tests">
  <a href="https://pypi.org/project/director-ai/"><img src="https://img.shields.io/pypi/v/director-ai.svg" alt="PyPI"></a>
  <a href="https://codecov.io/gh/anulum/director-ai"><img src="https://codecov.io/gh/anulum/director-ai/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://hub.docker.com/r/anulum/director-ai"><img src="https://img.shields.io/badge/docker-ready-blue.svg" alt="Docker"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
  <a href="https://huggingface.co/spaces/anulum/director-ai-guardrail"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-orange.svg" alt="HF Spaces"></a>
  <a href="https://doi.org/10.5281/zenodo.18822167"><img src="https://zenodo.org/badge/doi/10.5281/zenodo.18822167.svg" alt="DOI"></a>
  <a href="https://anulum.github.io/director-ai"><img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Docs"></a>
  <a href="https://www.bestpractices.dev/projects/12102"><img src="https://www.bestpractices.dev/projects/12102/badge" alt="OpenSSF Best Practices"></a>
  <a href="https://securityscorecards.dev/viewer/?uri=github.com/anulum/director-ai"><img src="https://api.securityscorecards.dev/projects/github.com/anulum/director-ai/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://api.reuse.software/info/github.com/anulum/director-ai"><img src="https://api.reuse.software/badge/github.com/anulum/director-ai" alt="REUSE"></a>
</p>

---

## What It Does

Director-AI sits between your LLM and the user. It scores every output for
hallucination before it reaches anyone — and can halt generation mid-stream if
coherence drops below threshold.

```mermaid
graph LR
    LLM["LLM<br/>(any provider)"] --> D["Director-AI"]
    D --> S["Scorer<br/>NLI + RAG"]
    D --> K["StreamingKernel<br/>token-level halt"]
    S --> V{Approved?}
    K --> V
    V -->|Yes| U["User"]
    V -->|No| H["HALT + evidence"]
```

**Four things make it different:**

1. **Token-level streaming halt** — not post-hoc review. Severs output the moment coherence degrades.
2. **Dual-entropy scoring** — NLI contradiction detection (DeBERTa) + RAG fact-checking against your knowledge base.
3. **Continuous batching** — server-level request queue coalesces NLI calls (2 GPU kernels per flush, not 2*N per request).
4. **Your data, your rules** — ingest your own documents. The scorer checks against *your* ground truth.

### Scope

100% Python — no compiled extensions required. Works on any platform with Python 3.11+.

| Layer | Packages | Install |
|-------|----------|---------|
| **Core** (zero heavy deps) | `CoherenceScorer`, `StreamingKernel`, `GroundTruthStore`, `SafetyKernel` | `pip install director-ai` |
| **NLI models** | DeBERTa, FactCG, MiniCheck, ONNX Runtime | `pip install director-ai[nli]` |
| **Vector DBs** | ChromaDB, Pinecone, Weaviate, Qdrant | `pip install director-ai[vector]` |
| **LLM judge** | OpenAI, Anthropic escalation | `pip install director-ai[openai]` |
| **Observability** | OpenTelemetry spans | `pip install director-ai[otel]` |
| **Server** | FastAPI + Uvicorn | `pip install director-ai[server]` |

## Quickstart

| Method | Command |
|--------|---------|
| **pip install** | `pip install director-ai` |
| **CLI scaffold** | `director-ai quickstart --profile medical` |
| **Colab** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/quickstart.ipynb) |
| **HF Spaces** | [Try it live](https://huggingface.co/spaces/anulum/director-ai-guardrail) |
| **Docker** | `docker run -p 8080:8080 ghcr.io/anulum/director-ai:latest` |

### 6-line guard

```python
from director_ai import guard
from openai import OpenAI

client = guard(
    OpenAI(),
    facts={"refund_policy": "Refunds within 30 days only"},
    threshold=0.6,
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the refund policy?"}],
)
```

### Catch and inspect a halt

```python
from director_ai import guard, HallucinationError
from openai import OpenAI

client = guard(OpenAI(), facts={"policy": "Refunds within 30 days only"})

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the refund policy?"}],
    )
except HallucinationError as exc:
    print(f"HALTED: coherence={exc.score.score:.3f}")
    print(f"Evidence: {exc.score.evidence}")
```

### Score a response

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("sky color", "The sky is blue due to Rayleigh scattering.")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
approved, score = scorer.review("What color is the sky?", "The sky is green.")

print(approved)     # False
print(score.score)  # 0.42
```

### Streaming halt

```python
from director_ai.core import CoherenceScorer
from director_ai.core.streaming import StreamingKernel

scorer = CoherenceScorer(threshold=0.5)
kernel = StreamingKernel(hard_limit=0.4, window_size=5)

accumulated = []
def coherence_cb(token):
    accumulated.append(token)
    return scorer.review("my prompt", " ".join(accumulated))[1].score

session = kernel.stream_tokens(token_generator, coherence_cb)

if session.halted:
    print(f"Halted at token {session.halt_index}: {session.halt_reason}")
```

## Installation

```bash
pip install director-ai                      # heuristic scoring
pip install director-ai[nli]                 # NLI model (DeBERTa)
pip install director-ai[vector]              # ChromaDB knowledge base
pip install "director-ai[nli,vector,server]" # production stack
```

Framework integrations: `[langchain]`, `[llamaindex]`, `[langgraph]`, `[haystack]`, `[crewai]`.

Full installation guide: [docs](https://anulum.github.io/director-ai/installation/).

## Docker

```bash
docker run -p 8080:8080 ghcr.io/anulum/director-ai:latest        # CPU
docker run --gpus all -p 8080:8080 ghcr.io/anulum/director-ai:gpu # GPU
```

## Benchmarks

### Accuracy — LLM-AggreFact (29,320 samples)

| Model | Balanced Acc | Params | Latency | Streaming |
|-------|-------------|--------|---------|-----------|
| Bespoke-MiniCheck-7B | **77.4%** | 7B | ~100 ms | No |
| **Director-AI (FactCG)** | **75.8%** | 0.4B | **14.6 ms** | **Yes** |
| MiniCheck-Flan-T5-L | 75.0% | 0.8B | ~120 ms | No |
| MiniCheck-DeBERTa-L | 72.6% | 0.4B | ~120 ms | No |

75.8% balanced accuracy at 17x fewer params than the leader. 14.6 ms/pair with
ONNX GPU batching — faster than every competitor at this accuracy tier.
Director-AI's unique value is the *system*: NLI + KB + streaming halt.

Full results: [`benchmarks/comparison/COMPETITOR_COMPARISON.md`](benchmarks/comparison/COMPETITOR_COMPARISON.md).

### Performance Trade-offs

| Backend | Latency (GPU) | Latency (CPU) | Accuracy | Streaming | When to use |
|---------|--------------|---------------|----------|-----------|-------------|
| Heuristic (no NLI) | <0.1 ms | <0.1 ms | ~55% | Yes | Prototyping, latency-critical |
| ONNX GPU batch | **14.6 ms/pair** | 383 ms/pair | **75.8%** | Yes | Production GPU |
| PyTorch GPU batch | 19.0 ms/pair | N/A | 75.8% | Yes | When ONNX unavailable |
| PyTorch GPU seq | 197 ms/pair | N/A | 75.8% | Yes | Single-pair scoring |
| Hybrid (NLI + LLM judge) | 200-500 ms | 500-2000 ms | ~78% est. | Yes | Max accuracy, summarisation |

Streaming cadence multiplies per-token overhead. At `score_every_n=4`, divide callback
cost by 4. See [`docs-site/guide/streaming-overhead.md`](docs-site/guide/streaming-overhead.md).

### End-to-End Pipeline (300 traces)

Full pipeline (CoherenceAgent + GroundTruthStore + StreamingKernel):

| Metric | Value |
|--------|-------|
| Catch rate (recall) | 46.7% |
| Precision | 56.9% |
| F1 | 51.3% |
| Evidence coverage | 100% (every rejection includes supporting chunks) |
| Avg latency | 15.8 ms (p95: 40 ms) |

Dialogue catch rate is 80%; QA and summarisation are lower (36%, 24%) due
to NLI weakness on short-form text. Hybrid mode improves summarisation.
Summarisation FPR reduced from 95% to 10.5% in v3.5.0 via bidirectional NLI + baseline calibration.
Run: `python -m benchmarks.e2e_eval`.

## Domain Presets

8 built-in profiles with tuned thresholds:

```bash
director-ai config --profile medical   # threshold=0.75, NLI on, reranker on
director-ai config --profile finance   # threshold=0.70, w_fact=0.6
director-ai config --profile legal     # threshold=0.68, w_logic=0.6
director-ai config --profile creative  # threshold=0.40, permissive
```

## Known Limitations

1. **Heuristic fallback is weak**: Without `[nli]`, scoring uses word-overlap heuristics (~55% accuracy). Use `strict_mode=True` to reject (0.9) instead of guessing.
2. **Summarisation FPR at 10.5%**: Reduced from 95% → 25.5% (v3.4.0, direct NLI) → 10.5% (v3.5.0, bidirectional NLI + baseline=0.20). Remaining FP is FactCG model limitation on highly abstractive text. AggreFact-CNN: 68.8%, ExpertQA: 59.1%.
3. **ONNX CPU is slow**: 383 ms/pair without GPU. Use `onnxruntime-gpu` for production.
4. **Weights are domain-dependent**: Default `w_logic=0.6, w_fact=0.4` suits general QA. Adjust for your domain.
5. **Chunked NLI**: Very short chunks (<3 sentences) may lose context.
6. **LLM-as-judge sends data externally**: When `llm_judge_enabled=True`, truncated prompt+response (500 chars) are sent to the configured provider (OpenAI/Anthropic). Do not enable in privacy-sensitive deployments without user consent.
7. **guard() provider coverage**: `guard()` auto-detects OpenAI-compatible clients (OpenAI, vLLM, Groq, LiteLLM, Ollama, Together) via `client.chat.completions.create` and Anthropic via `client.messages.create`. AWS Bedrock, Google Gemini, and Cohere have different SDK shapes — use the low-level `CoherenceScorer.review()` API instead.

## Migrating to 3.0

v3.0 removed all deprecated 1.x aliases (`DirectorModule`, `BackfireKernel`,
`StrangeLoopAgent`, `KnowledgeBase`, `MockActor`, `RealActor`) and deprecated
methods (`calculate_factual_entropy`, `calculate_logical_entropy`,
`simulate_future_state`, `review_action`, `process_query`, `process_batch_async`).

Enterprise classes (`TenantRouter`, `Policy`, `Violation`, `AuditLogger`,
`AuditEntry`) moved to `director_ai.enterprise`:

```python
# Before (2.x):
from director_ai import TenantRouter
# After (3.0):
from director_ai.enterprise import TenantRouter
```

See [CHANGELOG](CHANGELOG.md) for the full list of breaking changes.

## Citation

```bibtex
@software{sotek2026director,
  author    = {Sotek, Miroslav},
  title     = {Director-AI: Real-time LLM Hallucination Guardrail},
  year      = {2026},
  url       = {https://github.com/anulum/director-ai},
  version   = {3.4.0},
  license   = {AGPL-3.0-or-later}
}
```

## License

Dual-licensed:

1. **Open-Source**: [GNU AGPL v3.0](LICENSE) — research, personal use, open-source projects.
2. **Commercial**: [Proprietary license](https://www.anulum.li/licensing) — removes copyleft for closed-source and SaaS.

See [Licensing](docs-site/licensing.md) for pricing tiers and FAQ.

Contact: [anulum.li/contact](https://www.anulum.li/contact.html) | invest@anulum.li

## Community

Join the [Director-AI Discord](https://discord.gg/JvMdKv49) for CI notifications, release announcements, and support. The Discord bot also provides `/version`, `/docs`, `/install`, `/status`, and `/quickstart` slash commands.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). By contributing, you agree to AGPL v3 terms.
