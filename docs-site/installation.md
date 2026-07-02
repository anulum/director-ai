# Installation

## Buyer-facing bundles

Three bundles compose the granular extras so you do not have to assemble them by
hand. The granular extras (below) remain available for fine control.

```bash
pip install director-ai                  # core: scoring + opt-in streaming checks, zero ML deps
pip install director-ai-lite             # smallest first-run guard facade
pip install "director-ai[recommended]"   # production guardrail: NLI + RAG + REST API
pip install "director-ai[integrations]"  # framework adapters (LangChain, LlamaIndex, …)
pip install "director-ai[all]"           # the common capability set in one shot
```

## Base Package

```bash
pip install director-ai
```

Includes: coherence scorer, streaming kernel, safety kernel, ground truth store, heuristic scoring.

## Director-Lite Package

```bash
pip install director-ai-lite
```

Use this package when the first install should expose only the guard facade:

```python
from director_ai_lite import guard

result = guard(
    token_stream,
    facts={"capital": "Paris is the capital of France."},
    prompt="What is the capital of France?",
)
```

`director-ai-lite` is a thin distribution wrapper around `director_ai.lite`; the
halt implementation, heuristic scorer, optional model-backed scorer path, and
session object are the same code used by the full `director-ai` package.

## Supported Default Path

For new users and contributors, the supported default is Python-only:

```bash
pip install director-ai[server,vector]
director-ai quickstart --run
director-ai doctor
```

This path keeps setup to the Python package, FastAPI service, local Chroma
persistence, and the CLI. Rust acceleration, the Go gateway, Julia tuning,
Lean verification, TensorRT, and WASM are advanced optional paths. They should
not be required for the base install, quickstart, or first contributor tasks.

Run `director-ai doctor` after enabling optional extras. It reports the active
runtime stack and warns when environment settings request optional components
that are not installed.

See [Runtime Boundaries](guide/runtime-boundaries.md) for the supported default
path and the advanced runtime boundary.

Contributors who only touch Python, docs, CLI, or server surfaces can run the
same supported default through the repository gate:

```bash
make python-only-check
make python-only-check PYTHON=.venv/bin/python PYTHON_ONLY_CHECK_ARGS="--no-tests"
```

## Recommended Setup

Use the default path for first local runs, first production trials, and first
contributor tasks:

```bash
pip install director-ai[server,vector]
director-ai quickstart --run
director-ai doctor
```

That gives one supported path before choosing specialized backends:

| Layer | Default Choice | Why |
|-------|----------------|-----|
| Runtime | Python package | Smallest supported surface |
| API | FastAPI service | Local and production-compatible HTTP path |
| Proxy | Generated quickstart proxy | Standard guarded-chat entry point |
| Retrieval | Local Chroma | Persistent facts without external services |
| Diagnostics | `director-ai doctor` | Checks optional runtime drift |

## Advanced Backend Matrix

Pick an advanced backend only when its capability is required. The base install
and quickstart must remain usable without these extras.

### Scoring and Inference

| Extra | Command | Adds |
|-------|---------|------|
| `nli` | `pip install director-ai[nli]` | FactCG-DeBERTa-v3-Large NLI (75.6% per-ds mean BA, 14.6ms/pair ONNX GPU) |
| `embed` | `pip install director-ai[embed]` | Embedding cosine-similarity scorer (~65% BA, 3ms CPU) |
| `nli-lite` | `pip install director-ai[nli-lite]` | Experimental distilled NLI backend; production claims require a validated student artefact |
| `minicheck` | `pip install director-ai[minicheck]` | MiniCheck alternative |
| `onnx` | `pip install director-ai[onnx]` | ONNX Runtime inference (14.6ms/pair GPU, portable CPU fallback) |
| `quantize` | `pip install director-ai[quantize]` | bitsandbytes 8-bit quantization |

### Retrieval and Knowledge Base

| Extra | Command | Adds |
|-------|---------|------|
| `vector` | `pip install director-ai[vector]` | ChromaDB vector store |
| `embeddings` | `pip install director-ai[embeddings]` | sentence-transformers (bge-large) |

### Provider and Agent Integrations

| Extra | Command | Adds |
|-------|---------|------|
| `openai` | `pip install director-ai[openai]` | OpenAI SDK guard |
| `anthropic` | `pip install director-ai[anthropic]` | Anthropic SDK guard |
| `langchain` | `pip install director-ai[langchain]` | LangChain integration |
| `llamaindex` | `pip install director-ai[llamaindex]` | LlamaIndex integration |
| `langgraph` | `pip install director-ai[langgraph]` | LangGraph integration |
| `haystack` | `pip install director-ai[haystack]` | Haystack integration |
| `crewai` | `pip install director-ai[crewai]` | CrewAI integration |
| `guardrails` | `pip install director-ai[guardrails] guardrails-ai` | Guardrails AI custom validator; install Guardrails AI from an index that provides it |
| Vercel AI SDK | `cd packages/vercel-ai && npm install && npm run build` | TypeScript AI SDK middleware |

### Operations, Training, and Development

| Extra | Command | Adds |
|-------|---------|------|
| `finetune` | `pip install director-ai[finetune]` | Experimental model-adaptation tooling for evidence-gated evaluation work; not a default production accuracy claim |
| `grpc` | `pip install director-ai[grpc]` | gRPC server (grpcio, protobuf) |
| `physical` | `pip install director-ai[physical]` | MuJoCo adapter support; ROS 2 and CARLA use their vendor installs behind the same runtime boundary |
| `server` | `pip install director-ai[server]` | FastAPI server |
| `otel` | `pip install director-ai[otel]` | OpenTelemetry tracing |
| `enterprise` | `pip install director-ai[enterprise]` | Multi-tenant, audit, Redis |
| `docs` | `pip install director-ai[docs]` | MkDocs documentation tools |
| `dev` | `pip install director-ai[dev]` | pytest, ruff, mypy, sphinx |

## Production Presets

```bash
pip install director-ai[nli,vector,embeddings,openai]
```

FactCG NLI (75.6% per-dataset mean BA, 14.6ms/pair ONNX GPU batch), ChromaDB + bge-large embeddings, OpenAI SDK interception.

For managed service deployments, start from the recommended setup first and add
provider, NLI, or observability extras only after `director-ai doctor` reports
the base stack as healthy.

## GPU Acceleration

For GPU-accelerated NLI scoring:

```bash
pip install director-ai[nli,quantize]
```

Set device and dtype:

```python
from director_ai import CoherenceScorer

scorer = CoherenceScorer(
    use_nli=True,
    nli_device="cuda",
    nli_torch_dtype="float16",
    nli_quantize_8bit=True,  # reduces VRAM, slightly slower than fp32
)
```

## Python Version

Requires Python 3.11+.
