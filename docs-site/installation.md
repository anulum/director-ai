# Installation

## Base Package

```bash
pip install director-ai
```

Includes: coherence scorer, streaming kernel, safety kernel, ground truth store, heuristic scoring.

## Optional Extras

| Extra | Command | Adds |
|-------|---------|------|
| `nli` | `pip install director-ai[nli]` | FactCG-DeBERTa-v3-Large NLI (75.6% per-ds mean BA, 14.6ms/pair ONNX GPU) |
| `minicheck` | `pip install director-ai[minicheck]` | MiniCheck alternative (72.6% bal. acc) |
| `vector` | `pip install director-ai[vector]` | ChromaDB vector store |
| `embeddings` | `pip install director-ai[embeddings]` | sentence-transformers (bge-large) |
| `onnx` | `pip install director-ai[onnx]` | ONNX Runtime inference (14.6ms/pair GPU, portable CPU fallback) |
| `quantize` | `pip install director-ai[quantize]` | bitsandbytes 8-bit quantization |
| `openai` | `pip install director-ai[openai]` | OpenAI SDK guard |
| `anthropic` | `pip install director-ai[anthropic]` | Anthropic SDK guard |
| `langchain` | `pip install director-ai[langchain]` | LangChain integration |
| `llamaindex` | `pip install director-ai[llamaindex]` | LlamaIndex integration |
| `langgraph` | `pip install director-ai[langgraph]` | LangGraph integration |
| `haystack` | `pip install director-ai[haystack]` | Haystack integration |
| `crewai` | `pip install director-ai[crewai]` | CrewAI integration |
| `finetune` | `pip install director-ai[finetune]` | Domain-specific NLI fine-tuning (torch, transformers, datasets) |
| `grpc` | `pip install director-ai[grpc]` | gRPC server (grpcio, protobuf) |
| `server` | `pip install director-ai[server]` | FastAPI server |
| `otel` | `pip install director-ai[otel]` | OpenTelemetry tracing |
| `enterprise` | `pip install director-ai[enterprise]` | Multi-tenant, audit, Redis |
| `docs` | `pip install director-ai[docs]` | MkDocs documentation tools |
| `dev` | `pip install director-ai[dev]` | pytest, ruff, mypy, sphinx |

## Recommended Production Setup

```bash
pip install director-ai[nli,vector,embeddings,openai]
```

FactCG NLI (75.6% per-dataset mean BA, 14.6ms/pair ONNX GPU batch), ChromaDB + bge-large embeddings, OpenAI SDK interception.

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
