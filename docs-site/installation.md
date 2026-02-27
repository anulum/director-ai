# Installation

## Base Package

```bash
pip install director-ai
```

Includes: coherence scorer, streaming kernel, safety kernel, ground truth store, heuristic scoring.

## Optional Extras

| Extra | Command | Adds |
|-------|---------|------|
| `nli` | `pip install director-ai[nli]` | DeBERTa NLI model (torch, transformers) |
| `minicheck` | `pip install director-ai[minicheck]` | MiniCheck scorer (recommended for production) |
| `vector` | `pip install director-ai[vector]` | ChromaDB vector store |
| `embeddings` | `pip install director-ai[embeddings]` | sentence-transformers (bge-large) |
| `quantize` | `pip install director-ai[quantize]` | bitsandbytes 8-bit quantization |
| `openai` | `pip install director-ai[openai]` | OpenAI SDK guard |
| `anthropic` | `pip install director-ai[anthropic]` | Anthropic SDK guard |
| `langchain` | `pip install director-ai[langchain]` | LangChain integration |
| `llamaindex` | `pip install director-ai[llamaindex]` | LlamaIndex integration |
| `langgraph` | `pip install director-ai[langgraph]` | LangGraph integration |
| `haystack` | `pip install director-ai[haystack]` | Haystack integration |
| `crewai` | `pip install director-ai[crewai]` | CrewAI integration |
| `server` | `pip install director-ai[server]` | FastAPI server |
| `docs` | `pip install director-ai[docs]` | MkDocs documentation tools |
| `dev` | `pip install director-ai[dev]` | pytest, ruff, mypy, sphinx |

## Recommended Production Setup

```bash
pip install director-ai[minicheck,vector,embeddings,openai]
```

This gives you MiniCheck NLI (72.6% balanced accuracy), ChromaDB with bge-large embeddings, and OpenAI SDK interception.

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
    nli_quantize_8bit=True,  # <80ms per window on consumer GPU
)
```

## Python Version

Requires Python 3.10+.
