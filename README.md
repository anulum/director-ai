<p align="center">
  <img src="docs/assets/header.png" width="1280" alt="Director AI — Coherence Engine">
</p>

<h1 align="center">Director AI</h1>

<p align="center">
  <strong>Drop-in coherence guardrail for llama.cpp, vLLM, OpenAI, and any LLM.<br>
  NLI + vector RAG fact-checking. 4.56 &micro;s scoring. Rust hot path.</strong>
</p>

<p align="center">
  <a href="https://github.com/anulum/director-ai/actions/workflows/ci.yml"><img src="https://github.com/anulum/director-ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/anulum/director-ai/releases"><img src="https://img.shields.io/badge/version-0.8.1-green.svg" alt="Version 0.8.0"></a>
</p>

---

## What It Does

Score any LLM output for hallucination. Halt incoherent token streams in real time.
Fact-check responses against your own knowledge base.

```python
from director_ai.core import CoherenceScorer

scorer = CoherenceScorer(threshold=0.6, use_nli=True)
approved, score = scorer.review(
    "What year was the Eiffel Tower built?",
    "The Eiffel Tower was completed in 1889."
)
print(f"Approved: {approved}, Coherence: {score.score:.2f}")
# Approved: True, Coherence: 0.94
```

### Key Metric

```
Coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)
```

- **H_logical**: NLI contradiction probability (DeBERTa-v3-mnli-fever-anli)
- **H_factual**: RAG ground truth deviation (ChromaDB + sentence-transformers)
- Score < 0.6 = rejected. Score < 0.5 = emergency stop.

## Installation

```bash
pip install director-ai                    # Lightweight (~5 MB, no torch)
pip install director-ai[nli]               # + DeBERTa NLI model (~2 GB)
pip install director-ai[vector]            # + ChromaDB + sentence-transformers
pip install director-ai[server]            # + FastAPI server
pip install director-ai[nli,vector,server] # Everything
```

## Quick Start

### 1. Score a response

```python
from director_ai.core import CoherenceScorer

scorer = CoherenceScorer(threshold=0.6, use_nli=True)
approved, score = scorer.review("Capital of France?", "The capital is Paris.")
```

### 2. Stream-gate a chatbot

```python
from director_ai.core import CoherenceAgent
from director_ai.integrations import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")
agent = CoherenceAgent()
session = agent.process_streaming("Explain gravity", provider=provider)

if session.halted:
    print(f"Halted at token {session.halt_index}: {session.halt_reason}")
else:
    print(session.output)
```

### 3. Batch-audit Q&A pairs

```python
from director_ai.core import BatchProcessor, CoherenceAgent

agent = CoherenceAgent()
processor = BatchProcessor(agent)
result = processor.process_batch(["What is 2+2?", "Explain DNA"])
print(f"Passed: {result.succeeded}/{result.total}")
```

### 4. Ingest your own knowledge base

```python
from director_ai.core import VectorGroundTruthStore, ChromaBackend

store = VectorGroundTruthStore(
    backend=ChromaBackend(persist_directory="./my_facts"),
)
store.ingest([
    "The speed of light is 299,792,458 m/s.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Earth orbits the Sun at 149.6 million km.",
])

# Now score against your facts
from director_ai.core import CoherenceScorer
scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
approved, score = scorer.review("How fast is light?", "Light is about 300,000 km/s.")
```

### 5. Ingest a directory of documents

```python
from director_ai.core import VectorGroundTruthStore, ChromaBackend

store = VectorGroundTruthStore(
    backend=ChromaBackend(persist_directory="./kb"),
)
n = store.ingest_from_directory("./docs", glob="**/*.md")
print(f"Indexed {n} documents")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
```

### 6. CLI

```bash
director-ai review "What color is the sky?" "The sky is blue."
director-ai process "Explain photosynthesis"
director-ai batch input.jsonl --output results.jsonl
director-ai ingest facts.txt --persist ./chroma_data
director-ai serve --port 8080
```

## Integrations

### LLM Providers

Built-in adapters with real SSE streaming for OpenAI, Anthropic, HuggingFace,
and local servers (llama.cpp, vLLM, Ollama):

```python
from director_ai.integrations import OpenAIProvider, AnthropicProvider, LocalProvider

provider = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")
for token in provider.stream_generate("Explain gravity"):
    print(token, end="", flush=True)
```

### LangChain

```python
from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

chain = LLMChain(llm=llm, callbacks=[CoherenceCallbackHandler(threshold=0.6)])
```

Install with: `pip install director-ai[langchain]`

## Architecture

```
                    ┌─────────────────────────┐
                    │   Coherence Agent        │
                    │   (Main Orchestrator)    │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼──────┐ ┌───────▼────────┐
    │  Generator     │ │ Coherence   │ │  Streaming     │
    │  (LLM Provider │ │ Scorer      │ │  Kernel        │
    │   + Streaming) │ │ (NLI+RAG)   │ │  (Token Gate)  │
    └────────────────┘ └──────┬──────┘ └────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Vector Store     │
                    │  (Chroma/Memory)  │
                    └───────────────────┘
```

| Module | Purpose |
|--------|---------|
| `CoherenceScorer` | Dual-entropy scorer: NLI contradiction + RAG fact-check |
| `StreamingKernel` | Token-by-token gate: hard limit + sliding window + trend detection |
| `CoherenceAgent` | Orchestrator: generate candidates, score, emit best |
| `VectorGroundTruthStore` | Semantic retrieval via ChromaDB + sentence-transformers |
| `BatchProcessor` | Bulk scoring with concurrency control |
| `OpenAI/Anthropic/HF/LocalProvider` | LLM adapters with SSE streaming |

## Benchmarks

Benchmark scripts evaluate the coherence scorer on public hallucination-detection
datasets. They require the DeBERTa NLI model (`~2 GB` download on first run):

```bash
pip install director-ai[nli]
python -m benchmarks.truthfulqa_eval 50    # TruthfulQA multiple-choice
python -m benchmarks.halueval_eval 100     # HaluEval QA/summarization/dialogue
```

Results are environment-dependent (model weights, threshold). Run locally and
compare against your baseline.

## Performance (Rust Hot Path)

The `backfire-kernel/` Rust workspace reimplements safety-critical paths with
PyO3 FFI. All operations clear the 50 ms deadline by orders of magnitude:

| Operation | Latency | Headroom |
|-----------|---------|----------|
| Safety kernel (10 tokens) | 265 ns | 188,679x |
| Full scoring pipeline | 4.56 &micro;s | 10,965x |
| SSGF gradient (analytic Jacobian) | 4.14 &micro;s | 12,077x |

## Testing

```bash
pytest tests/ -v                 # 397+ tests
pytest tests/test_consumer_api.py -v  # Consumer API only
pytest benchmarks/ -v -m slow    # Benchmarks (requires NLI)
cd backfire-kernel && cargo test  # 153 Rust tests
```

<details>
<summary><strong>Research Extensions</strong></summary>

Optional SCPN research modules (Lyapunov stability, UPDE integrators,
consciousness gate). Zero cross-imports with core.

```bash
pip install director-ai[research]
```

See [docs/RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md) for details.

</details>

## License

Dual-licensed:

1. **Open-Source**: [GNU AGPL v3.0](LICENSE) — academic, personal, open-source
2. **Commercial**: [ANULUM](https://www.anulum.li/licensing) — closed-source

## Citation

```bibtex
@software{sotek2026director,
  author  = {Sotek, Miroslav},
  title   = {Director AI: Coherence Engine},
  year    = {2026},
  url     = {https://github.com/anulum/director-ai},
  version = {0.8.1},
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). By contributing you agree to AGPL v3 terms.
