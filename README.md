<p align="center">
  <img src="docs/assets/header.png" width="1280" alt="Director-AI — Real-time LLM Hallucination Guardrail">
</p>

<h1 align="center">Director-AI</h1>

<p align="center">
  <strong>Real-time LLM hallucination guardrail — NLI + RAG fact-checking with token-level streaming halt</strong>
</p>

<p align="center">
  <a href="https://github.com/anulum/director-ai/actions/workflows/ci.yml"><img src="https://github.com/anulum/director-ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/director-ai/"><img src="https://img.shields.io/pypi/v/director-ai.svg" alt="PyPI"></a>
  <a href="https://codecov.io/gh/anulum/director-ai"><img src="https://codecov.io/gh/anulum/director-ai/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://img.shields.io/badge/mypy-checked-blue"><img src="https://img.shields.io/badge/mypy-checked-blue.svg" alt="mypy"></a>
  <a href="https://hub.docker.com/r/anulum/director-ai"><img src="https://img.shields.io/badge/docker-ready-blue.svg" alt="Docker"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
  <a href="https://huggingface.co/spaces/anulum/director-ai-guardrail"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-orange.svg" alt="HF Spaces"></a>
  <a href="https://anulum.github.io/director-ai"><img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Docs"></a>
  <a href="https://discord.gg/director-ai"><img src="https://img.shields.io/badge/Discord-Join-7289DA.svg" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/anulum/director-ai-guardrail"><strong>Try it live on Hugging Face Spaces &rarr;</strong></a>
</p>

<p align="center">
  <a href="https://anulum.github.io/director-ai/pitch.html"><strong>Sales Pitch &amp; Pricing</strong></a> &middot;
  <a href="https://www.anulum.li/contact.html">Contact Sales</a> &middot;
  <a href="mailto:invest@anulum.li">invest@anulum.li</a>
</p>

---

## What It Does

Director-AI sits between your LLM and the user. It scores every output for
hallucination before it reaches anyone — and can halt generation mid-stream if
coherence drops below threshold.

```python
from director_ai import CoherenceAgent

agent = CoherenceAgent()
result = agent.process("What color is the sky?")

print(result.coherence.score)      # 0.94 — high coherence
print(result.coherence.approved)   # True
print(result.coherence.h_logical)  # 0.10 — low contradiction probability
print(result.coherence.h_factual)  # 0.10 — low factual deviation
```

**Three things make it different:**

1. **Token-level streaming halt** — not post-hoc review. The safety kernel
   monitors coherence token-by-token and severs output the moment it degrades.
2. **Dual-entropy scoring** — NLI contradiction detection (DeBERTa) + RAG
   fact-checking against your own knowledge base. Both must pass.
3. **Your data, your rules** — ingest PDFs, directories, or any text into a
   ChromaDB-backed knowledge base. The scorer checks LLM output against *your*
   ground truth, not a generic model.

## Architecture

```
          ┌──────────────────────────┐
          │    Coherence Agent       │
          │    (Orchestrator)        │
          └─────────┬────────────────┘
                    │
       ┌────────────┼────────────────┐
       │            │                │
┌──────▼──────┐ ┌───▼──────────┐ ┌───▼────────────┐
│  Generator  │ │  Coherence   │ │  Safety        │
│  (LLM       │ │  Scorer      │ │  Kernel        │
│   backend)  │ │              │ │  (streaming    │
│             │ │  NLI + RAG   │ │   interlock)   │
└─────────────┘ └───┬──────────┘ └────────────────┘
                    │
          ┌─────────▼─────────┐
          │  Ground Truth     │
          │  Store            │
          │  (ChromaDB / RAM) │
          └───────────────────┘
```

## Installation

```bash
# Basic install (heuristic scoring, no GPU needed)
pip install director-ai

# With NLI model (DeBERTa-based contradiction detection)
pip install director-ai[nli]

# With vector store (ChromaDB for custom knowledge bases)
pip install director-ai[vector]

# With high-quality embeddings (bge-large-en-v1.5)
pip install director-ai[embeddings]

# With 8-bit quantized NLI (<80ms on GPU)
pip install director-ai[quantize]

# Framework integrations
pip install director-ai[langchain]
pip install director-ai[llamaindex]
pip install director-ai[langgraph]
pip install director-ai[haystack]
pip install director-ai[crewai]

# With REST API server
pip install director-ai[server]

# Everything
pip install "director-ai[nli,vector,server,embeddings]"

# Development
git clone https://github.com/anulum/director-ai.git
cd director-ai
pip install -e ".[dev]"
```

## Usage

### Score a single response

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("sky color", "The sky is blue due to Rayleigh scattering.")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
approved, score = scorer.review("What color is the sky?", "The sky is green.")

print(approved)     # False — contradicts ground truth
print(score.score)  # 0.42
```

### With a real LLM backend

```python
from director_ai import CoherenceAgent

# Works with any OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama, etc.)
agent = CoherenceAgent(llm_api_url="http://localhost:8080/completion")
result = agent.process("Explain quantum entanglement")

if result.halted:
    print("Output blocked — coherence too low")
else:
    print(result.output)
```

### Token-level streaming with halt

```python
from director_ai.core import StreamingKernel

kernel = StreamingKernel(hard_limit=0.4, window_size=5, window_threshold=0.5)

session = kernel.stream_tokens(
    token_generator=my_token_iterator,
    coherence_callback=lambda tok: my_scorer(tok),
)

for event in session.events:
    if event.halted:
        print(f"\n[HALTED — {session.halt_reason}]")
        break
    print(event.token, end="")
```

### NLI-based scoring (requires torch)

```python
from director_ai.core import CoherenceScorer

scorer = CoherenceScorer(use_nli=True, threshold=0.6)
approved, score = scorer.review(
    "The Earth orbits the Sun.",
    "The Sun orbits the Earth."
)
print(score.h_logical)  # High — NLI detects contradiction
```

### Custom knowledge base with ChromaDB

```python
from director_ai.core import VectorGroundTruthStore

store = VectorGroundTruthStore()  # Uses ChromaDB
store.add_fact("company policy", "Refunds are available within 30 days.")
store.add_fact("pricing", "Enterprise plan starts at $99/month.")

scorer = CoherenceScorer(ground_truth_store=store)
approved, score = scorer.review(
    "What is the refund policy?",
    "We offer full refunds within 90 days."  # Wrong
)
# approved = False — contradicts your KB
```

### LangChain integration

```bash
pip install director-ai[langchain,nli]
```

```python
from director_ai.integrations.langchain import DirectorAIGuard

guard = DirectorAIGuard(
    facts={"refund": "Refunds available within 30 days."},
    threshold=0.6,
    use_nli=True,
)

# Pipe after any LLM in a chain
chain = my_llm | guard
result = chain.invoke({"query": "What is the refund policy?"})

print(result["approved"])  # False if hallucinated
print(result["score"])     # 0.0–1.0 coherence
```

Raises `HallucinationError` if `raise_on_fail=True`. Async supported via `ainvoke()`.

### LlamaIndex integration

```bash
pip install director-ai[llamaindex,nli]
```

```python
from director_ai.integrations.llamaindex import DirectorAIPostprocessor

postprocessor = DirectorAIPostprocessor(
    facts={"pricing": "Enterprise plan starts at $99/month."},
    threshold=0.6,
)

# Filters out hallucinated nodes before they reach the user
query_engine = index.as_query_engine(
    node_postprocessors=[postprocessor]
)
response = query_engine.query("What does Enterprise cost?")
```

Adds `director_ai_score` metadata to surviving nodes. Also usable standalone via `postprocessor.check(query, response)`.

### LangGraph integration

```python
from director_ai.integrations.langgraph import director_ai_node, director_ai_conditional_edge

node = director_ai_node(facts={"policy": "Refunds within 30 days."}, on_fail="flag")
edge = director_ai_conditional_edge("output", "retry")
# Wire into your LangGraph StateGraph
```

### Haystack integration

```python
from director_ai.integrations.haystack import DirectorAIChecker

checker = DirectorAIChecker(facts={"policy": "Refunds within 30 days."})
result = checker.run(query="Refund policy?", replies=["60-day refunds."])
print(result["scores"])  # [CoherenceScore(...)]
```

### CrewAI integration

```python
from director_ai.integrations.crewai import DirectorAITool

tool = DirectorAITool(facts={"policy": "Refunds within 30 days."})
result = tool.check("Refund policy?", "We offer 30-day refunds.")
print(result["approved"])  # True
```

### Score caching

```python
scorer = CoherenceScorer(
    threshold=0.6,
    cache_size=1024,   # LRU cache for streaming deduplication
    cache_ttl=300,     # TTL in seconds
)
```

### More examples

| Example | Backend | What it shows |
|---------|---------|---------------|
| [`quickstart.py`](examples/quickstart.py) | None | Guard any output in 10 lines |
| [`openai_guard.py`](examples/openai_guard.py) | OpenAI | Score + streaming halt for GPT-4o |
| [`ollama_guard.py`](examples/ollama_guard.py) | Ollama | Local LLM guard with Llama 3 |
| [`langchain_guard.py`](examples/langchain_guard.py) | LangChain | Full chain guardrail |
| [`streaming_halt_demo.py`](examples/streaming_halt_demo.py) | Simulated | All 3 halt mechanisms visualised |

### Interactive demo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/quickstart.ipynb)

```bash
pip install director-ai gradio
python demo/app.py
```

## Scoring Formula

```
Coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)
```

| Component | Source | Range | Meaning |
|-----------|--------|-------|---------|
| H_logical | NLI model (DeBERTa) | 0-1 | Contradiction probability |
| H_factual | RAG retrieval | 0-1 | Ground truth deviation |

- **Score >= 0.6** → approved (configurable)
- **Score < 0.5** → safety kernel emergency halt

## Benchmarks

Evaluated on [LLM-AggreFact](https://github.com/lytang/LLM-AggreFact) (29,320 samples across 11 datasets):

| Model | AggreFact Balanced Acc | Latency (avg) |
|-------|----------------------|---------------|
| DeBERTa-v3-base (baseline) | 66.2% | 220 ms |
| Fine-tuned DeBERTa-v3-large | 64.7% | 223 ms |
| Fine-tuned DeBERTa-v3-base | 59.0% | 220 ms |

**Per-dataset highlights:**

| Dataset | Balanced Accuracy | Notes |
|---------|------------------|-------|
| Reveal | 80.7% | Strong on factual claims |
| FactCheck-GPT | 71.7% | Good on GPT-generated text |
| Lfqa | 64.8% | Long-form QA |
| RAGTruth | 58.9% | RAG-specific hallucination |
| AggreFact-CNN | 53.0% | Summarization (known weak spot) |

**Head-to-head** (same benchmark, same metric — [LLM-AggreFact leaderboard](https://llm-aggrefact.github.io/)):

| Tool | Bal. Acc | Params | Latency | Streaming |
|------|---------|--------|---------|-----------|
| Bespoke-MiniCheck-7B | **77.4%** | 7B | ~100 ms (GPU) | No |
| MiniCheck-Flan-T5-L | 75.0% | 0.8B | ~120 ms | No |
| MiniCheck-DeBERTa-L | 72.6% | 0.4B | ~120 ms | No |
| HHEM-2.1-Open | 71.8% | ~0.4B | ~200 ms | No |
| **Director-AI** | **66.2%** | 0.4B | 220 ms | **Yes** |

**Honest assessment**: The NLI scorer alone is not state-of-the-art. Director-AI's
value is in the *system* — combining NLI with your own KB facts, streaming
token-level gating, and configurable halt thresholds. No competitor offers
real-time streaming halt. The NLI component is pluggable; swap in any model
that improves on these numbers.

Full comparison with SelfCheckGPT, RAGAS, NeMo Guardrails, Lynx, and others
in [`benchmarks/comparison/`](benchmarks/comparison/). Benchmark scripts in
`benchmarks/`. Fine-tuning pipeline in `training/`.

## Package Structure

```
src/director_ai/
├── core/                           # Production API
│   ├── agent.py                    # CoherenceAgent — main orchestrator
│   ├── scorer.py                   # Dual-entropy coherence scorer
│   ├── kernel.py                   # Safety kernel (streaming interlock)
│   ├── streaming.py                # Token-level streaming oversight
│   ├── async_streaming.py          # Non-blocking async streaming
│   ├── nli.py                      # NLI scorer (DeBERTa)
│   ├── actor.py                    # LLM generator interface
│   ├── knowledge.py                # Ground truth store (in-memory)
│   ├── vector_store.py             # Vector store (ChromaDB / sentence-transformers)
│   ├── cache.py                    # LRU score cache (blake2b, TTL)
│   ├── policy.py                   # YAML declarative policy engine
│   ├── audit.py                    # Structured JSONL audit logger
│   ├── tenant.py                   # Multi-tenant KB isolation
│   ├── sanitizer.py                # Prompt injection hardening
│   └── types.py                    # CoherenceScore, ReviewResult
├── integrations/                   # Framework integrations
│   ├── langchain.py                # LangChain Runnable guardrail
│   ├── llamaindex.py               # LlamaIndex postprocessor
│   ├── langgraph.py                # LangGraph node + conditional edge
│   ├── haystack.py                 # Haystack 2.x component
│   └── crewai.py                   # CrewAI tool
├── cli.py                          # CLI: review, process, batch, serve
├── server.py                       # FastAPI REST wrapper
benchmarks/                         # AggreFact evaluation suite
training/                           # DeBERTa fine-tuning pipeline
```

## Testing

```bash
pytest tests/ -v
```

## License & Pricing

Dual-licensed:

1. **Open-Source**: [GNU AGPL v3.0](LICENSE) — research, personal use, open-source
   projects. Full source, self-host, no restrictions beyond AGPL copyleft obligations.
2. **Commercial**: Proprietary license from [ANULUM](https://www.anulum.li/licensing)
   — removes copyleft, allows closed-source and SaaS deployment.

### Commercial Tiers

| Tier | Monthly | Yearly | Best for |
|------|---------|--------|----------|
| **Hobbyist** | $9 | $90 | Students, side projects, experiments. 1 local deployment, community support (GitHub/Discord), delayed updates. |
| **Indie** | $49 | $490 | Solo devs, bootstrapped teams (<$2M ARR). 1 production deployment, email support, 12 months updates. |
| **Pro** | $249 | $2,490 | Startups & scale-ups. Unlimited internal devs, multiple envs, Slack priority support, early releases. |
| **Enterprise** | Custom | Custom | Large orgs. SLA (99.9%), on-prem/air-gapped, SOC2/HIPAA-ready, dedicated engineer, custom NLI fine-tunes. |

**Perpetual license**: $1,299 one-time (Indie equivalent).
**First 50 commercial licensees**: 50% off first year.

Contact: [anulum.li/contact](https://www.anulum.li/contact.html) or invest@anulum.li

See [NOTICE](NOTICE) for full terms and third-party acknowledgements.

## Known Limitations

1. **Heuristic fallback is weak**: Without `pip install director-ai[nli]`,
   scoring uses word-overlap heuristics (~55% accuracy). Pass `strict_mode=True`
   to disable heuristics and return neutral 0.5 instead.
2. **Summarisation is a weak spot**: NLI models (including DeBERTa) under-perform
   on summarisation datasets (AggreFact-CNN: 53%). Best for factual QA and
   claim verification.
3. **Single-document NLI**: Long documents are scored as a single premise.
   Chunked NLI scoring is on the roadmap.
4. **Weights are domain-dependent**: Default `w_logic=0.6, w_fact=0.4` suits
   general QA. Adjust via constructor args for your domain.

## Roadmap

### Shipped in v1.2.0

- **Score caching** — LRU cache with blake2b keys and TTL for streaming dedup
- **Framework integrations** — LangGraph nodes, Haystack components, CrewAI tools
- **Quantized NLI** — 8-bit bitsandbytes quantization for <80ms GPU inference
- **Upgraded embeddings** — bge-large-en-v1.5 via `SentenceTransformerBackend`
- **MkDocs site** — full API reference, deployment guides, domain cookbooks
- **Enhanced demo** — side-by-side comparison with token-level highlighting

### Shipped in v1.1.0

- **Native SDK interceptors** — `guard(OpenAI(), facts={...})` wraps
  any OpenAI/Anthropic client with transparent coherence scoring
- **MiniCheck backend** — 72.6% balanced accuracy on LLM-AggreFact
- **Evidence return** — every `CoherenceScore` carries top-K chunks,
  NLI premise/hypothesis, and similarity distances
- **Graceful fallbacks** — `fallback="retrieval"` / `"disclaimer"` +
  soft warning zone + streaming `on_halt` callback

### Completed

- [x] Score caching, LangGraph/Haystack/CrewAI, quantized NLI, MkDocs site
- [x] `director-ai eval` — structured CLI benchmarking
- [x] Native OpenAI/Anthropic SDK interceptors (`guard()`)
- [x] Evidence schema on all rejections
- [x] Graceful fallback patterns (retrieval, disclaimer, soft warning)
- [x] End-to-end guardrail benchmark (600+ traces, 8 metrics)
- [x] HuggingFace Spaces live demo

### Next

- [ ] Chunked NLI scoring for long documents
- [ ] Prometheus histogram latency buckets
- [ ] `director-ai serve --workers N` multi-process mode

## Citation

```bibtex
@software{sotek2026director,
  author    = {Sotek, Miroslav},
  title     = {Director-AI: Real-time LLM Hallucination Guardrail},
  year      = {2026},
  url       = {https://github.com/anulum/director-ai},
  version   = {1.2.1},
  license   = {AGPL-3.0-or-later}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. By contributing, you agree
to the [Code of Conduct](CODE_OF_CONDUCT.md) and AGPL v3 licensing terms.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
