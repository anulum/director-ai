# Changelog

## v1.2.0 (2026-02-27)

### Features
- **Evidence return**: every `CoherenceScore` carries `ScoringEvidence` with top-K chunks, NLI premise/hypothesis, similarity distances
- **Fallback modes**: `fallback="retrieval"` swaps in KB answer, `"disclaimer"` prepends confidence warning
- **Soft warning zone**: configurable `soft_limit` between threshold and full-pass, with disclaimer injection
- **Score caching**: LRU cache keyed on (query, prefix) to avoid redundant NLI/embedding computation in streams
- **8-bit quantization**: `nli_quantize_8bit=True` via bitsandbytes for <80ms per scoring window on consumer GPU
- **Configurable device/dtype**: `nli_device="cuda"`, `nli_torch_dtype="float16"` on CoherenceScorer
- **SentenceTransformerBackend**: direct embedding backend using bge-large-en-v1.5
- **ChromaDB custom embedder**: `embedding_model` parameter on ChromaBackend
- **LangGraph integration**: `director_ai_node()` and `director_ai_conditional_edge()`
- **Haystack integration**: `DirectorAIChecker` component for Haystack 2.x pipelines
- **CrewAI integration**: `DirectorAITool` for agent fact-checking
- **Enhanced HF Spaces demo**: side-by-side raw vs guarded comparison tab with token highlighting
- **MkDocs documentation site**: full API reference, deployment guide, cookbook

### Bug Fixes
- `AsyncStreamingKernel.on_halt` now properly wired (was silently ignored)
- MiniCheck AggreFact benchmark OOM fix: catches CUDA OOM, clears cache, continues

### Documentation
- MiniCheck documented as recommended NLI backend
- Production deployment guide with Docker, metrics, scaling
- Configuration cookbook for legal, medical, finance domains

## v1.1.0 (2026-02-27)

### Features
- **SDK Guard**: `guard()` wraps OpenAI/Anthropic clients with 2-line integration
- **MiniCheck NLI**: pluggable backend with 72.6% balanced accuracy on AggreFact
- **Streaming halt**: 3 mechanisms (hard limit, sliding window, downward trend) + `on_halt` callback
- **E2E benchmark**: 300 HaluEval traces, QA precision 81.8%, p95 latency 40ms
- **HF Spaces demo**: live interactive demo

## v1.0.0 (2026-02-26)

- Initial production release
- Coherence engine with dual-entropy scoring
- Safety kernel with emergency stop
- Ground truth store (keyword + vector)
- Policy engine, audit logging, multi-tenant support
