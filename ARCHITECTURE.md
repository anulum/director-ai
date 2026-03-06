# Architecture

Director-AI is a dual-entropy hallucination guardrail: NLI contradiction
detection + RAG fact-checking with token-level streaming halt.

## Directory Map

```
director-ai/
├── src/director_ai/
│   ├── core/
│   │   ├── scorer.py          CoherenceScorer — dual-entropy scoring
│   │   ├── kernel.py          SafetyKernel — output interlock
│   │   ├── streaming.py       StreamingKernel — token-level halt
│   │   ├── async_streaming.py AsyncStreamingKernel
│   │   ├── nli.py             NLIScorer — DeBERTa/FactCG/ONNX backends
│   │   ├── knowledge.py       GroundTruthStore — in-memory facts
│   │   ├── vector_store.py    VectorGroundTruthStore — ChromaDB/Pinecone/Qdrant
│   │   ├── agent.py           CoherenceAgent — orchestrator pipeline
│   │   ├── actor.py           LLMGenerator, MockGenerator
│   │   ├── batch.py           BatchProcessor — parallel evaluation
│   │   ├── config.py          DirectorConfig — YAML/JSON config
│   │   ├── audit.py           AuditLogger — JSONL audit trail
│   │   ├── tenant.py          TenantRouter — multi-tenant KB isolation
│   │   ├── sanitizer.py       InputSanitizer — prompt injection hardening
│   │   ├── backends.py        DeBERTa, ONNX, MiniCheck, Lite, Rust backends
│   │   ├── otel.py            OpenTelemetry spans
│   │   └── _heuristics.py     Word-overlap fallback scorer
│   │
│   ├── integrations/
│   │   ├── sdk_guard.py       guard() — OpenAI/Anthropic interceptor
│   │   ├── langchain_callback.py  LangChain Runnable
│   │   └── providers.py       LLM provider adapters
│   │
│   ├── enterprise/            Lazy-loaded enterprise modules
│   ├── cli.py                 CLI entry point
│   ├── server.py              FastAPI REST server
│   └── grpc_server.py         gRPC server
│
├── backfire-kernel/           Rust scorer backend (PyO3/maturin)
│   ├── Cargo.toml
│   └── crates/
│       └── backfire-ffi/      Python ↔ Rust bridge
│
├── tests/                     1869 tests, ≥90% coverage
├── benchmarks/                24 evaluators (AggreFact, FEVER, MNLI, ...)
├── notebooks/                 Jupyter quickstart + domain notebooks
├── docs-site/                 MkDocs documentation source
└── demo/                      HF Spaces Gradio demo
```

## Data Flow

```
LLM Provider ──► guard() / CoherenceAgent
                      │
                      ├──► CoherenceScorer
                      │       ├── NLIScorer (DeBERTa/FactCG/ONNX/Rust)
                      │       ├── GroundTruthStore / VectorGroundTruthStore
                      │       └── _heuristics (fallback)
                      │
                      ├──► StreamingKernel (token-level halt)
                      │
                      ├──► InputSanitizer (prompt injection)
                      │
                      └──► AuditLogger (JSONL)
                              │
                              ▼
                       User response (approved / halted)
```

## Scoring Pipeline

1. `InputSanitizer` checks prompt for injection patterns
2. `CoherenceScorer.review(prompt, response)`:
   - Chunk response if > 3 sentences
   - NLI entailment score per chunk (if `[nli]` installed)
   - RAG fact-check against `GroundTruthStore` (if facts loaded)
   - Weighted combination: `w_logic * nli + w_fact * rag`
   - LLM judge escalation (if enabled and score borderline)
3. `StreamingKernel` monitors per-token coherence during generation
4. Halt triggers: hard limit, gradient drop, sliding window average

## Backend Tiers

| Backend | Install | Accuracy | Latency |
|---------|---------|----------|---------|
| Heuristic | core | ~55% | <0.1 ms |
| DeBERTa | `[nli]` | ~73% | 197 ms |
| FactCG (ONNX) | `[nli,onnx]` | 75.8% | 14.6 ms |
| Rust (backfire) | `[rust]` | 75.8% | ~10 ms |
| Hybrid (NLI+Judge) | `[nli,openai]` | ~78% | 200-500 ms |

## Build Targets

| Target | Command |
|--------|---------|
| Python package | `pip install -e ".[dev]"` |
| Rust backend | `cd backfire-kernel && cargo build --release` |
| Tests (Python) | `pytest tests/ -v` |
| Tests (Rust) | `cd backfire-kernel && cargo test --workspace` |
| Docs | `mkdocs serve` |
| Benchmarks | `python -m benchmarks.run_all` |
