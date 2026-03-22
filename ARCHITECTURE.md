# Architecture

Director-AI is a dual-entropy hallucination guardrail: NLI contradiction
detection + RAG fact-checking with token-level streaming halt.

## Directory Map

```
director-ai/
├── src/director_ai/
│   ├── core/
│   │   ├── scoring/
│   │   │   ├── scorer.py          CoherenceScorer — dual-entropy scoring
│   │   │   ├── nli.py             NLIScorer — DeBERTa/FactCG/ONNX backends
│   │   │   ├── verified_scorer.py VerifiedScorer — sentence-level multi-signal
│   │   │   ├── meta_classifier.py DatasetTypeClassifier — adaptive thresholds
│   │   │   ├── lite_scorer.py     LiteScorer — zero-dep heuristic
│   │   │   ├── sharded_nli.py     ShardedNLIScorer — multi-GPU
│   │   │   ├── backends.py        DeBERTa, ONNX, MiniCheck, Lite, Rust
│   │   │   └── _heuristics.py     Word-overlap fallback
│   │   ├── retrieval/
│   │   │   ├── knowledge.py       GroundTruthStore — in-memory facts
│   │   │   ├── vector_store.py    VectorGroundTruthStore + 11 backends
│   │   │   ├── doc_chunker.py     Document chunking
│   │   │   ├── doc_parser.py      PDF/DOCX parsing
│   │   │   ├── doc_registry.py    Document metadata registry
│   │   │   └── embedding_tuner.py Domain embedding fine-tuner
│   │   ├── runtime/
│   │   │   ├── kernel.py          HaltMonitor — output interlock
│   │   │   ├── streaming.py       StreamingKernel — token-level halt
│   │   │   ├── async_streaming.py AsyncStreamingKernel
│   │   │   ├── batch.py           BatchProcessor — parallel evaluation
│   │   │   ├── review_queue.py    ReviewQueue — continuous batching
│   │   │   └── session.py         ConversationSession — multi-turn
│   │   ├── safety/
│   │   │   ├── sanitizer.py       InputSanitizer — prompt injection
│   │   │   ├── policy.py          Policy — rule engine
│   │   │   └── audit.py           AuditLogger — JSONL audit trail
│   │   ├── training/
│   │   │   ├── finetune.py        NLI fine-tuning
│   │   │   ├── finetune_benchmark.py  Pre/post benchmark
│   │   │   ├── finetune_validator.py  Data validation
│   │   │   └── tuner.py           Threshold grid search
│   │   ├── agent.py               CoherenceAgent — orchestrator
│   │   ├── actor.py               LLMGenerator, MockGenerator
│   │   ├── config.py              DirectorConfig — YAML/env/profile
│   │   ├── cache.py               ScoreCache — LRU
│   │   ├── types.py               CoherenceScore, ReviewResult, etc.
│   │   ├── tenant.py              TenantRouter — multi-tenant
│   │   └── otel.py                OpenTelemetry spans
│   │
│   ├── integrations/
│   │   ├── sdk_guard.py           guard() — 5 SDK shapes
│   │   ├── langchain.py           LangChain Runnable
│   │   ├── llamaindex.py          LlamaIndex NodePostprocessor
│   │   ├── langgraph.py           LangGraph node/edge
│   │   ├── haystack.py            Haystack 2.x component
│   │   ├── crewai.py              CrewAI tool
│   │   └── fastapi_guard.py       FastAPI middleware
│   │
│   ├── cli.py                     CLI (18 commands)
│   ├── server.py                  FastAPI REST server
│   ├── grpc_server.py             gRPC server
│   ├── knowledge_api.py           Document ingestion API router
│   └── proxy.py                   OpenAI-compatible guardrail proxy
│
├── backfire-kernel/               Rust scorer backend (PyO3/maturin)
│
├── tests/                         2800+ tests, ≥90% coverage
├── benchmarks/                    27 evaluators
├── notebooks/                     16 Jupyter notebooks
├── docs-site/                     MkDocs documentation
└── demo/                          HF Spaces Gradio demo
```

## Data Flow

```
LLM Provider ──► guard() / CoherenceAgent
                      │
                      ├──► CoherenceScorer
                      │       ├── H_logical + H_factual (parallel)
                      │       ├── NLIScorer (DeBERTa/FactCG/ONNX/Rust)
                      │       ├── GroundTruthStore / VectorGroundTruthStore
                      │       ├── review_batch() — coalesced NLI (2 kernels)
                      │       └── _heuristics (fallback)
                      │
                      ├──► ReviewQueue (continuous batching)
                      │       └── accumulate → flush → review_batch()
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
| Heuristic (lite) | core | ~65% | <0.5 ms |
| DeBERTa | `[nli]` | 75.8% | 197 ms (CPU), 19 ms (GPU batch) |
| FactCG (ONNX) | `[nli,onnx]` | 75.8% | 14.6 ms (GPU batch) |
| Rust (backfire) | `[rust]` | ~65% | ~1 ms |
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
