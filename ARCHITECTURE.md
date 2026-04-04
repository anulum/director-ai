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
│   │   │   ├── _llm_judge.py      LLMJudge — LLM-as-judge escalation
│   │   │   ├── _task_scoring.py   Task-type detection + dialogue/summarisation
│   │   │   ├── nli.py             NLIScorer — DeBERTa/FactCG backends
│   │   │   ├── _nli_export.py     ONNX/TensorRT export + dynamic batcher
│   │   │   ├── verified_scorer.py VerifiedScorer — sentence-level multi-signal
│   │   │   ├── meta_classifier.py DatasetTypeClassifier — adaptive thresholds
│   │   │   ├── meta_confidence.py Meta-confidence signal computation
│   │   │   ├── lite_scorer.py     LiteScorer — zero-dep heuristic
│   │   │   ├── sharded_nli.py     ShardedNLIScorer — multi-GPU
│   │   │   ├── backends.py        DeBERTa, ONNX, MiniCheck, Lite, Rust
│   │   │   ├── consensus.py       Cross-model factual agreement
│   │   │   ├── temporal_freshness.py  Staleness risk scoring
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
│   │   │   ├── sanitizer.py       InputSanitizer — prompt injection (Stage 1: regex)
│   │   │   ├── injection.py       InjectionDetector — intent-grounded detection (Stage 2: NLI)
│   │   │   ├── policy.py          Policy — rule engine
│   │   │   └── audit.py           AuditLogger — JSONL audit trail
│   │   ├── verification/           (v3.10.0 — stdlib only)
│   │   │   ├── json_verifier.py   JSON Schema + value grounding
│   │   │   ├── tool_call_verifier.py  Tool existence + fabrication
│   │   │   ├── code_verifier.py   Python AST + import + API check
│   │   │   └── types.py           Result dataclasses
│   │   ├── calibration/            (v3.10.0)
│   │   │   ├── feedback_store.py  SQLite human correction store
│   │   │   ├── online_calibrator.py  Threshold sweep + CIs
│   │   │   └── conformal.py       Conformal prediction intervals
│   │
│   ├── compliance/                 (v3.10.0 — EU AI Act Article 15)
│   │   ├── audit_log.py           Scored interaction audit trail
│   │   ├── reporter.py            Article15Report + metrics + markdown
│   │   ├── drift_detector.py      Statistical drift (z-test, severity)
│   │   └── feedback_loop_detector.py  Art 15(4) feedback loop detection
│   │
│   ├── agentic/                    (v3.10.0 — agent loop safety)
│   │   └── loop_monitor.py        Circular call, goal drift, budget monitor
│   │
│   │   ├── testing/                (v3.10.0 — self-test)
│   │   │   └── adversarial_suite.py   25 hallucination + 27 injection adversarial patterns
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
│   │   ├── voice.py               VoiceGuard — sync token filter for TTS
│   │   ├── langchain.py           LangChain Runnable
│   │   ├── llamaindex.py          LlamaIndex NodePostprocessor
│   │   ├── langgraph.py           LangGraph node/edge
│   │   ├── haystack.py            Haystack 2.x component
│   │   ├── crewai.py              CrewAI tool
│   │   └── fastapi_guard.py       FastAPI middleware
│   │
│   ├── voice/                     (v3.12 — async voice AI pipeline)
│   │   ├── guard.py               AsyncVoiceGuard — async token scoring
│   │   ├── adapters.py            TTSAdapter ABC + ElevenLabs, OpenAI, Deepgram
│   │   └── pipeline.py            voice_pipeline() — guard + TTS → audio bytes
│   │
│   ├── cli.py                     CLI dispatcher (25 commands)
│   ├── _cli_bench.py              CLI: eval/bench/tune/finetune/export
│   ├── _cli_serve.py              CLI: serve/proxy/stress-test
│   ├── _cli_verify.py             CLI: doctor/license/compliance/verify
│   ├── _cli_ingest.py             CLI: document ingestion
│   ├── server.py                  FastAPI REST server
│   ├── _server_models.py          Pydantic request/response models
│   ├── _server_helpers.py         Evidence serialisation helpers
│   ├── grpc_server.py             gRPC server
│   ├── knowledge_api.py           Document ingestion API router
│   └── proxy.py                   OpenAI-compatible guardrail proxy
│
├── backfire-kernel/               Rust scorer backend (PyO3/maturin)
│
├── tests/                         4120+ tests, ≥90% coverage
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
                      │       ├── review_batch() — coalesced NLI (2 GPU calls)
                      │       └── _heuristics (fallback)
                      │
                      ├──► ReviewQueue (continuous batching)
                      │       └── accumulate → flush → review_batch()
                      │
                      ├──► StreamingKernel (token-level halt)
                      │
                      ├──► InputSanitizer (Stage 1: regex injection detection)
                      │
                      ├──► InjectionDetector (Stage 2: NLI intent-drift detection)
                      │
                      └──► AuditLogger (JSONL)
                              │
                              ▼
                       User response (approved / halted)
```

## Scoring Pipeline

1. `InputSanitizer` checks prompt for injection patterns (Stage 1)
2. `InjectionDetector` measures output divergence from intent via bidirectional NLI (Stage 2, optional)
3. `CoherenceScorer.review(prompt, response)`:
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
