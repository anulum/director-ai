# Architecture

Last updated: 2026-04-29

Director-AI is a dual-entropy hallucination guardrail: NLI contradiction
detection + RAG fact-checking with token-level streaming halt.

Python is the primary runtime. Rust handles hot-path compute through
`backfire-kernel` (PyO3). A Go gateway fronts the deployment when
operators want a high-concurrency HTTP entry point, and calls back
into the Python scorer over gRPC. A Julia analytics module runs
offline on exported score logs. A Lean 4 proof artefact pins the
halt-monitor safety contract. All language components are additive —
the Python path stands on its own without any of them.

## Shipped Today - 2026-04-29

- The Safety Surface Map is now the public boundary between default halt
  coverage, opt-in runtime hooks, advanced runtime paths, and disabled research
  modules.
- `CoherenceAgent` hook wiring now covers containment guards, containment
  anchors, physical-grounding hooks, and passport verifiers as optional
  constructor inputs.
- Rust acceleration for safety primitives is documented as an optional
  `backfire-kernel` path with pure-Python fallback.
- Roadmap status now separates completed simplification work from planned
  physical, attestation, and Rust-extraction hardening.

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
│   │   ├── trajectory/             Monte-Carlo pre-execution simulator
│   │   ├── routing/                Predictive prompt-risk routing
│   │   ├── trace_safe/             Mid-trajectory safety oracle
│   │   ├── policy_compiler/        Compliance-doc → Policy pipeline
│   │   ├── causal_verifier/        Causal counterfactual verifier
│   │   ├── symbolic_chain/         Neural-symbolic reasoning chain
│   │   ├── ontology/               Ontological consistency oracle
│   │   ├── irreversibility/        Point-of-no-return forecaster
│   │   ├── multimodal_guard/       Vision-NLI hallucination guard
│   │   ├── self_evolving/          Online LoRA micro-fine-tune worker
│   │   ├── knowledge_graph/        Skill-graph with policy-aware walks
│   │   ├── agent_identity/         Signed passport + behavioural fingerprint
│   │   ├── meta_guard/             Self-referential scorer
│   │   ├── defense_genome/         Evolutionary adversarial registry
│   │   ├── swarm_equilibrium/      Nash / Stackelberg stability scorer
│   │   ├── emergence_oracle/       Interaction-graph swarm forecaster
│   │   ├── autopoietic/            Meta-layer module hot-swap
│   │   ├── multi_scale_alignment/  Agent → swarm → org value lattice
│   │   ├── provenance/             HMAC + Merkle citation integrity
│   │   ├── formal_verification/    DPLL built-in + Z3 / Lean adapters
│   │   ├── federated_privacy/      MPC / DP failure-pattern sharing
│   │   ├── continual_adversarial/  Auto-generated adversarial suites
│   │   ├── swarm_economics/        Inter-agent economic risk scorer
│   │   ├── sustainability/         Multi-day budget + carbon throttle
│   │   ├── cyber_physical/         Pre-action physical-grounding hook
│   │   ├── containment/            HMAC reality anchor + breakout guard
│   │   ├── zk_attestation/         Cross-org passports with Merkle proofs
│   │   ├── observability/          Per-token OTEL + Langfuse tracing
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
│   │   ├── inference_server_hooks.py  vLLM/TGI/llama.cpp pre-sampling hook
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
│   ├── grpc_server.py             legacy DirectorService gRPC (Python-only)
│   ├── grpc_scoring.py            director.v1 CoherenceScoring gRPC
│   ├── proto/                     generated director.v1 Python stubs
│   ├── knowledge_api.py           Document ingestion API router
│   └── proxy.py                   OpenAI-compatible guardrail proxy
│
├── backfire-kernel/               Rust scorer backend (PyO3/maturin)
│   └── crates/backfire-core/src/
│       ├── compute.rs             12 Rust compute functions (sanitizer, unicode,
│       │                          task type, numeric, temporal, reasoning,
│       │                          word overlap, NLI softmax/div/conf, lite score)
│       ├── signals.rs             VerifiedScorer signals (entity, negation, etc.)
│       └── kernel.rs              Safety kernel, streaming gate
│
├── tests/                         4120+ tests, ≥90% coverage
├── benchmarks/                    28 evaluators
│   └── rust_compute_bench.py      Rust vs Python benchmark (10 compute fns)
├── notebooks/                     16 Jupyter notebooks
├── docs-site/                     MkDocs documentation
├── demo/                          HF Spaces Gradio demo
│
├── schemas/
│   ├── proto/director/v1/director.proto   wire schema (frozen v1)
│   └── generate.sh                regenerates Python + Go stubs
│
├── gateway/go/                    Go gateway — HTTP front door
│   ├── cmd/director-gateway/      binary entrypoint
│   ├── internal/config,auth,ratelimit,proxy,audit,server,scoring/
│   ├── proto/director/v1/         generated Go stubs
│   └── bench/                     k6 load scripts + A/B bench
│
├── tools/julia_tuner/             Julia offline analytics
│   └── src/DirectorThresholdTuner.jl   Bayesian + bootstrap threshold
│
└── formal/HaltMonitor/            Lean 4 safety proofs
    └── HaltMonitor/{Core,Properties}.lean
```

## Language roles

| Language | Where | Purpose |
|----------|-------|---------|
| Python   | `src/director_ai/` | primary API, scoring, RAG, CLI, servers |
| Rust     | `backfire-kernel/` | hot-path compute via PyO3 |
| Go       | `gateway/go/` | concurrent HTTP front door with optional scoring sidecar |
| Julia    | `tools/julia_tuner/` | offline threshold tuning with uncertainty bands |
| Lean 4   | `formal/HaltMonitor/` | machine-checked proof that no sub-threshold token is emitted |

## Safety Surface Map

The safety surface is intentionally split into default, opt-in, advanced, and
research-only layers. A module listed in the tree above is not automatically
part of the default halt path; it must appear in the default layer below or be
configured explicitly by the caller.

### Default halt path

These components are active in the normal Python path used by `guard()`,
`CoherenceAgent`, the FastAPI server, and the proxy unless the caller chooses a
lighter profile:

| Layer | Components | Default responsibility |
|-------|------------|------------------------|
| Input filtering | `InputSanitizer`, regex injection checks, PII detectors when enabled by policy | Reject or redact unsafe input before generation/scoring. |
| Factual scoring | `CoherenceScorer`, `NLIScorer` when `[nli]` is installed, `GroundTruthStore`, `VectorGroundTruthStore` | Score logical and factual consistency against configured facts. |
| Interlock | `HaltMonitor`, `StreamingKernel` | Stop output when the coherence floor, window average, or trend rule fails. |
| Evidence | `HaltEvidence`, `HaltTraceAttribution`, counterfactual halt diagnostics, top-K contradictory chunks, scorer metadata | Carry machine-readable reason data, trace attribution, single-fact diagnostics, and halt margins into API responses, logs, and OpenTelemetry spans. |
| Audit | `AuditLogger`, hashed prompt metadata, tenant ids | Persist tenant-safe records for review and compliance reporting. |

### Scientific deployment hook decision table

Start with the default halt path, then add only the hooks that match the
deployment risk. Research modules stay disabled until there is a named test
packet and rollback path for that deployment.

| Deployment need | Enable first | Add when needed | Keep disabled until |
|-----------------|--------------|-----------------|---------------------|
| Factual RAG or paper QA | `CoherenceScorer`, `[nli]`, `GroundTruthStore` or `VectorGroundTruthStore`, `HaltMonitor` | `InjectionDetector`, structured verifiers, audit logging | Physical hooks, passports, and experimental modules |
| Public API or proxy | Default halt path, `InputSanitizer`, `AuditLogger`, tenant ids | `InjectionDetector`, OpenTelemetry spans, review queue | Cross-org passports and physical adapters |
| Multi-agent tool workflow | Default halt path, structured verifiers, trace attribution | `ContainmentGuard`, `RealityAnchor`, `PassportVerifier` for hand-off | Self-evolving, swarm, and autopoietic research modules |
| Lab automation or robotics | Default halt path plus dry physical-action review outside the live actuator path | `GroundingHook` after per-tenant budgets, adapter isolation, and action replay tests | Blocking real-world actions without the high-risk physical flag and external test packet |
| Cross-organisation hand-off | Default halt path plus signed audit records | `PassportVerifier` with pinned issuer keys and Merkle proof checks | zk proof backends until fuzzing and cross-language contract tests pass |
| Research evaluation | Default halt path with fixed datasets and repeatable seeds | `director_ai.experimental` hooks one at a time | Any hook without a rollback plan or validation card |

### Opt-in runtime hooks

These hooks are implemented and tested, but they are inert until passed to the
agent, selected through configuration, or installed through the matching extra:

| Hook | Activation boundary | Role |
|------|---------------------|------|
| `InjectionDetector` | Stage 2 detector selected in policy/config | NLI-based intent-drift detection after regex filtering. |
| `ReviewQueue` | Continuous batching config | Coalesce scorer calls for higher-throughput services. |
| `AsyncStreamingKernel` | Async voice or streaming integration | Async token oversight with timeout and cancellation handling. |
| `InferenceServerHook` | vLLM, TGI, or llama.cpp adapter calls `check()` before accepting a candidate token | Mask or reject a candidate token and emit one `SafetyEvent` with `hook_scope="inference_server"`. |
| `ContainmentGuard` + `RealityAnchor` | `CoherenceAgent(containment_guard=..., containment_anchor=...)` | Verify anchored execution scope and block breakout findings. |
| `GroundingHook` | `CoherenceAgent(grounding_hook=...)` | Check a proposed physical action against kinematics and constraints. |
| `PassportVerifier` | `CoherenceAgent(passport_verifier=...)` | Verify cross-org attestation bundles before agent hand-off. |
| Structured verifiers | `verify_json`, `verify_tool_call`, `verify_code`, `verify_numeric`, `verify_reasoning_chain` | Validate structured outputs on explicit request. |
| Observability callbacks | tracing config or callback list | Emit token traces and spans without changing halt decisions. |

### CoherenceAgent hook wiring

`CoherenceAgent` remains usable without optional hooks. When a deployment
passes hook instances explicitly, the agent wires them as follows:

| Hook input | Runtime effect |
|------------|----------------|
| `containment_guard` + `containment_anchor` | Check each completed output against the anchored execution scope before returning it. |
| `grounding_hook` | Validate proposed physical actions against kinematic and constraint checks through `verify_physical_action`. |
| `passport_verifier` | Validate cross-organisation passport bundles through `verify_passport` before agent hand-off. |

### Advanced runtime surface

These paths are operationally useful but outside the default support surface.
They must be enabled through extras, Compose profiles, separate build commands,
or deployment-specific configuration:

| Surface | Boundary |
|---------|----------|
| Rust `backfire-kernel` | `[rust]`, `maturin`, or packaged wheel; Python fallback remains valid. |
| Go gateway | Separate `gateway/go` build and gRPC scoring sidecar. |
| Julia tuner | Offline score-log analytics only. |
| Lean proof artefacts | Separate `lake build`; proof surface does not run in the Python API. |
| ONNX/TensorRT | `[onnx]`/GPU deployment paths and exported model artefacts. |
| Voice adapters | `[voice]` plus provider-specific credentials and TTS/STT packages. |
| Vector DB vendors | `[vector]` plus selected backend dependency and deployment config. |

### Research modules disabled by default

These modules are not default halt coverage. Treat them as research or
advanced integration points until a roadmap item promotes them into one of the
layers above. Public access goes through `director_ai.experimental`, which
requires either `DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS=1` or an explicit
`enable_experimental_hooks()` call before loading a hook:

```python
from director_ai import experimental

experimental.enable_experimental_hooks()
trajectory = experimental.load_hook("trajectory")
```

| Area | Modules |
|------|---------|
| Pre-action simulation | `trajectory`, `trace_safe`, `causal_verifier`, `irreversibility` |
| Semantic consistency | `symbolic_chain`, `ontology`, `multimodal_guard`, `knowledge_graph` |
| Self-monitoring and adaptation | `meta_guard`, `self_evolving`, `defense_genome`, `continual_adversarial`, `autopoietic` |
| Swarm and organisation-level checks | `swarm_equilibrium`, `emergence_oracle`, `multi_scale_alignment`, `swarm_economics` |
| Provenance and privacy | `agent_identity`, `provenance`, `formal_verification`, `federated_privacy`, `zk_attestation` research backends |
| Resource policy | `sustainability` |

## Responsibility Consolidation Map

This map is the public ownership boundary for overlapping safety modules. It
does not remove imports or change runtime behaviour. It tells contributors
which module should own new APIs, and which surfaces should be folded into that
owner before they are promoted out of the research layer.

| Named responsibility | Canonical owner | Overlapping modules | Deprecation direction |
|----------------------|-----------------|---------------------|-----------------------|
| Runtime halt and stream interlock | `runtime.kernel`, `runtime.streaming`, `runtime.async_streaming` | `kernel.py` compatibility aliases, trace-safe stop decisions | Keep `HaltMonitor`/`StreamingKernel` as the only default halt API. Other modules may emit risk signals, not independent halt contracts. |
| Halt evidence and trace attribution | `types.HaltEvidence`, `observability`, future `SafetyEvent` schema | ad hoc evidence fields in streaming, containment, attestation, trajectory, ontology | Fold all hook-specific explanations into `SafetyEvent` and attach it to `HaltEvidence`; do not add new per-hook evidence formats. |
| Counterfactual diagnostics | `causal_verifier`, `types.CounterfactualHaltDiagnostic` | trace-safe oracle diagnostics, CoherenceAgent counterfactual helpers, ontology contradiction explanations | Keep graph/intervention logic in `causal_verifier`; other modules should call it or emit inputs for it. |
| Adaptive defence updates | `continual_adversarial` plus `defense_genome` registry | `self_evolving`, `meta_guard`, `autopoietic` | Treat `meta_guard` as monitor-only and `self_evolving` as a gated trainer. New defence changes should flow through a reviewed registry/update pipeline. |
| Provenance and signed facts | `provenance` | `agent_identity`, `zk_attestation`, KB Merkle snapshots | `provenance` owns fact/citation integrity. `agent_identity` owns public passports. `zk_attestation` remains an advanced proof backend behind those APIs. |
| Cross-org agent hand-off | `agent_identity` | `zk_attestation.passport`, provenance chains, federated privacy summaries | Public hand-off APIs should be passport-centred; direct proof-backend APIs remain advanced integration points. |
| Pre-action physical risk | `cyber_physical` | `trajectory`, `irreversibility`, `containment` | `cyber_physical` owns kinematic checks. `trajectory` and `irreversibility` provide risk inputs. `containment` owns scope anchoring, not physical simulation. |
| Multi-agent and organisation-level risk | `multi_scale_alignment` | `swarm_equilibrium`, `emergence_oracle`, `swarm_economics` | Promote a single future swarm-risk report; keep game, emergence, and economic scorers as internal contributors. |
| Multimodal factual grounding | `multimodal_guard` | `symbolic_chain`, `ontology`, knowledge-graph checks | `multimodal_guard` owns media claim extraction and verification. Symbolic and ontology modules provide optional consistency checks. |
| Resource and sustainability policy | `sustainability` | routing budgets, review queue backpressure, tenant quotas | Keep carbon, cost, and long-horizon budget policy in `sustainability`; operational throttles report into it. |

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

| Backend | Install | Accuracy (per-ds mean BA) | Latency |
|---------|---------|---------------------------|---------|
| Heuristic (lite) | core | ~65% | <0.5 ms |
| DeBERTa | `[nli]` | 75.6% | 197 ms (CPU), 19 ms (GPU batch) |
| FactCG (ONNX) | `[nli,onnx]` | 75.6% (77.76% tuned) | 14.6 ms (GPU batch) |
| Rust (backfire) | `[rust]` | ~65% | ~1 ms |
| Hybrid (NLI+Judge) | `[nli,openai]` | ~78% | 200-500 ms |

## Build Targets

| Target | Command |
|--------|---------|
| Python package | `pip install -e ".[dev]"` |
| Rust backend | `cd backfire-kernel && cargo build --release` |
| Go gateway | `cd gateway/go && go build ./cmd/director-gateway` |
| Julia tuner deps | `make julia-instantiate` |
| Lean proofs | `cd formal/HaltMonitor && lake build` |
| Tests (Python) | `pytest tests/ -v` |
| Tests (Rust) | `cd backfire-kernel && cargo test --workspace` |
| Tests (Go) | `cd gateway/go && go test ./...` |
| Tests (Julia) | `make test-julia` |
| Tests (Lean) | `make test-lean` |
| Tests (everything) | `make test-all` |
| Regenerate proto stubs | `make proto` |
| gRPC scoring server | `make grpc-scoring` |
| Docs | `mkdocs serve` |
| Benchmarks | `python -m benchmarks.run_all` |
| A/B gateway bench | `make ab-bench` |
