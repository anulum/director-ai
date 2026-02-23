# Changelog

All notable changes to Director-Class AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.1] - 2026-02-23

### Added
- **Semantic off-topic gate**: `_semantic_divergence()` in `CoherenceScorer` uses
  sentence-transformers cosine similarity to detect off-topic responses before
  falling through to string-match for value accuracy
- **Directory ingestion**: `VectorGroundTruthStore.ingest_from_directory(path, glob)`
  recursively reads `.txt`, `.md`, and `.jsonl` files from a directory

### Changed
- `ChromaBackend` now requires `sentence-transformers` (raises `ImportError` instead
  of silent fallback to Chroma's default embedding)
- Factual scoring cascade: NLI → semantic gate (off-topic detection) → string match
- README benchmark section updated with honest status and run instructions
- Research section condensed to link to `docs/RESEARCH_GUIDE.md`

## [0.8.0] - 2026-02-22

### Added
- **Backfire Kernel** (`backfire-kernel/`) — Rust workspace implementing the safety gate as a native
  performance kernel with Python FFI via PyO3:
  - `backfire-types` — shared type definitions (402 LOC, 10 tests)
  - `backfire-core` — safety gate hot path: `SafetyKernel`, `StreamingKernel`, coherence scoring
    (1,087 LOC, 33 tests, 7 benchmarks)
  - `backfire-physics` — UPDE Euler-Maruyama integrator, SEC Lyapunov functional
    (1,279 LOC, 35 tests)
  - `backfire-consciousness` — TCBO persistent homology observer, PGBO bridge operator
    (971 LOC, 29 tests)
  - `backfire-ssgf` — Stochastic Synthesis of Geometric Fields engine: gram-softplus/RBF decoders,
    Kuramoto microcycle, Jacobi eigensolver, 8 cost terms, L16 closure, outer-cycle orchestrator
    (1,750 LOC, 46 tests, 22 benchmarks)
  - `backfire-ffi` — PyO3 Python bindings exposing 13 classes (940 LOC)
  - **Total**: ~6,429 Rust LOC, 153 tests, 29 Criterion benchmarks
- **Analytic Jacobian gradient** for SSGF geometry engine:
  - `GradientMethod` enum: `FiniteDifference` (legacy) vs `Analytic` (new default)
  - **7,609x speedup** on gradient computation (31.5 ms → 4.14 µs)
  - **133x speedup** on full outer-cycle step (34.6 ms → 261 µs)
- **Phase 4 hardening** (H47-H64): subprocess path validation, batch per-line size limit,
  WebSocket error handling, NaN/Inf clamp logging, null prompt guard, timeout validation,
  LLM error type distinction, CORS origins limit, return type annotations, help text limits,
  bool coercion edge cases
- **Bulk document ingestion**: `VectorGroundTruthStore.ingest(texts, metadatas)` for batch loading
- **CLI `ingest` command**: `director-ai ingest <file>` supporting `.txt` and `.jsonl` formats
  with optional `--persist <dir>` for ChromaDB persistence
- **LLM provider streaming**: `OpenAIProvider` and `LocalProvider` support SSE streaming via
  `stream_generate()` method; `CoherenceAgent.process_streaming()` wired to real providers
- **LangChain integration**: `CoherenceCallbackHandler` in `director_ai.integrations.langchain_callback`
  (requires `pip install director-ai[langchain]`)
- **Benchmark scaffolds**: `benchmarks/truthfulqa_eval.py` and `benchmarks/halueval_eval.py`
  for public evaluation on TruthfulQA and HaluEval datasets
- `SAMPLE_FACTS` constant exported from `director_ai.core` for explicit fact injection in tests

### Changed
- **BREAKING**: `GroundTruthStore` no longer ships hardcoded facts; pass `facts=` to constructor
  or use `SAMPLE_FACTS` for the original demo data
- `ChromaBackend` now uses `SentenceTransformerEmbeddingFunction` (all-MiniLM-L6-v2) for real
  semantic retrieval
- `process_streaming()` coherence callback now scores accumulated text (not individual tokens)
- `CoherenceAgent` defaults to empty `VectorGroundTruthStore` (no auto-indexed demo facts)
- Server no longer instantiates `GroundTruthStore` with hardcoded facts
- README rewritten as consumer-focused "drop-in coherence guardrail" pitch; SCPN research
  moved to collapsible `<details>` section

### Fixed
- mypy: incompatible type assignment in CLI `ingest` command (use `VectorBackend` base type)
- mypy: missing type annotation for `prompts` in `cli.py`
- NLI `_model_score()` RuntimeError check now runs before `import torch`
- FastAPI-dependent test properly skipped when FastAPI not installed
- All ruff lint errors resolved (I001, E402, F841, N806, F401)
- black formatting aligned with CI version (black 26.x)

## [0.7.0] - 2026-02-16

### Changed
- **BREAKING**: `torch>=2.0` and `transformers>=4.30` moved from hard deps to `[nli]` optional group
  - Consumer install is now ~5MB instead of ~2GB
  - `pip install director-ai[nli]` for NLI model support
- Lazy torch import in `scorer.py` — no longer crashes without torch installed
- `CoherenceAgent` defaults to `VectorGroundTruthStore(InMemoryBackend)` instead of keyword-only `GroundTruthStore`
- All "hardware interlock" language replaced with "software safety gate" (10 files)
- README rewritten: consumer-first, accurate claims, "What This Is / Is NOT" section

### Deprecated
- `CoherenceScorer.calculate_factual_entropy()` → use `calculate_factual_divergence()`
- `CoherenceScorer.calculate_logical_entropy()` → use `calculate_logical_divergence()`
- `CoherenceScorer.simulate_future_state()` → use `compute_divergence()`
- `CoherenceScorer.review_action()` → use `review()`
- `CoherenceAgent.process_query()` → use `process()`
- All deprecated aliases emit `DeprecationWarning` and will be removed in v1.0.0

## [0.4.0] - 2026-02-16

### Added
- **NLI Scorer** (`core/nli.py`):
  - Real NLI-based logical divergence scoring using DeBERTa-v3-base-mnli
  - Lazy model loading with `@lru_cache` singleton
  - Heuristic fallback when model is unavailable
  - `score()` and `score_batch()` API
- **Vector Store** (`core/vector_store.py`):
  - Pluggable `VectorBackend` protocol (ABC)
  - `InMemoryBackend` — word-overlap cosine proxy for testing
  - `ChromaBackend` — production ChromaDB integration
  - `VectorGroundTruthStore` — semantic retrieval with keyword fallback
  - New optional dependency group: `pip install director-ai[vector]`
- **Streaming Kernel** (`core/streaming.py`):
  - Token-by-token oversight with `StreamingKernel.stream_tokens()`
  - Three halt mechanisms: hard limit, sliding window average, downward trend
  - `TokenEvent` and `StreamSession` dataclasses with full metrics
  - Backward-compatible `stream_output()` method
- **Physics Bridge** (`core/bridge.py`):
  - `PhysicsBackedScorer` blending heuristic + L16 physics coherence scores
  - Graceful degradation when research deps are absent
  - Configurable `physics_weight` and `simulation_steps`
- **SSGF Outer Cycle** (`research/physics/ssgf_cycle.py`):
  - Full SSGF geometry learning engine integrating TCBO, PGBO, and L16 closure
  - Gram-softplus decoder (z→W), spectral bridge, micro-step with geometry feedback
  - Composite cost U_total with 4 terms (c_micro, r_graph, c_tcbo, c_pgbo)
  - Finite-difference gradient update on latent geometry vector z
- **Lyapunov Stability Proof** (`research/physics/lyapunov_proof.py`):
  - Symbolic verification using SymPy: V≥0, V(θ*)=0, dV/dt≤0, K_c formula
  - Numerical verification via Monte Carlo UPDE trials
  - `run_all_proofs()` entry point, `ProofResult` dataclass
- **Cross-validation tests** (`tests/test_cross_validation.py`):
  - 14 tests validating canonical Omega_n, Knm anchors, UPDE coupling, SEC consistency
- **pytest markers**: `@pytest.mark.physics`, `@pytest.mark.consciousness`, `@pytest.mark.consumer`, `@pytest.mark.integration`
- **Code coverage**: pytest-cov integration (86.93%), CI coverage artifact upload
- **Sphinx documentation**: `docs/conf.py`, API reference pages for core and research
- **PyPI publication pipeline**: `MANIFEST.in`, `.github/workflows/publish.yml` with trusted publishing
- Test count: 44 → 144 (3.3x increase)

### Changed
- **mypy strict pass**: 0 errors across 26 source files (was advisory, now enforced in CI)
- CI workflow: coverage reporting in test job, type stubs for requests, mypy no longer soft-fail
- `EthicalFunctional.__init__` / `ConsiliumAgent.perceive` / `ConsiliumAgent.decide`: `Optional` type annotations fixed (PEP 484 compliance)
- Various numpy `no-any-return` fixes for type safety

## [0.3.1] - 2026-02-16

### Added
- **Track 1 — L16 Physics** (`research/physics/`):
  - `scpn_params.py` — Canonical Omega_n (16 frequencies) and Knm coupling matrix builder
  - `sec_functional.py` — SEC as a Lyapunov functional with stability proof, critical coupling estimate
  - `l16_mechanistic.py` — UPDE Euler-Maruyama integrator with L16 oversight loop
  - `l16_closure.py` — PI controllers with anti-windup, H_rec Lyapunov candidate, PLV gate, refusal rules
- **Track 2 — Consciousness Gate** (`research/consciousness/`):
  - `tcbo.py` — Topological Consciousness Boundary Observable (delay embedding + persistent homology → p_h1)
  - `tcbo.py` — TCBOController (PI feedback adjusting gap-junction kappa)
  - `pgbo.py` — Phase→Geometry Bridge Operator (covariant drive u_mu → rank-2 tensor h_munu)
  - `benchmark.py` — 4 mandatory verification benchmarks (kappa increase, anesthesia, PI recovery, PGBO properties)
- **Consumer enhancements** (Track 3):
  - `tests/conftest.py` — shared pytest fixtures for agent, scorer, kernel, store, generator
  - `tests/test_consumer_api.py` — 21 tests covering full consumer API surface
  - `tests/test_research_imports.py` — 17 tests for physics + consciousness module imports and smoke tests
- Test count: 6 → 44 (7.3× increase)

## [0.3.0] - 2026-02-15

### Added
- **Dual-trajectory architecture**: single repo with consumer (`core/`) and research (`research/`) profiles
- `director_ai.core` package — Coherence Engine (consumer-ready, no SCPN vocabulary)
  - `CoherenceScorer` — dual-entropy scorer (NLI + RAG)
  - `SafetyKernel` — software safety gate
  - `MockGenerator` / `LLMGenerator` — LLM candidate generation
  - `GroundTruthStore` — RAG ground truth retrieval
  - `CoherenceAgent` — orchestrator pipeline with `process()` returning `ReviewResult`
  - `CoherenceScore` / `ReviewResult` dataclasses in `types.py`
- `director_ai.research` package — SCPN Research Extensions (requires `[research]` extras)
  - `research.consilium` — L15 Ethical Functional (moved from `src/consilium/`)
  - `research.physics` — scaffold for L16 mechanistic physics (Track 1)
  - `research.consciousness` — scaffold for TCBO/PGBO (Track 2)
- `pyproject.toml` optional dependencies: `[research]` for scipy, `[dev]` for tooling
- `requests>=2.28` added to core dependencies (for `LLMGenerator`)
- Backward-compatible aliases on all renamed classes/methods

### Changed
- Package structure: flat `src/` → proper `src/director_ai/` Python package
- `DirectorModule` → `CoherenceScorer` (consumer vocabulary)
- `BackfireKernel` → `SafetyKernel`
- `MockActor` → `MockGenerator`, `RealActor` → `LLMGenerator`
- `KnowledgeBase` → `GroundTruthStore`
- `StrangeLoopAgent` → `CoherenceAgent`
- `test_pinocchio.py` → `test_deception_detection.py`
- All test imports updated to `from director_ai.core import ...`
- `scipy` moved from core dependencies to `[research]` optional

### Removed
- Old flat `src/` module files (replaced by `src/director_ai/` package)

## [0.2.0] - 2026-01-21

### Added
- Consilium subsystem (`src/consilium/director_core.py`)
  - `EthicalFunctional` class: mathematical definition of ethical value
  - `ConsiliumAgent` class: active inference agent with OODA loop
  - Real telemetry integration (git status, pytest execution)
  - Predictive outcome modelling for action selection
- `StrangeLoopAgent` v2 with real LLM connection support (`RealActor`)
- Advanced coherence test suite (`test_advanced_coherence.py`)

### Changed
- `StrangeLoopAgent` now supports both mock and real LLM backends
- Director module enhanced with dual-entropy (logical + factual) scoring

## [0.1.0] - 2025-12-29

### Added
- Initial prototype of Director-Class AI (Layer 16)
- `StrangeLoopAgent` orchestrating Actor, Director, and Backfire Kernel
- `DirectorModule` with SEC (Sustainable Ethical Coherence) metric
- `MockActor` simulating Layer 11 narrative engine
- `BackfireKernel` safety gate simulation
- `KnowledgeBase` RAG mock with SCPN ground truth facts
- Test suite: Pinocchio test, RAG integration test
- Demo script for end-to-end flow validation
- Documentation: Manifesto, Architecture, Roadmap, Technical Spec, API Reference

[Unreleased]: https://github.com/anulum/director-ai/compare/v0.8.1...HEAD
[0.8.1]: https://github.com/anulum/director-ai/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/anulum/director-ai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/anulum/director-ai/compare/v0.4.0...v0.7.0
[0.4.0]: https://github.com/anulum/director-ai/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/anulum/director-ai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/anulum/director-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/anulum/director-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/anulum/director-ai/releases/tag/v0.1.0
