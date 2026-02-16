# Changelog

All notable changes to Director-Class AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-02-16

### Added
- **Configuration Manager** (`core/config.py`):
  - `DirectorConfig` dataclass with `from_env()`, `from_yaml()`, `from_profile()` factory methods
  - Three built-in profiles: `fast` (no NLI, 1 candidate), `thorough` (NLI, 3 candidates), `research` (NLI, 5 candidates)
  - Environment variable loading with `DIRECTOR_` prefix (e.g. `DIRECTOR_USE_NLI=true`)
  - YAML/JSON file loading with PyYAML fallback
  - `to_dict()` with API key redaction
- **Metrics & Observability** (`core/metrics.py`):
  - Thread-safe `MetricsCollector` with counters, histograms, and gauges
  - Pre-registered metrics: `reviews_total`, `coherence_score`, `review_duration_seconds`, etc.
  - `timer()` context manager for latency tracking
  - `prometheus_format()` for Prometheus text exposition
  - Module-level singleton: `from director_ai.core.metrics import metrics`
- **Request Batching** (`core/batch.py`):
  - `BatchProcessor` with sync (`process_batch`) and async (`process_batch_async`) modes
  - Thread pool executor with configurable `max_concurrency`
  - `review_batch()` for bulk (prompt, response) scoring
  - `BatchResult` dataclass with success/failure counts and duration
- **LLM Provider Adapters** (`integrations/providers.py`):
  - `LLMProvider` ABC with unified `generate_candidates(prompt, n)` interface
  - `OpenAIProvider` — ChatCompletion API (supports Azure via `base_url`)
  - `AnthropicProvider` — Messages API with content block extraction
  - `HuggingFaceProvider` — Inference API adapter
  - `LocalProvider` — OpenAI-compatible local servers (llama.cpp, vLLM, Ollama)
- **FastAPI Server** (`server.py`):
  - `create_app(config)` factory with lifespan-managed state
  - REST: `/v1/health`, `/v1/review`, `/v1/process`, `/v1/batch`, `/v1/metrics`, `/v1/config`
  - `/v1/metrics/prometheus` — Prometheus-compatible scrape endpoint
  - WebSocket: `/v1/stream` — real-time coherence streaming
  - CORS middleware, Pydantic request/response models
- **CLI Tool** (`cli.py`):
  - `director-ai version` — show version
  - `director-ai review <prompt> <response>` — score a single pair
  - `director-ai process <prompt>` — run full pipeline
  - `director-ai batch <file.jsonl> [--output results.jsonl]` — bulk processing
  - `director-ai serve [--port N] [--host H] [--profile P]` — start API server
  - `director-ai config [--profile P]` — show/set configuration
- **Docker Support**:
  - Multi-stage `Dockerfile` (python:3.11-slim builder + runtime)
  - `docker-compose.yml` with optional ChromaDB sidecar (`--profile full`)
  - `.dockerignore` for lean builds
- New optional dependency group: `pip install director-ai[server]` for FastAPI + uvicorn
- CLI entry point: `director-ai` command registered via `[project.scripts]`
- 6 new test files: `test_config.py`, `test_metrics.py`, `test_batch.py`, `test_providers.py`, `test_server.py`, `test_cli.py`

### Changed
- Version bump: 0.5.0 → 0.6.0
- `core/__init__.py` exports: added `DirectorConfig`, `MetricsCollector`, `metrics`, `BatchProcessor`, `BatchResult`

## [0.5.0] - 2026-02-16

### Added
- **Async Streaming Kernel** (`core/async_streaming.py`):
  - `AsyncStreamingKernel` — async/await version of `StreamingKernel` for WebSocket production
  - Async generator `stream_tokens()` yielding `TokenEvent` objects
  - `stream_to_session()` convenience wrapper returning `StreamSession`
  - Supports both sync and async coherence callbacks
  - Same 3 halt mechanisms as sync version (hard limit, sliding window, trend)
- **GPU-Accelerated UPDE** (`research/physics/gpu_upde.py`):
  - `TorchUPDEStepper` — drop-in replacement for `UPDEStepper` using PyTorch
  - Auto device resolution: CUDA → MPS → CPU fallback
  - `step_n()` batches N steps entirely on GPU (single host↔device transfer)
  - `TorchUPDEConfig` dataclass with device="auto"
  - NumPy CPU fallback mirrors `UPDEStepper` exactly
- **Integration Tests**:
  - `test_nli_integration.py` — 8 real DeBERTa model tests (with `@slow` marker)
  - `test_chroma_integration.py` — 10 ChromaDB in-memory fixture tests
- **Jupyter Notebook Demos** (`notebooks/`):
  - `01_coherence_engine.ipynb` — Core consumer API walkthrough
  - `02_streaming_oversight.ipynb` — Sync + async streaming demo
  - `03_vector_store.ipynb` — VectorGroundTruthStore with InMemoryBackend
  - `04_physics_bridge.ipynb` — PhysicsBackedScorer L16 integration
  - `05_ssgf_geometry.ipynb` — SSGF geometry learning cycle
  - `06_lyapunov_proofs.ipynb` — Symbolic + numerical stability proofs
- **Sphinx → GitHub Pages** (`.github/workflows/docs.yml`):
  - Auto-deploy on push to main when docs/ or src/ change
  - `sphinx-build -W --keep-going` strict mode
- **PyPI Release Pipeline** (`.github/workflows/publish.yml`):
  - test.pypi.org staging job before production publish
  - Trusted publisher (OIDC) for both TestPyPI and PyPI
- New optional dependency group: `pip install director-ai[gpu]` for PyTorch

### Changed
- Version bump: 0.4.0 → 0.5.0
- `MANIFEST.in` now includes `notebooks/*.ipynb`

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
  - `SafetyKernel` — hardware-level output interlock
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
- `BackfireKernel` hardware interlock simulation
- `KnowledgeBase` RAG mock with SCPN ground truth facts
- Test suite: Pinocchio test, RAG integration test
- Demo script for end-to-end flow validation
- Documentation: Manifesto, Architecture, Roadmap, Technical Spec, API Reference

[Unreleased]: https://github.com/anulum/director-ai/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/anulum/director-ai/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/anulum/director-ai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/anulum/director-ai/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/anulum/director-ai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/anulum/director-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/anulum/director-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/anulum/director-ai/releases/tag/v0.1.0
