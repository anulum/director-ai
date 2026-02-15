# Changelog

All notable changes to Director-Class AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/anulum/director-ai/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/anulum/director-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/anulum/director-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/anulum/director-ai/releases/tag/v0.1.0
