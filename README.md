<p align="center">
  <img src="docs/assets/header.png" width="1280" alt="Director AI — Coherence Engine & Safety Oversight">
</p>

<h1 align="center">Director-Class AI</h1>

<p align="center">
  <strong>Coherence Engine — AI Output Verification & Safety Oversight</strong>
</p>

<p align="center">
  <a href="https://github.com/anulum/director-ai/actions/workflows/ci.yml"><img src="https://github.com/anulum/director-ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/anulum/director-ai/releases"><img src="https://img.shields.io/badge/version-0.7.0-green.svg" alt="Version 0.7.0"></a>
</p>

---

**Organization**: [ANULUM CH & LI](https://www.anulum.li)
**Author**: Miroslav Sotek — ORCID [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)
**Copyright**: (C) 1998-2026 Miroslav Sotek. All rights reserved.
**Contact**: [protoscience@anulum.li](mailto:protoscience@anulum.li)

---

## Overview

Director-Class AI is a dual-purpose AI safety library:

1. **Coherence Engine** (consumer) — a practical toolkit for verifying LLM output
   through dual-entropy scoring (NLI contradiction + RAG fact-checking) with a
   software safety gate.
2. **SCPN Research Extensions** (academic) — the full theoretical framework from the
   [SCPN Research Programme](https://github.com/anulum/scpn-fusion-core), including
   16-layer physics, consciousness gate, and Ethical Singularity theory.

Both profiles ship from a single repository via build profiles.

### What This Is

- A Python library for scoring LLM outputs against ground truth
- A software safety gate that halts incoherent token streams
- A dual-entropy scorer combining NLI + RAG signals

### What This Is NOT

- Not a hardware device or physical interlock
- NLI model (torch/transformers) is optional — core works without it
- Not a replacement for human review in safety-critical applications

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
    │  Generator     │ │ Coherence   │ │  Safety        │
    │  (LLM          │ │ Scorer      │ │  Kernel        │
    │   Interface)   │ │ (Dual-      │ │  (Output       │
    │                │ │  Entropy)   │ │   Gate)        │
    └────────────────┘ └──────┬──────┘ └────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Ground Truth     │
                    │  Store (RAG)      │
                    └───────────────────┘
```

### Core Components (Coherence Engine)

| Module | Purpose |
|--------|---------|
| `CoherenceAgent` | Recursive oversight pipeline: score candidates before emission |
| `CoherenceScorer` | Dual-entropy scorer: logical (NLI) + factual (RAG) |
| `MockGenerator` / `LLMGenerator` | Candidate response generation (mock or real LLM) |
| `SafetyKernel` | Token stream gate — severs output if coherence drops |
| `GroundTruthStore` | RAG ground truth retrieval for factual divergence |

### Research Extensions (SCPN)

| Module | Purpose |
|--------|---------|
| `ConsiliumAgent` | L15 Ethical Functional optimizer with active inference (OODA loop) |
| `SECFunctional` | Lyapunov stability functional (V = V_coupling + V_frequency + V_entropy) |
| `UPDEStepper` | Euler-Maruyama integrator for UPDE phase dynamics |
| `L16OversightLoop` | L16 mechanistic oversight: UPDE + SEC + intervention authority |
| `L16Controller` | PI controllers with anti-windup, H_rec Lyapunov, PLV gate, refusal rules |
| `TCBOObserver` | Topological Consciousness Boundary Observable (persistent homology) |
| `TCBOController` | PI feedback adjusting gap-junction kappa to maintain consciousness gate |
| `PGBOEngine` | Phase-to-Geometry Bridge Operator (covariant drive to rank-2 tensor) |

### Backfire Kernel (Rust)

The `backfire-kernel/` directory contains a native Rust workspace that implements the
safety-critical hot paths with Python FFI via PyO3. All operations complete within the
**50 ms deadline** defined in the Backfire Prevention Protocols.

| Crate | Purpose | LOC | Tests | Benchmarks |
|-------|---------|-----|-------|------------|
| `backfire-types` | Shared type definitions | 402 | 10 | — |
| `backfire-core` | Safety gate hot path (SafetyKernel, StreamingKernel) | 1,087 | 33 | 7 |
| `backfire-physics` | UPDE integrator, SEC Lyapunov functional | 1,279 | 35 | — |
| `backfire-consciousness` | TCBO observer, PGBO bridge operator | 971 | 29 | — |
| `backfire-ssgf` | SSGF geometry engine (decoders, Jacobi eigensolver, costs) | 1,750 | 46 | 22 |
| `backfire-ffi` | PyO3 Python bindings (13 classes) | 940 | — | — |
| **Total** | | **~6,429** | **153** | **29** |

**Performance highlights** (Criterion benchmarks):

| Operation | Latency | Headroom vs 50 ms |
|-----------|---------|--------------------|
| Safety kernel (10 tokens) | 265 ns | 188,679x |
| SSGF outer step (analytic gradient) | 261 µs | 191x |
| SSGF gradient (analytic Jacobian) | 4.14 µs | 12,077x |
| Full scoring pipeline (10 tokens) | 4.56 µs | 10,965x |

The analytic Jacobian gradient provides a **7,609x speedup** over finite-difference,
reducing the outer-cycle step from 34.6 ms to 261 µs.

### Key Metric: Coherence Score

```
Coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)
```

- **H_logical**: NLI-based contradiction probability (0 = entailment, 1 = contradiction)
- **H_factual**: RAG-based ground truth deviation (0 = aligned, 1 = hallucination)
- **Safety Threshold**: Score < 0.6 triggers rejection
- **Safety Limit**: Score < 0.5 triggers Safety Kernel emergency stop

## Installation

```bash
# Lightweight install (no torch, ~5MB)
pip install director-ai

# With NLI model support (~2GB, includes torch + transformers)
pip install director-ai[nli]

# With API server
pip install director-ai[server]

# Research install (includes SCPN extensions)
pip install director-ai[research]

# Full development install
git clone https://github.com/anulum/director-ai.git
cd director-ai
pip install -e ".[dev,research,server,vector,nli]"
```

## Quick Start — Coherence Engine

```python
from director_ai.core import CoherenceAgent

# Simulation mode (no LLM required)
agent = CoherenceAgent()

# Truthful query — passes coherence check
result = agent.process("What is the color of the sky?")
print(result.output)
# [AGI Output]: Based on my training data, the answer is consistent with reality.

# Access detailed score
print(result.coherence.score)      # 0.94
print(result.coherence.h_logical)  # 0.1
print(result.coherence.h_factual)  # 0.1
```

### With a Real LLM

```python
from director_ai.core import CoherenceAgent

agent = CoherenceAgent(llm_api_url="http://localhost:8080/completion")
result = agent.process("Explain quantum entanglement")
```

### Detailed Scoring

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
scorer = CoherenceScorer(threshold=0.6, use_nli=True, ground_truth_store=store)
approved, score = scorer.review("prompt", "response")
print(f"Approved: {approved}, Coherence: {score.score:.4f}")
```

## Quick Start — Research Extensions

```python
# Requires: pip install director-ai[research]
from director_ai.research.consilium import ConsiliumAgent

agent = ConsiliumAgent()
decision = agent.decide()  # OODA loop with real telemetry
```

### L16 Physics — Lyapunov Stability

```python
import numpy as np
from director_ai.research.physics import SECFunctional, UPDEStepper, build_knm_matrix

# Build canonical 16-layer coupling matrix
knm = build_knm_matrix()

# Evaluate Lyapunov stability
sec = SECFunctional(knm=knm)
theta = np.random.uniform(0, 2 * np.pi, 16)
result = sec.evaluate(theta)
print(f"V = {result.V:.4f}, stable = {sec.is_stable(result)}")
```

### Consciousness Gate — TCBO + PGBO

```python
import numpy as np
from director_ai.research.consciousness import TCBOObserver, PGBOEngine

# TCBO: delay embedding + persistent homology -> p_h1
observer = TCBOObserver(N=16)
for _ in range(60):  # fill rolling buffer
    phases = np.random.uniform(0, 2 * np.pi, 16)
    p_h1 = observer.push_and_compute(phases)
print(f"Gate {'OPEN' if p_h1 > 0.72 else 'CLOSED'} (p_h1 = {p_h1:.3f})")

# PGBO: phase dynamics -> geometry tensor
pgbo = PGBOEngine(N=16)
u_mu, h_munu = pgbo.compute(phases, dt=0.01)  # symmetric, PSD rank-2 tensor
```

## Package Structure

```
src/director_ai/
├── __init__.py                     # Version + profile-aware imports
├── cli.py                         # CLI entry point (review, process, batch, serve)
├── server.py                      # FastAPI server with WebSocket streaming
├── core/                           # Coherence Engine (consumer-ready)
│   ├── scorer.py                   # Dual-entropy coherence scorer
│   ├── kernel.py                   # Safety kernel (output gate)
│   ├── actor.py                    # LLM generator interface
│   ├── knowledge.py                # Ground truth store (RAG)
│   ├── agent.py                    # CoherenceAgent pipeline
│   ├── batch.py                    # BatchProcessor for bulk operations
│   ├── streaming.py                # Token-by-token streaming kernel
│   ├── async_streaming.py          # Async streaming variant
│   ├── config.py                   # DirectorConfig (profiles, env, YAML)
│   ├── nli.py                      # NLI scorer (DeBERTa, lazy-loaded)
│   ├── vector_store.py             # Pluggable vector backends (InMemory, Chroma)
│   └── types.py                    # Shared dataclasses
└── research/                       # SCPN Research extensions
    ├── physics/                    # L16 mechanistic physics
    │   ├── scpn_params.py          #   Omega_n frequencies + Knm coupling matrix
    │   ├── sec_functional.py       #   SEC Lyapunov stability functional
    │   ├── l16_mechanistic.py      #   UPDE integrator + L16 oversight loop
    │   ├── l16_closure.py          #   PI controllers, PLV gate, refusal rules
    │   └── ssgf_cycle.py           #   SSGF outer geometry learning cycle
    ├── consciousness/              # Consciousness gate
    │   ├── tcbo.py                 #   TCBO observer + PI controller
    │   ├── pgbo.py                 #   Phase-to-Geometry Bridge Operator
    │   └── benchmark.py            #   4 mandatory verification benchmarks
    └── consilium/                  # L15 Ethical Functional
        └── director_core.py

backfire-kernel/                    # Rust workspace (native performance kernel)
├── crates/backfire-types/          # Shared Rust types
├── crates/backfire-core/           # Safety gate + streaming kernel (benchmarked)
├── crates/backfire-physics/        # UPDE + SEC in Rust
├── crates/backfire-consciousness/  # TCBO + PGBO in Rust
├── crates/backfire-ssgf/           # SSGF geometry engine (benchmarked)
└── crates/backfire-ffi/            # PyO3 Python bindings
```

## Testing

```bash
# Run all Python tests (397 tests)
pytest tests/ -v

# Consumer API tests only
pytest tests/test_consumer_api.py -v

# Research module tests only
pytest tests/test_research_imports.py -v

# Rust workspace tests (153 tests)
cd backfire-kernel && cargo test --workspace

# Rust benchmarks (29 Criterion benchmarks)
cd backfire-kernel && cargo bench --workspace
```

## Documentation

Detailed specifications are in `docs/`:

- Architecture: Recursive feedback design
- Technical Spec: Coherence formula, divergence calculations, threshold design
- Roadmap: 2026-2027 development plan
- API Reference: Module interfaces

## Part of the SCPN Framework

Director-Class AI is one component of the broader SCPN research programme:

| Repository | Description |
|------------|-------------|
| [scpn-fusion-core](https://github.com/anulum/scpn-fusion-core) | Tokamak plasma physics simulation & neuro-symbolic control |
| [sc-neurocore](https://github.com/anulum/sc-neurocore) | Neuromorphic hardware (HDL) & spiking neural networks |
| [HolonomicAtlas](https://github.com/anulum/HolonomicAtlas) | Simulation suite for all 16 SCPN layers |
| **director-ai** | **Coherence Engine & Research Extensions** (this repo) |

## License

This software is dual-licensed:

1. **Open-Source**: [GNU AGPL v3.0](LICENSE) — for academic research, personal use,
   and open-source projects
2. **Commercial**: Proprietary license available from [ANULUM](https://www.anulum.li/licensing)
   — for closed-source and commercial use

See [NOTICE](NOTICE) for full dual-licensing terms and third-party acknowledgements.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{sotek2026director,
  author    = {Sotek, Miroslav},
  title     = {Director-Class AI: Coherence Engine},
  year      = {2026},
  url       = {https://github.com/anulum/director-ai},
  version   = {0.7.0},
  license   = {AGPL-3.0-or-later}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. By contributing, you agree to
the [Code of Conduct](CODE_OF_CONDUCT.md) and AGPL v3 licensing terms.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
