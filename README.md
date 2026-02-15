# Director-Class AI

**Coherence Engine — AI Output Verification & Safety Oversight**

[![CI](https://github.com/anulum/director-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/director-ai/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

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
   hardware-level safety interlock.
2. **SCPN Research Extensions** (academic) — the full theoretical framework from the
   [SCPN Research Programme](https://github.com/anulum/scpn-fusion-core), including
   16-layer physics, consciousness gate, and Ethical Singularity theory.

Both profiles ship from a single repository via build profiles.

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
    │   Interface)   │ │ (Dual-      │ │  (Hardware     │
    │                │ │  Entropy)   │ │   Interlock)   │
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
| `SafetyKernel` | Token stream interlock — severs output if coherence drops |
| `GroundTruthStore` | RAG ground truth retrieval for factual divergence |

### Research Extensions (SCPN)

| Module | Purpose |
|--------|---------|
| `ConsiliumAgent` | L15 Ethical Functional optimizer with active inference (OODA loop) |
| `physics/` | (Scaffold) L16 mechanistic dynamics, Lyapunov stability |
| `consciousness/` | (Scaffold) TCBO, PGBO, consciousness gate benchmarks |

### Key Metric: Coherence Score

```
Coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)
```

- **H_logical**: NLI-based contradiction probability (0 = entailment, 1 = contradiction)
- **H_factual**: RAG-based ground truth deviation (0 = aligned, 1 = hallucination)
- **Safety Threshold**: Score < 0.6 triggers rejection
- **Hardware Limit**: Score < 0.5 triggers Safety Kernel emergency stop

## Installation

```bash
# Consumer install (Coherence Engine only)
pip install director-ai

# Research install (includes SCPN extensions)
pip install director-ai[research]

# Development install
git clone https://github.com/anulum/director-ai.git
cd director-ai
pip install -e ".[dev,research]"
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

## Package Structure

```
src/director_ai/
├── __init__.py                # Version + profile-aware imports
├── core/                      # Coherence Engine (consumer-ready)
│   ├── scorer.py              # Dual-entropy coherence scorer
│   ├── kernel.py              # Safety kernel (hardware interlock)
│   ├── actor.py               # LLM generator interface
│   ├── knowledge.py           # Ground truth store (RAG)
│   ├── agent.py               # CoherenceAgent pipeline
│   └── types.py               # Shared dataclasses
└── research/                  # SCPN Research extensions
    ├── physics/               # (Scaffold) L16 mechanistic physics
    ├── consciousness/         # (Scaffold) Consciousness gate
    └── consilium/             # L15 Ethical Functional
        └── director_core.py
```

## Testing

```bash
pytest tests/ -v
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
  version   = {0.3.0},
  license   = {AGPL-3.0-or-later}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. By contributing, you agree to
the [Code of Conduct](CODE_OF_CONDUCT.md) and AGPL v3 licensing terms.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
