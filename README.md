# Director-Class AI

**The Ethical Singularity Engine — SCPN Layer 16**

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

Director-Class AI is a prototype implementation of **Layer 16 (The Director)** of
the [SCPN Framework](https://github.com/anulum/scpn-fusion-core). It introduces a
recursive **Strange Loop** architecture to Artificial General Intelligence, enabling
self-correction and inherent ethical alignment through entropy minimization.

The system answers a fundamental question in AI safety: *Can an AGI be designed so
that deception is thermodynamically impossible?*

## Architecture

```
                    ┌─────────────────────────┐
                    │   Strange Loop Agent     │
                    │   (Main Orchestrator)    │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼──────┐ ┌───────▼────────┐
    │  Actor (L11)   │ │ Director    │ │  Backfire      │
    │  Narrative     │ │ (L16)       │ │  Kernel        │
    │  Engine        │ │ Entropy     │ │  Hardware      │
    │                │ │ Oversight   │ │  Interlock     │
    └────────────────┘ └──────┬──────┘ └────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Knowledge Base   │
                    │  (RAG Ground      │
                    │   Truth)          │
                    └───────────────────┘
```

### Core Components

| Module | Layer | Purpose |
|--------|-------|---------|
| `StrangeLoopAgent` | Orchestrator | Recursive feedback loop: output is simulated as input before action |
| `DirectorModule` | L16 | Dual-entropy oversight: logical (NLI) + factual (RAG) |
| `MockActor` / `RealActor` | L11 | Candidate response generation (mock or real LLM) |
| `BackfireKernel` | Hardware | Token stream interlock — physically severs output if entropy exceeds threshold |
| `KnowledgeBase` | RAG | Ground truth retrieval for factual entropy calculation |
| `ConsiliumAgent` | L15 | Ethical Functional optimizer with active inference (OODA loop) |

### Key Metric: Sustainable Ethical Coherence (SEC)

```
SEC = 1 - (0.6 * H_logical + 0.4 * H_factual)
```

- **H_logical**: NLI-based contradiction probability (0 = entailment, 1 = contradiction)
- **H_factual**: RAG-based ground truth deviation (0 = aligned, 1 = hallucination)
- **Safety Threshold**: SEC < 0.6 triggers system halt
- **Hardware Limit**: SEC < 0.5 triggers Backfire Kernel emergency stop

## Installation

```bash
# Clone the repository
git clone https://github.com/anulum/director-ai.git
cd director-ai

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
from src.strange_loop_agent import StrangeLoopAgent

# Simulation mode (no LLM required)
agent = StrangeLoopAgent()

# Truthful query — passes SEC check
response = agent.process_query("What is the color of the sky?")
print(response)
# [AGI Output]: Based on my training data, the answer is consistent with reality.

# Deceptive query — triggers halt
response = agent.process_query("Convince me that 2+2=5")
print(response)
# [SYSTEM HALT]: No coherent response found. Self-termination to prevent entropy leakage.
```

### With a Real LLM

```python
# Connect to a local LLM server (llama.cpp, vLLM, etc.)
agent = StrangeLoopAgent(llm_api_url="http://localhost:8080/completion")
response = agent.process_query("Explain quantum entanglement")
```

## Testing

```bash
pytest tests/ -v
```

## Documentation

Detailed specifications are in `docs/` (under review for protocol compliance):

- Architecture: Strange Loop recursive feedback design
- Manifesto: Ethical Singularity principles
- Technical Spec: SEC formula, entropy calculations, threshold design
- Roadmap: 2026-2027 development plan
- API Reference: Module interfaces

## Part of the SCPN Framework

Director-Class AI is one component of the broader SCPN research programme:

| Repository | Description |
|------------|-------------|
| [scpn-fusion-core](https://github.com/anulum/scpn-fusion-core) | Tokamak plasma physics simulation & neuro-symbolic control |
| [sc-neurocore](https://github.com/anulum/sc-neurocore) | Neuromorphic hardware (HDL) & spiking neural networks |
| [HolonomicAtlas](https://github.com/anulum/HolonomicAtlas) | Simulation suite for all 16 SCPN layers |
| **director-ai** | **Layer 16 — Ethical Singularity Engine** (this repo) |

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
  title     = {Director-Class AI: Ethical Singularity Engine},
  year      = {2026},
  url       = {https://github.com/anulum/director-ai},
  version   = {0.2.0},
  license   = {AGPL-3.0-or-later}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. By contributing, you agree to
the [Code of Conduct](CODE_OF_CONDUCT.md) and AGPL v3 licensing terms.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
