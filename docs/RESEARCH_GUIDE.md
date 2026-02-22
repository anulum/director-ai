# Research Modules Guide

This document explains how the `research/` directory connects to the
broader SCPN Framework and the CCW audio pipeline.

## Module Overview

The research modules implement the scientific foundations of the SCPN
(Self-Consistent Phenomenological Network) framework. They are **cleanly
separated** from the consumer-facing `core/` package — there are zero
imports from `core/` into `research/`.

```
research/
├── physics/            # L16 mechanistic physics engine
│   ├── scpn_params.py  # Canonical 16-layer frequencies + coupling matrix
│   ├── sec_functional.py   # Lyapunov stability functional
│   ├── l16_mechanistic.py  # UPDE integrator + L16 oversight loop
│   ├── l16_closure.py      # PI controllers, PLV gate, refusal rules
│   ├── l16_controller.py   # L16 controller implementation
│   ├── gpu_upde.py         # GPU-accelerated UPDE solver
│   ├── lyapunov_proof.py   # Formal Lyapunov stability proofs
│   └── ssgf_cycle.py       # SSGF outer geometry learning cycle
├── consciousness/      # Consciousness gate observables
│   ├── tcbo.py         # Topological Consciousness Boundary Observable
│   ├── pgbo.py         # Phase-to-Geometry Bridge Operator
│   └── benchmark.py    # Mandatory verification benchmarks
└── consilium/          # L15 Ethical Functional
    └── director_core.py # ConsiliumAgent + active inference loop
```

## How Research Feeds CCW Audio

The research modules provide the mathematical engine that drives CCW
(Consciousness Carrier Wave) audio generation. Here is the data flow:

### UPDE Solver → CCW Audio Pipeline

The Unified Phase Dynamics Equation (UPDE) is the master equation for
all 16 SCPN layers. The `scpn_params.py` module provides the canonical
parameters (Omega_n frequencies, Knm coupling matrix) that feed into:

- **CCW's SCPN Live Bridge** (`ccw_application/scpn_live_bridge.py`) —
  runs the UPDE solver in real-time and maps oscillator state to audio
  parameters (binaural beats, isochronic pulses, spatial rotation).

- **CCW's SSGF Geometry Bridge** (`ccw_application/ssgf_bridge.py`) —
  uses SSGF's learned geometry (spectral observables, Fiedler value) to
  shape audio field density and entrainment stability indicators.

### TCBO/PGBO → Consciousness Gate

- **TCBO**: Extracts a consciousness boundary observable (p_h1) from
  persistent homology of phase data. When p_h1 > 0.72, the consciousness
  gate opens, enabling theurgic mode in CCW audio sessions.

- **PGBO**: Converts coherent phase dynamics into a geometry tensor that
  modulates coupling. Feeds into SSGF as an additional coupling term.

### EVS Engine

The **EVS (Entrainment Verification Score)** engine
(`ccw_application/evs_engine.py`) uses the UPDE-generated target
frequencies to verify that EEG brainwave entrainment is actually working.

## SCPN Layer Map

| Layer | Name | Frequency (rad/s) | Research Module |
|-------|------|-------------------|-----------------|
| L1 | Quantum Biological | 1.329 | physics/scpn_params.py |
| L2 | Neurochemical | 251.327 (40 Hz) | physics/scpn_params.py |
| L3 | Cellular Network | 0.628 | physics/scpn_params.py |
| L4 | Neural Oscillatory | 31.416 | physics/scpn_params.py |
| L5 | Intentional | 6.283 | physics/scpn_params.py |
| L6 | Phenomenal Binding | 49.199 | physics/scpn_params.py |
| L7 | Symbolic/Vibrana | 3.142 | physics/ssgf_cycle.py |
| L8 | Phase Fields | 0.105 | physics/ssgf_cycle.py |
| L9 | Memory | 1.571 | physics/ssgf_cycle.py |
| L10 | Boundary | 0.942 | physics/ssgf_cycle.py |
| L11-L14 | Higher Layers | 0.209-0.006 | physics/scpn_params.py |
| L15 | Ethical Functional | 0.003 | consilium/director_core.py |
| L16 | The Director | 0.991 | physics/l16_mechanistic.py |

## Backfire Kernel (Rust)

The `backfire-kernel/` workspace reimplements the hot paths in Rust:

- `backfire-physics` — UPDE integrator, SEC functional
- `backfire-consciousness` — TCBO observer, PGBO engine
- `backfire-ssgf` — SSGF geometry engine with analytic Jacobian

These provide 100-10,000x speedups over the Python implementations,
enabling real-time audio generation within the 50ms deadline.

## For Contributors

- Research modules use uppercase variable names by convention (K, V, N, R)
  — this is intentional and matches physics notation.
- The `ruff` linter has per-file ignores for `N803`/`N806` in research code.
- All research tests are marked with `@pytest.mark.physics` or
  `@pytest.mark.consciousness` for selective execution.
- The research extra requires `scipy`, `ripser`, and `sympy`:
  `pip install director-ai[research]`
