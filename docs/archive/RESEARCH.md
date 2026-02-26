# Research Extensions

> **Experimental.** These modules implement physics-inspired coherence dynamics
> from an academic research programme (SCPN — Self-Consistent Phenomenological
> Network). They are not required for production use and are not part of the
> stable consumer API. Interfaces may change between minor versions.

Install with:

```bash
pip install director-ai[research]
```

## What's Here

The `research/` package contains three module groups:

### physics/ — Phase dynamics and stability

Kuramoto-type oscillator models with Lyapunov stability analysis. These explore
whether phase-coherence dynamics can serve as an alternative coherence metric
alongside NLI/RAG scoring.

| Module | Purpose |
|--------|---------|
| `scpn_params.py` | Canonical 16-oscillator frequency set and coupling matrix |
| `sec_functional.py` | Lyapunov stability functional (V >= 0, dV/dt <= 0) |
| `l16_mechanistic.py` | Euler-Maruyama UPDE integrator with oversight loop |
| `l16_closure.py` | PI controllers with anti-windup, PLV gating, refusal rules |
| `gpu_upde.py` | CuPy/NumPy auto-dispatch UPDE solver |
| `lyapunov_proof.py` | Symbolic + numerical Lyapunov stability verification |
| `ssgf_cycle.py` | Geometry learning cycle (latent z → weight matrix W → spectral observables) |

### consciousness/ — Topological observables

Persistent homology applied to multichannel phase data. Extracts a scalar
observable (p_h1) from H1 cycles in delay-embedded signals.

| Module | Purpose |
|--------|---------|
| `tcbo.py` | Observer (delay embedding → ripser → p_h1) + PI controller for gap-junction coupling |
| `pgbo.py` | Phase→Geometry Bridge Operator (covariant drive u_mu → rank-2 tensor h_munu) |
| `benchmark.py` | 4 verification benchmarks (kappa increase, anesthesia, PI recovery, PGBO properties) |

### consilium/ — Ethical functional optimizer

Active inference agent that observes system state (errors, test failures,
complexity, coverage) and selects actions to minimise an ethical cost functional.

| Module | Purpose |
|--------|---------|
| `director_core.py` | EthicalFunctional (suffering/coherence/diversity tradeoff) + ConsiliumAgent (OODA loop) |

## Quick Start

```python
import numpy as np
from director_ai.research.physics import SECFunctional, build_knm_matrix

knm = build_knm_matrix()  # 16x16 coupling matrix
sec = SECFunctional(knm=knm)
theta = np.random.uniform(0, 2 * np.pi, 16)
result = sec.evaluate(theta)
print(f"V = {result.V:.4f}, stable = {sec.is_stable(result)}")
```

```python
import numpy as np
from director_ai.research.consciousness import TCBOObserver, PGBOEngine

observer = TCBOObserver(N=16)
for _ in range(60):
    phases = np.random.uniform(0, 2 * np.pi, 16)
    p_h1 = observer.push_and_compute(phases)
print(f"p_h1 = {p_h1:.3f}")

pgbo = PGBOEngine(N=16)
u_mu, h_munu = pgbo.compute(phases, dt=0.01)
```

## Connection to Other Projects

These modules share parameter sets and coupling matrices with:

- [SCPN-CODEBASE](https://github.com/anulum/scpn-fusion-core) — full simulation engine
- [sc-neurocore](https://github.com/anulum/sc-neurocore) — neuromorphic hardware (HDL)
- CCW Standalone — audio generation pipeline (uses UPDE → audio mapping)

The Rust `backfire-kernel/` workspace reimplements hot paths for real-time use
(100-10,000x speedup over Python).

## Conventions

- Uppercase variable names (K, V, N, R) follow physics notation — the linter
  has per-file ignores for `N803`/`N806` in research code.
- Tests use `@pytest.mark.physics` or `@pytest.mark.consciousness` markers.
- Research extras require `scipy`, `ripser`, and `sympy`.

## Further Reading

Detailed specifications are in the `docs/` subdirectory:

- `ARCHITECTURE_STRANGE_LOOP.md` — recursive feedback design
- `TECHNICAL_SPEC_V2.md` — coherence formula derivation
- `BACKFIRE_PREVENTION_PROTOCOLS.md` — safety interlock design
- `ROADMAP_2026_2027.md` — development plan

The theoretical background is described in the SCPN paper series
(see `01_MANUSCRIPTS/SCPN_PAPERS/` in the parent repository).
