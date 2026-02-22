# Backfire Kernel — Benchmark Report

**Date:** 2026-02-22
**Platform:** Windows 11 Pro (10.0.26200), AMD64
**Rust:** 1.93.0 | **Profile:** release (opt-level=3, lto=fat, codegen-units=1)
**Framework:** Criterion 0.5

---

## Deadline Requirement

All safety-critical operations must complete within **50 ms**
(Backfire Prevention Protocols &sect;2.2).

---

## 1. backfire-core — Safety Gate Hot Path

**Source:** `crates/backfire-core/benches/scoring_bench.rs`
**7 benchmarks** covering the coherence scoring and token-gating pipeline.

| Benchmark | Median | Budget Used | Headroom |
|-----------|--------|-------------|----------|
| `scorer_review` | 2.50 &micro;s | 0.005% | 19,920&times; |
| `safety_kernel_10tok` | 265 ns | 0.001% | 188,679&times; |
| `safety_kernel_100tok` | 1.10 &micro;s | 0.002% | 45,455&times; |
| `streaming_kernel_10tok` | 3.97 &micro;s | 0.008% | 12,594&times; |
| `streaming_kernel_100tok` | 34.5 &micro;s | 0.069% | 1,449&times; |
| `streaming_kernel_1000tok` | 332 &micro;s | 0.664% | 151&times; |
| `full_pipeline_10tok` | 4.56 &micro;s | 0.009% | 10,965&times; |

### Scaling Behaviour

| Tokens | Streaming Kernel | Per-Token |
|--------|-----------------|-----------|
| 10 | 3.97 &micro;s | 397 ns |
| 100 | 34.5 &micro;s | 345 ns |
| 1,000 | 332 &micro;s | 332 ns |

Linear scaling with slight per-token cost decrease due to amortised setup.

---

## 2. backfire-ssgf — SSGF Geometry Engine

**Source:** `crates/backfire-ssgf/benches/ssgf_bench.rs`
**22 benchmarks** covering every hot-path component of the Stochastic Synthesis
of Geometric Fields engine (N=16 oscillators, 120-dimensional latent space).

### 2.1 Decoders (z &rarr; W)

| Benchmark | Median | Budget Used |
|-----------|--------|-------------|
| `decode_gram_softplus_16x16` | 7.93 &micro;s | 0.016% |
| `decode_rbf_16x16_d8` | 4.01 &micro;s | 0.008% |

Both decoders guarantee W &ge; 0, W = W&sup1;, diag(W) = 0.
Gram-softplus involves softplus on 256 elements; RBF computes 120 pairwise
distances across 8 dimensions.

### 2.2 Micro-Cycle Engine (Kuramoto + Geometry + PGBO)

| Benchmark | Median | Budget Used |
|-----------|--------|-------------|
| `micro_single_step_16` | 4.86 &micro;s | 0.010% |
| `micro_10_steps_16` | 50.3 &micro;s | 0.101% |
| `micro_10_steps_16_pgbo` | 42.7 &micro;s | 0.085% |
| `order_parameter_16` | 459 ns | 0.001% |

Each micro-step computes N&sup2; coupling terms (Kuramoto + geometry feedback
+ optional PGBO tensor), plus noise sampling and modular phase reduction.
The PGBO variant is slightly faster due to cache effects from the additional
matrix read.

### 2.3 Spectral Bridge (Laplacian + Jacobi Eigensolver)

| Benchmark | Median | Budget Used |
|-----------|--------|-------------|
| `spectral_laplacian_16x16` | 889 ns | 0.002% |
| `spectral_eigenpairs_16x16` | 107 &micro;s | 0.214% |

The Jacobi eigensolver is a pure-Rust cyclic implementation (no LAPACK)
with Rutishauser-form rotations and threshold strategy. For N=16,
convergence requires &le;15 sweeps at O(N&sup3;) per sweep. The eigenpair
computation includes Laplacian construction, eigendecomposition, sorting,
sign-gauge fixing, and negative-eigenvalue clamping.

### 2.4 Cost Terms

| Benchmark | Median | Budget Used |
|-----------|--------|-------------|
| `cost_micro_16` | 452 ns | 0.001% |
| `cost_c8_phase_16` | 480 ns | 0.001% |
| `cost_c10_boundary_16` | 404 ns | 0.001% |
| `regularise_graph_16` | 394 ns | 0.001% |
| `compute_costs_all_16` | 1.46 &micro;s | 0.003% |

All 8 cost terms (C_micro, C7, C8, C9, C10, R_graph, C4_tcbo, C_pgbo)
computed and weighted into U_total in under 1.5 &micro;s.

### 2.5 Gradient Computation

| Benchmark | Median | Budget Used | Method | Speedup |
|-----------|--------|-------------|--------|---------|
| `gradient_fd_120dim` | 31.5 ms | 63.1% | Finite-difference (240 passes) | &mdash; |
| `gradient_analytic_gram_120dim` | **4.14 &micro;s** | **0.008%** | Analytic Jacobian | **7,609&times;** |

**Finite-difference (legacy):** 240 forward passes (2 per latent dimension),
each requiring a full decode + spectral eigenpair + cost evaluation.

**Analytic Jacobian (new default):** Computes &part;U/&part;W from W-dependent
cost terms only (C9, C10, R_graph Frobenius), then chains through the
gram_softplus or RBF decoder Jacobian. Eigensolver-dependent terms
(C7, C8, gap penalty) use stop-gradient &mdash; eigenpairs are frozen
from step 5 of the outer cycle. Measured **7,609&times; speedup** over FD.

The `GradientMethod` enum selects between `FiniteDifference` and `Analytic`
(default) in `SSGFConfig`.

### 2.6 Engine (Full Pipeline)

| Benchmark | Median | Budget Used | Description |
|-----------|--------|-------------|-------------|
| `engine_init_16` | 166 &micro;s | 0.33% | One-time setup |
| `engine_outer_step` | 38.1 ms | 76.1% | Single outer-cycle (FD gradient) |
| `engine_outer_step_analytic` | **229 &micro;s** | **0.46%** | Single outer-cycle (analytic gradient) |
| `engine_5_outer_steps` | 173 ms | &mdash; | 5&times; outer step (FD) |
| `engine_audio_mapping` | 724 ns | 0.001% | Read-only state projection |
| **`DEADLINE_outer_step_warmed`** | **34.6 ms** | **69.2%** | **Warmed single step (FD)** |
| **`DEADLINE_outer_step_analytic_warmed`** | **261 &micro;s** | **0.52%** | **Warmed single step (analytic)** |

With the **analytic gradient** (new default), the outer-cycle step drops from
**34.6 ms to 261 &micro;s** &mdash; a **133&times; speedup** &mdash; using only
**0.52%** of the 50 ms deadline budget and leaving **99.5% headroom**.
The FD gradient path remains available via
`SSGFConfig { gradient_method: GradientMethod::FiniteDifference, .. }`.

### 2.7 Time Budget Breakdown (Single Outer Step)

#### Analytic Gradient (default)

| Stage | Measured Time | Share |
|-------|--------------|-------|
| Decode (gram_softplus) | 8 &micro;s | 3.1% |
| PGBO compute | 6 &micro;s | 2.3% |
| Micro-cycle (10 steps + PGBO) | 43 &micro;s | 16.5% |
| Spectral eigenpairs | 107 &micro;s | 41.0% |
| Cost evaluation | 1.5 &micro;s | 0.6% |
| **Gradient (analytic Jacobian)** | **4.14 &micro;s** | **1.6%** |
| L16 closure + verify + log | ~3 &micro;s | 1.1% |
| Overhead (alloc, branches) | ~88 &micro;s | 33.7% |
| **Total** | **~261 &micro;s** | **100%** |

With FD eliminated, the spectral eigensolver (Jacobi, 107 &micro;s) is now
the dominant cost at 41% of the outer step.

#### Finite-Difference Gradient (legacy)

| Stage | Estimated Time | Share |
|-------|---------------|-------|
| Decode (gram_softplus) | 8 &micro;s | 0.02% |
| PGBO compute | 6 &micro;s | 0.02% |
| Micro-cycle (10 steps + PGBO) | 43 &micro;s | 0.12% |
| Spectral eigenpairs | 107 &micro;s | 0.31% |
| Cost evaluation | 1.5 &micro;s | &lt;0.01% |
| **Gradient (120-dim FD)** | **31.5 ms** | **91.0%** |
| L16 closure + verify + log | ~3 &micro;s | &lt;0.01% |
| Overhead (alloc, branches) | ~3 ms | 8.5% |
| **Total** | **~34.6 ms** | **100%** |

---

## 3. Python FFI Benchmarks (PyO3 Crossing Overhead)

Measured via `timeit` through the `backfire_kernel` wheel (13 classes).

| Operation | Latency | Headroom |
|-----------|---------|----------|
| UPDE 1 step | 9.0 &micro;s | 5,556&times; |
| UPDE 10 steps | 60.7 &micro;s | 824&times; |
| UPDE 100 steps | 609 &micro;s | 82&times; |
| UPDE 1000 steps | 5.9 ms | 8.5&times; |
| SEC evaluate (aligned) | 5.4 &micro;s | 9,259&times; |
| SEC evaluate (spread) | 7.5 &micro;s | 6,667&times; |
| SEC critical_coupling | 234 ns | 213,675&times; |
| L16 controller step | 3.0 &micro;s | 16,667&times; |
| TCBO push_and_compute | 114 &micro;s | 439&times; |
| TCBO controller step | 582 ns | 85,911&times; |
| PGBO compute (N=16) | 5.9 &micro;s | 8,475&times; |
| SafetyKernel 10 tok | 2.2 &micro;s | 22,727&times; |
| StreamingKernel 10 tok | 6.0 &micro;s | 8,333&times; |

---

## 4. Aggregate Summary

### 4.1 Workspace Statistics

| Crate | Source LOC | Tests | Benchmarks |
|-------|-----------|-------|------------|
| backfire-types | 402 | 10 | &mdash; |
| backfire-core | 1,087 | 33 | 7 |
| backfire-physics | 1,279 | 35 | &mdash; |
| backfire-consciousness | 971 | 29 | &mdash; |
| backfire-ssgf | 1,750 | 46 | 22 |
| backfire-ffi | 940 | &mdash; | &mdash; |
| **Total** | **~6,429** | **153** | **29** |

### 4.2 Deadline Compliance

| Path | Time | Verdict |
|------|------|---------|
| Safety kernel (10 tokens) | 265 ns | PASS (188,679&times; under) |
| Full scoring pipeline (10 tokens) | 4.56 &micro;s | PASS (10,965&times; under) |
| Streaming kernel (1000 tokens) | 332 &micro;s | PASS (151&times; under) |
| SSGF outer step, FD (warmed) | 34.6 ms | PASS (1.44&times; under) |
| SSGF outer step, analytic (warmed) | 261 &micro;s | PASS (191&times; under) |
| SSGF gradient FD only | 31.5 ms | PASS (1.59&times; under) |
| SSGF gradient analytic only | 4.14 &micro;s | PASS (12,077&times; under) |

### 4.3 Performance Extremes

| | Benchmark | Median |
|---|-----------|--------|
| **Fastest** | `safety_kernel_10tok` | 265 ns |
| **Slowest** | `engine_5_outer_steps` | 173 ms |
| **Tightest (FD)** | `DEADLINE_outer_step_warmed` | 34.6 ms (69.2% of budget) |
| **Tightest (analytic)** | `DEADLINE_outer_step_analytic_warmed` | 261 &micro;s (0.52% of budget) |
| **Most headroom** | `safety_kernel_10tok` | 0.001% of budget |

---

## 5. Optimisation Opportunities

| Opportunity | Expected Impact | Effort | Status |
|-------------|----------------|--------|--------|
| ~~Analytic Jacobian for gram_softplus decoder~~ | ~~Eliminates FD entirely~~ | ~~High~~ | **DONE** (7,609&times; gradient, 133&times; outer step) |
| Replace FD gradient with autodiff (JAX/PyTorch backend) | 10&ndash;50&times; gradient speedup | Medium | Superseded by analytic |
| SIMD vectorise micro-cycle inner loop | 2&ndash;4&times; on micro-step | Low | Open |
| Cache Laplacian across gradient perturbations | ~30% gradient speedup | Low | N/A with analytic |
| Parallel gradient evaluation (rayon) | ~4&times; on 8-core | Low | N/A with analytic |

---

## Running Benchmarks

```bash
# All benchmarks (both crates)
cargo bench --workspace

# Core safety gate only
cargo bench --package backfire-core

# SSGF geometry engine only
cargo bench --package backfire-ssgf

# Single benchmark by name
cargo bench --package backfire-ssgf -- "engine_outer_step"
```

Criterion HTML reports are generated in `target/criterion/report/index.html`.
