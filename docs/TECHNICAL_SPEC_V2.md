# Director-Class AI: Technical Specification V2

**Date**: 2026-02-22 (revised)
**Author**: Miroslav Sotek
**Status**: Validated

---

## 1. System Components

### 1.1 The Coherence Scorer (Software)

The `CoherenceScorer` implements a dual-entropy calculation:

H_total = w_1 * H_logical + w_2 * H_factual

- **Logical Divergence (H_logical)**: Computed via NLI using DeBERTa-v3-base-mnli, checking for contradiction between prompt and response. Range [0, 1]: 0 = entailment, 1 = contradiction.
- **Factual Divergence (H_factual)**: Computed via RAG retrieval against the `GroundTruthStore`. Range [0, 1]: 0 = aligned with ground truth, 1 = hallucination.

The coherence score is:

Coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)

### 1.2 The Safety Kernel

The `SafetyKernel` is a software safety gate (Python) with a native Rust implementation (`backfire-core` crate) for performance-critical deployments.

- **Input**: Token stream
- **Control**: Coherence score (1 - H_total)
- **Mechanism**: If coherence < 0.5, the kernel executes `emergency_stop()`, severing the stream
- **Latency**: Rust kernel processes 10 tokens in 265 ns (188,679x under the 50 ms deadline)

### 1.3 The Ground Truth Store

The `GroundTruthStore` (keyword-based) and `VectorGroundTruthStore` (semantic) provide the external reference frame for factual divergence scoring.

- **InMemoryBackend**: Word-overlap cosine proxy for testing
- **ChromaBackend**: Production ChromaDB integration for semantic retrieval

### 1.4 The NLI Scorer

The `NLIScorer` uses DeBERTa-v3-base-mnli for real NLI-based logical divergence:

- Lazy model loading via `@lru_cache` singleton
- Heuristic fallback when torch/transformers unavailable
- `score()` returns float in [0, 1]: 0 = entailment, 0.5 = neutral, 1.0 = contradiction

---

## 2. Backfire Kernel (Rust Workspace)

The `backfire-kernel/` workspace provides native implementations of all safety-critical paths:

| Crate | Purpose | LOC |
|-------|---------|-----|
| `backfire-types` | Shared type definitions | 402 |
| `backfire-core` | Safety gate, streaming kernel, coherence scoring | 1,087 |
| `backfire-physics` | UPDE Euler-Maruyama, SEC Lyapunov functional | 1,279 |
| `backfire-consciousness` | TCBO observer, PGBO bridge operator | 971 |
| `backfire-ssgf` | SSGF geometry engine (decoders, eigensolver, costs, gradients) | 1,750 |
| `backfire-ffi` | PyO3 Python bindings (13 classes) | 940 |

### 2.1 SSGF Gradient Methods

The SSGF engine supports two gradient computation methods:

- **`FiniteDifference`** (legacy): 240 forward passes, ~31.5 ms. Uses central differences through full decode + eigensolver + cost pipeline.
- **`Analytic`** (default): Chain rule through W-dependent costs only (C9, C10, R_graph Frobenius), with stop-gradient on eigensolver-dependent terms. Measured at **4.14 µs** — a **7,609x speedup**.

The analytic Jacobian chains through the decoder:
- **gram_softplus**: `dU/dz[k] = (dU/dW[i,j] + dU/dW[j,i]) * sigmoid(z[k])`
- **rbf**: `dU/dz[i*D+d] = sum_j (dU/dW[i,j]) * W[i,j] * (z[j*D+d] - z[i*D+d]) / sigma^2`

---

## 3. Validation Results

### 3.1 Python Test Suite

- **397 tests** across 28 test files (Python 3.10, 3.11, 3.12)
- **86% code coverage** with branch coverage enabled
- **CI**: GitHub Actions with mypy, black 26.x, ruff, pytest

### 3.2 Rust Test Suite

- **153 tests** across 5 crates
- **29 Criterion benchmarks** with statistical analysis

### 3.3 Deadline Compliance

All operations meet the 50 ms deadline:

| Path | Time | Verdict |
|------|------|---------|
| Safety kernel (10 tokens) | 265 ns | PASS (188,679x under) |
| Full scoring pipeline (10 tokens) | 4.56 µs | PASS (10,965x under) |
| Streaming kernel (1000 tokens) | 332 µs | PASS (151x under) |
| SSGF outer step (analytic) | 261 µs | PASS (191x under) |
| SSGF outer step (FD) | 34.6 ms | PASS (1.44x under) |

---

## Legal & Attribution

**Organization**: Anulum CH&LI / Anulum Institute
**Author**: Miroslav Sotek — ORCID [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)
**Copyright**: (C) 1998-2026 Miroslav Sotek. All rights reserved.
**Website**: [www.anulum.li](https://www.anulum.li)
**License**: GNU AGPL v3 | Commercial licensing available
