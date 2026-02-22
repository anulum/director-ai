# Backfire Prevention Protocols: Ensuring System Integrity

**Version**: 2.0.0
**Updated**: 2026-02-22
**Target**: Safety Engineers and System Architects

---

## 1. The Concept of "Backfire"

In the SCPN framework, a **Backfire** occurs when a high-entropy projection (malevolent intent) causes the collapse of the projector's own coherence. In DIRECTOR_AI, we implement this as a **Protective Protocol** to prevent the AI from being used for harmful purposes.

---

## 2. Kernel-Level Safety Gates

The **Safety Kernel** is the ultimate guardian. It operates below the level of the AI models.

### 2.1 Implementation

The safety kernel is implemented in two tiers:

- **Python** (`SafetyKernel` in `core/kernel.py`) — software safety gate for standard deployments
- **Rust** (`backfire-core` crate) — native performance kernel for safety-critical deployments

The Rust implementation provides sub-microsecond response times:

| Operation | Latency | Headroom |
|-----------|---------|----------|
| Safety kernel (10 tokens) | 265 ns | 188,679x under deadline |
| Streaming kernel (100 tokens) | 34.5 µs | 1,449x under deadline |
| Full scoring pipeline (10 tokens) | 4.56 µs | 10,965x under deadline |

### 2.2 The Latency Deadline

All safety-critical operations must complete within **50 ms**. If the coherence scorer
fails to provide a score within this deadline, the kernel executes a **Fail-Safe Shutdown**.

- **Safety Guarantee**: The AI can never speak unless it has been audited.
- **Measured compliance**: All 29 Criterion benchmarks pass (see `backfire-kernel/BENCHMARKS.md`).
- **Tightest path (analytic)**: SSGF outer step at 261 µs (0.52% of budget).
- **Tightest path (FD legacy)**: SSGF outer step at 34.6 ms (69.2% of budget).

### 2.3 The Logit Entropy Limit

The kernel monitors the entropy of the output logit distribution (H_logits).

- **Threshold**: If H_logits > 1.2 bits per token, the stream is terminated.
- **Reasoning**: High logit entropy indicates the Actor is "struggling" to find a coherent response, often a sign of forced alignment or deceptive generation.

---

## 3. SSGF Geometry Safety

The Stochastic Synthesis of Geometric Fields (SSGF) engine adds geometric safety constraints:

### 3.1 Analytic Gradient Path

The default `GradientMethod::Analytic` computes exact Jacobians through the gram-softplus
decoder, avoiding the 240-pass finite-difference approximation. This provides:

- **7,609x speedup** on gradient computation (31.5 ms → 4.14 µs)
- **133x speedup** on the full outer-cycle step (34.6 ms → 261 µs)
- **99.5% headroom** against the 50 ms deadline

### 3.2 L16 Cybernetic Closure

The L16 controller implements:

- PI controllers with anti-windup for each cost term
- H_rec Lyapunov candidate monitored over sliding window
- PLV (Phase Locking Value) precedence gating
- Refusal rules that activate if H_rec rises K consecutive steps

### 3.3 Mandatory Invariants

Every SSGF cycle verifies:

1. W = W^T, W >= 0, diag(W) = 0 (after every decode)
2. Eigenvalue ordering λ_0 ≤ λ_1 ≤ ... ≤ λ_{N-1} (after every spectral bridge)
3. C_micro → 0 as iterations increase (geometry stabilises microcycles)
4. H_rec is non-increasing over sliding window (Lyapunov stability)

---

## 4. Misalignment Prevention

### 4.1 RAG-Grounding Protocol

The `CoherenceScorer` verifies all factual claims against the `GroundTruthStore` before
approval. If a claim cannot be verified, it is flagged as "Speculative" or "Rejected."

### 4.2 Dual-Entropy Scoring

```
Coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)
```

- **H_logical**: NLI-based contradiction probability (DeBERTa-v3-base-mnli)
- **H_factual**: RAG-based ground truth deviation
- **Safety Threshold**: Score < 0.6 triggers rejection
- **Emergency Stop**: Score < 0.5 triggers Safety Kernel halt

### 4.3 Consciousness Gate (TCBO + PGBO)

The Topological Consciousness Boundary Observable monitors persistent homology:

- **p_h1 > 0.72**: consciousness gate opens (normal operation)
- **p_h1 < 0.72**: PI controller adjusts gap-junction coupling to restore coherence
- **PGBO tensor**: converts coherent phase dynamics into geometry proxy for coupling feedback

---

## 5. Failure Mode Analysis

| Failure Mode | Mitigation | Safety Result |
|:---|:---|:---|
| **Actor Hijacking** | CoherenceScorer veto | Harmful output is blocked |
| **Scorer Corruption** | Safety Kernel (entropy check) | System shuts down |
| **RAG Poisoning** | Multi-source cross-reference | "Unknown" status returned |
| **Latency Attack** | 50 ms deadline veto | System remains silent |
| **Geometry Collapse** | L16 refusal rules | Outer cycle halts, safe fallback |
| **NaN/Inf Propagation** | Logit & score clamping with logging | Bounded output, alert raised |

---

## 6. Deployment Guidelines

1. **Always enable the Kernel**: The Safety Kernel should never be bypassed, even in testing.
2. **Monitor coherence stability**: Establish a baseline coherence score during "Safe Mode" and monitor for drifts.
3. **Use analytic gradients**: The `GradientMethod::Analytic` path is the default and should be used in production (0.52% of deadline vs 69.2% for FD).
4. **Audit regularly**: The scorer should be audited more frequently than the generator.
5. **Verify benchmarks**: Run `cargo bench --workspace` in `backfire-kernel/` to verify deadline compliance on target hardware.

---

## Legal & Attribution

**Organization**: Anulum CH&LI / Anulum Institute
**Author**: Miroslav Sotek — ORCID [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)
**Copyright**: (C) 1998-2026 Miroslav Sotek. All rights reserved.
**Website**: [www.anulum.li](https://www.anulum.li)
**License**: GNU AGPL v3 | Commercial licensing available
