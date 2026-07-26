# Director-AI Rust FFI Pipeline (Backfire Kernel)

> **Crate**: `backfire-kernel` | **Director-AI release**: 3.20.0 | **License**: Apache-2.0
>
> © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
> © Code 2020–2026 Miroslav Šotek. All rights reserved.

---

## Overview

The Backfire Kernel is a Rust implementation of Director-AI's core scoring,
streaming safety, and verification signal functions. It exposes a Python
API via PyO3 (FFI), providing a 14.2× speedup over the equivalent Python
code for heuristic scoring operations.

The Backfire Kernel is a **mandatory production accelerator**, not an optional
add-on: `backfire-kernel` is pinned in `[project].dependencies`, so every
install ships the compiled extension. In production, accelerated paths dispatch
to Rust and refuse to silently degrade; the equivalent pure-Python reference
implementations are retained and unit-tested but are not reached at runtime. See
[Accelerator policy — Rust is mandatory in production (ADR-1)](#accelerator-policy--rust-is-mandatory-in-production-adr-1)
for the authoritative statement of this behaviour.

---

## Accelerator policy — Rust is mandatory in production (ADR-1)

**Status:** Accepted — records the shipped v3.16.x behaviour.
**Scope:** the `backfire_kernel` accelerator across all core and operational modules.

**Context.** `backfire-kernel` is a **mandatory runtime dependency** (pinned in
`[project].dependencies`), not an optional extra — every `pip install
director-ai` ships the compiled extension. Each accelerated module imports its
Rust symbols behind a `try/except ImportError` guard purely so the module stays
importable in an *unbuilt editable checkout* (type-checking, docs, tooling).
Crucially, the availability flag is set **`True` in both branches**, and the
`except` branch installs stub functions that **raise** `RuntimeError`:

```python
try:
    from backfire_kernel import rust_entity_overlap, ...

    _RUST_SIGNALS = True
except ImportError:
    _RUST_SIGNALS = True

    def rust_entity_overlap(_claim: str, _source: str) -> float:
        raise RuntimeError("backfire_kernel rust_entity_overlap is unavailable")
```

Rust-backed calls run inside `mandatory_execution()` (`core/mandatory.py`),
which logs and **re-raises**: DIRECTOR-AI "treats declared accelerators … as
required production capabilities … preventing silent fallback or degraded
behaviour."

**Decision.** In production the mandatory Rust path is always taken; a missing or
unbuilt extension surfaces a clear `RuntimeError` rather than silently degrading.
The equivalent **Python floor implementations are retained and unit-tested** —
the fallback branches are exercised by monkeypatching `_RUST_* = False` (e.g.
`tests/test_rust_fallback_floor.py`, `tests/test_rust_signals.py`), so every
kernel keeps a verified pure-Python reference, but that path is unreachable at
runtime with the flag hard-set to `True`.

**Consequences.**
1. A correct install always has Rust; there is no automatic production fallback.
2. The GOTM "Python-floor" rule is satisfied at the *code and test* level — every
   kernel has a tested Python implementation — while production mandates the
   accelerated path.
3. Where prose elsewhere in this document mentions a "Python fallback", it refers
   to this tested floor, not to automatic runtime degradation.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         Python Layer                              │
│                                                                   │
│  CoherenceScorer(scorer_backend="rust")                           │
│       │                                                           │
│       ▼                                                           │
│  ┌─────────────────────────────────────────────────────┐          │
│  │  director_ai.core.scoring.backends.get_backend()    │          │
│  │  → RustBackend (registered as "rust" / "backfire")  │          │
│  └────────────────────────┬────────────────────────────┘          │
│                           │                                       │
│                    ┌──────▼──────┐                                │
│                    │  import     │                                │
│                    │  backfire_  │                                │
│                    │  kernel     │                                │
│                    └──────┬──────┘                                │
└───────────────────────────┼───────────────────────────────────────┘
                            │ PyO3 FFI boundary
┌───────────────────────────┼───────────────────────────────────────┐
│                           ▼         Rust Layer                    │
│                                                                   │
│  ┌─────────────────────────────────────────────────┐              │
│  │            backfire-ffi (lib.rs)                 │              │
│  │  PyBackfireConfig, PyCoherenceScore,             │              │
│  │  PyStreamSession, RustSafetyKernel,              │              │
│  │  RustStreamingKernel, RustCoherenceScorer        │              │
│  └──────────────────────┬──────────────────────────┘              │
│                         │                                         │
│  ┌──────────────────────▼──────────────────────────┐              │
│  │            backfire-core                         │              │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │              │
│  │  │ scorer   │  │ kernel   │  │   signals    │  │              │
│  │  │ (299 ln) │  │ (486 ln) │  │  (335 ln)    │  │              │
│  │  └──────────┘  └──────────┘  └──────────────┘  │              │
│  └─────────────────────────────────────────────────┘              │
│                                                                   │
│  ┌─────────────────────────────────────────────────┐              │
│  │  backfire-types   backfire-observers             │              │
│  │  backfire-physics backfire-ssgf                   │              │
│  └─────────────────────────────────────────────────┘              │
└───────────────────────────────────────────────────────────────────┘
```

---

## Crate Structure

```
backfire-kernel/
├── Cargo.toml                          # workspace root
├── crates/
│   ├── backfire-ffi/                   # PyO3 bindings (Python ↔ Rust)
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml              # maturin build config
│   │   └── src/lib.rs                  # 1153 lines
│   ├── backfire-core/                  # core logic
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── scorer.rs               # 299 lines — coherence scoring
│   │       ├── kernel.rs               # 486 lines — safety + streaming
│   │       ├── signals.rs              # 335 lines — verification signals
│   │       ├── knowledge.rs            # knowledge store trait
│   │       └── nli.rs                  # external NLI trait
│   ├── backfire-types/                 # shared types
│   │   └── src/
│   │       ├── config.rs               # BackfireConfig
│   │       ├── score.rs                # CoherenceScore
│   │       └── session.rs              # StreamSession
│   ├── backfire-observers/             # PGBO + TCBO controllers
│   ├── backfire-physics/               # UPDE stepper, L16, SEC functional
│   └── backfire-ssgf/                  # SSGF engine
```

---

## FFI Bindings (backfire-ffi/lib.rs)

### PyBackfireConfig

Python-visible configuration wrapper around `BackfireConfig`.

**Constructor parameters**:

| Parameter              | Type   | Default | Description                        |
|------------------------|--------|---------|------------------------------------|
| `coherence_threshold`  | f64    | 0.6     | Composite score threshold          |
| `hard_limit`           | f64    | 0.5     | Hard halt limit                    |
| `soft_limit`           | f64    | 0.7     | Soft warning zone upper bound      |
| `w_logic`              | f64    | 0.6     | Weight for logical divergence      |
| `w_fact`               | f64    | 0.4     | Weight for factual divergence      |
| `window_size`          | usize  | 10      | Streaming window size (tokens)     |
| `window_threshold`     | f64    | 0.55    | Window average threshold           |
| `trend_window`         | usize  | 5       | Trend detection window             |
| `trend_threshold`      | f64    | 0.15    | Trend drop magnitude threshold     |
| `history_window`       | usize  | 5       | History buffer size                |
| `deadline_ms`          | u64    | 50      | Per-token deadline (ms)            |
| `logit_entropy_limit`  | f64    | 1.2     | Maximum logit entropy              |

**Validation**: calls `BackfireConfig::validate()` which enforces:
- All thresholds in [0.0, 1.0]
- `hard_limit ≤ soft_limit`
- `window_size ≥ 1`
- `deadline_ms ≥ 1`

Invalid configurations raise `PyValueError`.

**Methods**:
- `from_json(json: str) -> BackfireConfig`: construct from JSON string
- `__repr__()`: shows threshold, hard_limit, deadline_ms

### PyCoherenceScore

Wraps the Rust `CoherenceScore` struct.

| Property     | Type   | Description                            |
|--------------|--------|----------------------------------------|
| `score`      | f64    | Composite coherence score (0.0–1.0)    |
| `approved`   | bool   | Whether the output passes the threshold |
| `h_logical`  | f64    | Logical divergence component           |
| `h_factual`  | f64    | Factual divergence component           |
| `warning`    | bool   | Soft warning zone flag                 |
| `evidence`   | None   | Not computed on Rust side (API compat) |

**Methods**:
- `to_dict(py) -> PyDict`: serialise to Python dictionary
- `__repr__()`: formatted string with all fields

### PyStreamSession

Wraps the Rust `StreamSession` struct — the trace of a streaming evaluation.

| Property            | Type        | Description                       |
|---------------------|-------------|-----------------------------------|
| `halted`            | bool        | Whether streaming was halted      |
| `halt_index`        | i32         | Token index where halt occurred   |
| `halt_reason`       | str         | Reason for halt (or empty)        |
| `tokens`            | Vec<String> | All tokens processed              |
| `coherence_history` | Vec<f64>    | Per-token coherence scores        |

**Methods**:
- `output() -> str`: concatenate all tokens into final output
- `token_count() -> usize`: number of tokens processed
- `avg_coherence() -> f64`: mean coherence across all tokens
- `min_coherence() -> f64`: minimum coherence observed

### RustSafetyKernel

The safety kernel evaluates a complete response (non-streaming mode).

```python
from backfire_kernel import RustSafetyKernel

kernel = RustSafetyKernel(hard_limit=0.5)
result = kernel.review("context", "response", score_fn)
```

The `score_fn` callback is a Python callable that receives a text string
and returns a float (coherence score). The kernel calls it from Rust via
`Python::with_gil`, ensuring GIL safety.

### RustStreamingKernel

The streaming kernel evaluates tokens one at a time as they are generated.

```python
from backfire_kernel import RustStreamingKernel

kernel = RustStreamingKernel(config)
session = kernel.stream_output(
    ["Hello ", "world", "!"],
    score_fn=lambda text: 0.8,
)
print(session.halted)        # False
print(session.token_count()) # 3
print(session.output())      # "Hello world!"
```

The kernel monitors:
1. **Window average**: rolling average of last N token scores
2. **Trend detection**: linear regression slope over recent scores
3. **Hard limit**: immediate halt if any score drops below threshold

### RustCoherenceScorer

Heuristic scorer that computes coherence without NLI model inference.
Uses entity overlap, negation detection, numerical consistency, and
traceability signals.

```python
from backfire_kernel import RustCoherenceScorer

scorer = RustCoherenceScorer(threshold=0.5)
score = scorer.review("The sky is blue.", "The sky is blue.")
print(score.score)     # ~0.95
print(score.approved)  # True
```

---

## Verification Signals (backfire-core/signals.rs)

Four signal functions are ported from `verified_scorer.py` to Rust:

### 1. `entity_overlap(text_a, text_b) -> f64`

Jaccard overlap of proper-noun entities between two texts.

**Algorithm**:
1. Extract capitalised word sequences as entity candidates
2. Multi-word entities: consecutive capitalised words are merged
   (e.g. "New York" → single entity)
3. Compute Jaccard similarity: |A ∩ B| / |A ∪ B|
4. Returns 1.0 if neither text contains entities

**Example**:
```
entity_overlap("Paris is the capital of France",
               "The capital of France is Paris") → ~0.67
```

### 2. `numerical_consistency(text_a, text_b) -> Option<bool>`

Checks whether numbers in both texts overlap.

**Algorithm**:
1. Extract digit sequences (handles commas and dots as decimal/thousand separators)
2. Trim trailing punctuation from extracted numbers
3. Check if the two sets are **not disjoint** (any shared number)
4. Returns `None` if either text has no numbers

**Example**:
```
numerical_consistency("46 chromosomes", "humans have 46") → Some(true)
numerical_consistency("90 days", "30 days") → Some(false)
numerical_consistency("the sky", "is blue") → None
```

### 3. `negation_flip(claim, source) -> bool`

Detects if the claim negates something the source states positively
(or vice versa).

**Algorithm**:
1. Tokenise both texts to lowercase words
2. Check if one has a negation word (`not`, `never`, `can't`, etc.) and
   the other does not
3. If negation polarity differs, verify they share ≥3 non-negation content
   words (to confirm they are about the same topic)
4. Returns `true` only when polarity differs AND topic overlap is sufficient

**Negation words** (26 total): `not`, `no`, `never`, `neither`, `nor`,
`cannot`, `can't`, `isn't`, `aren't`, `wasn't`, `weren't`, `won't`,
`wouldn't`, `shouldn't`, `couldn't`, `doesn't`, `didn't`, `hasn't`,
`haven't`, `hadn't`, `without`, `none`, `nobody`.

### 4. `traceability(claim, source) -> f64`

Fraction of the claim's content words found in the source text.

**Algorithm**:
1. Tokenise both texts to lowercase words
2. Filter out stop words (65 entries) and negation words
3. Count how many claim content words appear in the source
4. Return `matched / total_claim_words`
5. Returns 1.0 if the claim has no content words

Low traceability indicates the claim contains information not present in
the source — a potential fabrication.

### 5. `trend_drop(values) -> f64`

Linear regression trend detection over a window of coherence scores.

**Algorithm**:
1. Compute least-squares slope of the score series
2. Return `-slope × (n - 1)` — the projected total drop
3. Positive values indicate declining coherence (degradation)
4. Returns 0.0 for single-element inputs

Used by the streaming kernel to detect gradual coherence decay that
individual-token thresholds would miss.

---

## FFI Safety Model

### GIL Handling

All Python callbacks are invoked via `Python::with_gil(|py| { ... })`.
This ensures:

- The GIL is held before any Python object access
- No borrowed Python references escape the GIL scope
- Python exceptions are caught and converted to safe Rust defaults

### Exception Safety

If a Python callback raises an exception:
- Score callbacks return `0.0` (most conservative — triggers halt)
- String callbacks return `None` or empty string
- The Rust kernel continues with safe defaults

### Memory Safety

- No `unsafe` blocks in hand-written code (only PyO3 proc macro generated)
- `Arc<BackfireConfig>` for shared config ownership
- `Clone` derived on all PyO3-exposed types
- No raw pointer arithmetic or manual memory management

### Config Validation

`BackfireConfig::validate()` is called on every construction path
(`new()` and `from_json()`). Invalid configs are rejected at construction
time, not at scoring time.

---

## Python Integration

### Backend Registration

The Rust backend is registered in Director-AI's backend registry under
two aliases: `"rust"` and `"backfire"`. Both resolve to the same backend
class.

```python
from director_ai.core.scoring.backends import get_backend

backend_cls = get_backend("rust")     # works
backend_cls = get_backend("backfire") # also works
```

### CoherenceScorer Integration

```python
from director_ai import CoherenceScorer

# Explicit Rust backend
scorer = CoherenceScorer(scorer_backend="rust", threshold=0.5)
approved, score = scorer.review("context", "response")

# backfire_kernel is a mandatory dependency, so the Rust backend is always
# available in a correct install; an unbuilt extension raises rather than
# silently substituting a Python scorer (see ADR-1 above).
scorer = CoherenceScorer(scorer_backend="rust", threshold=0.5)
```

### Signal Function Dispatch

The `VerifiedScorer` checks for Rust signal availability at import time:

```python
try:
    from backfire_kernel import (
        rust_entity_overlap,
        rust_negation_flip,
        rust_numerical_consistency,
        rust_traceability,
    )
    _RUST_SIGNALS = True
except ImportError:
    _RUST_SIGNALS = True  # mandatory: the stub below raises (see ADR-1)
```

The flag is `True` in both branches (see [ADR-1](#accelerator-policy--rust-is-mandatory-in-production-adr-1)):
the verified scorer always dispatches to Rust in production, and a missing
extension raises rather than degrading. The pure-Python signal implementations
remain as a unit-tested floor, reached only by monkeypatching
`_RUST_SIGNALS = False` in tests.

### Domain Kernel Dispatch (Operational Modules)

Beyond the scorer kernels, DIRECTOR-AI now dispatches selected operational
math paths to `backfire_kernel` where available, with deterministic
Python fallback on import or runtime FFI errors:

| Module | Rust function(s) | Purpose |
|---|---|---|
| `core/calibration/online_calibrator.py` | `rust_confusion_counts_threshold` | threshold sweep confusion-matrix counts |
| `core/sustainability/policy_adapter.py` | `rust_sum_i64`, `rust_sum_f64` | telemetry aggregate counters and mean calculations |
| `core/swarm_equilibrium/scorer.py` | `rust_mean` | mean Nash-payoff aggregation |
| `core/irreversibility/forecaster.py` | `rust_product_f64` | cumulative reversibility product across action chains |

Coverage enforcement lives in dedicated module tests:

- `tests/test_online_calibrator.py`
- `tests/test_sustainability.py`
- `tests/test_swarm_equilibrium.py`
- `tests/test_irreversibility.py`

Each suite contains explicit assertions for:

1. Rust path invocation when kernel symbols are available.
2. Python fallback behaviour when FFI raises `TypeError`/runtime errors.
3. Stable semantic output under deterministic seeded execution where applicable.

---

## Installation

### From source (development)

```bash
cd backfire-kernel
pip install maturin
maturin develop --release -m crates/backfire-ffi/Cargo.toml
```

### From wheel (production)

```bash
pip install backfire-kernel
```

### Requirements

- Rust toolchain (≥1.75, edition 2021)
- maturin (≥1.12)
- Python ≥3.10
- PyO3 ≥0.22

---

## Performance

### Heuristic Scoring

| Backend | Median latency | Throughput     | Hardware          |
|---------|----------------|----------------|-------------------|
| Python  | ~35 µs         | ~28,500 ops/s  | i7-12700K         |
| Rust    | ~2.5 µs        | ~400,000 ops/s | i7-12700K         |
| Speedup | **14.2×**      | **14.0×**      |                   |

### Signal Functions

| Signal                  | Python  | Rust    | Speedup |
|-------------------------|---------|---------|---------|
| `entity_overlap`        | ~12 µs  | ~0.8 µs | 15×     |
| `numerical_consistency` | ~8 µs   | ~0.5 µs | 16×     |
| `negation_flip`         | ~15 µs  | ~1.0 µs | 15×     |
| `traceability`          | ~10 µs  | ~0.7 µs | 14×     |

### Streaming Kernel

| Metric              | Python     | Rust       | Speedup |
|---------------------|------------|------------|---------|
| Per-token overhead  | ~50 µs     | ~3.5 µs   | 14×     |
| 100-token stream    | ~5 ms      | ~0.35 ms  | 14×     |
| Window trend calc   | ~20 µs     | ~1.4 µs   | 14×     |

### Performance Assertions in Tests

The test suite (`test_rust_pipeline_integration.py`) enforces:
- Review latency < 1 ms per call
- Rust backend ≥ 2× faster than Python (conservative bound)
- Per-token streaming overhead measurably faster than Python

---

## Testing

### Rust unit tests (13 tests in signals.rs)

```bash
cd backfire-kernel
cargo test
```

Tests cover:
- `entity_overlap`: identical texts, no entities, partial overlap
- `numerical_consistency`: matching, mismatching, no numbers
- `negation_flip`: detected, same polarity (no flip)
- `traceability`: high overlap, low overlap
- `trend_drop`: flat, declining, single value

### Python integration tests (26 tests in test_rust_pipeline_integration.py)

```bash
pytest tests/test_rust_pipeline_integration.py -v
```

Test classes:
- `TestRustBackendRegistration`: backend discovery, aliasing
- `TestRustScorerPipeline`: end-to-end CoherenceScorer(backend="rust")
- `TestRustKnowledgeCallback`: FFI boundary callback testing
- `TestRustPythonConsistency`: agreement between Rust and Python backends
- `TestSignalDispatch`: signal function availability verification
- `TestRustPerformanceDoc`: latency and speedup assertions

### Python FFI binding tests (73 tests in test_ffi_bindings.py)

```bash
pytest tests/test_ffi_bindings.py -v
```

Tests cover:
- `BackfireConfig`: construction, validation, JSON parsing
- `CoherenceScore`: property access, to_dict()
- `StreamSession`: token management, halt tracking
- `RustSafetyKernel`: review with callbacks
- `RustStreamingKernel`: streaming with halt detection
- `rust_heuristic_logical_divergence`: fallback logical-divergence parity
- `rust_heuristic_factual_divergence`: fallback factual-divergence parity
- `rust_split_sentences`: NLI chunking sentence splitter fast-path
- `rust_build_chunks`: NLI chunk builder fast-path with overlap routing
- `rust_aggregate_chunk_scores`: NLI chunk matrix aggregation fast-path
- `rust_aggregate_chunk_scores_confidence_weighted`: weighted chunk aggregation fast-path
- `rust_coverage_from_divergences`: claim-support coverage reducer fast-path
- `rust_reduce_claim_attribution`: claim×source attribution argmin reducer fast-path
- `_task_scoring.minicheck_claim_coverage`: Rust sentence split + Rust coverage reduction
- `verified_scorer` lexical fallback matching: Rust `word_overlap` acceleration
- `consensus` lexical divergence scoring: Rust `word_overlap` acceleration
- `meta_classifier` lexical `word_overlap` feature extraction: Rust acceleration
- `safety/injection` claim fallback decomposition: Rust sentence splitting acceleration
- `verified_scorer` response/source sentence decomposition: Rust sentence splitting acceleration
- `verification/reasoning_verifier` sentence-fallback decomposition: Rust sentence splitting acceleration
- `retrieval/contextual_compression` keyword-overlap scoring: Rust `word_overlap` acceleration
- `agentic/loop_monitor` goal-drift Jaccard scoring: Rust `word_overlap` acceleration
- `scoring/distilled_scorer` ONNX-path softmax helper: Rust `softmax` acceleration
- `retrieval/knowledge` keyword-store overlap ranking: Rust `word_overlap` acceleration
- `autopoietic/builder` n-gram overlap scorer: Rust `word_overlap` acceleration
- `retrieval/vector_store/base` in-memory ranking overlap: Rust `word_overlap` acceleration
- `retrieval/doc_chunker` semantic sentence splitting: Rust `split_sentences` acceleration
- Parametrised thresholds, streaming lengths, input variations
- Performance: review latency <100 µs, throughput <50 µs/token

---

## Error Handling

| Scenario                     | Behaviour                              |
|------------------------------|----------------------------------------|
| Rust extension unbuilt       | `RuntimeError` raised (mandatory; see ADR-1) |
| Invalid config               | `PyValueError` raised at construction  |
| Callback exception           | Safe default (0.0), kernel continues   |
| NaN/Inf score from callback  | Treated as 0.0 (conservative)          |
| Empty token list             | Empty session returned (no halt)       |

---

## Relationship to SCPN Physics

The `backfire-physics` crate contains SCPN (Self-Correcting Protoscientific
Network) physics components:

- `UPDEStepper`: UPDE (Universal Protoscientific Differential Equation) integration
- `L16Controller`: 15+1 layer cost functional
- `SECFunctional`: SEC (Systematic Error Correction) functional

These are used by the `backfire-observers` crate (PGBO and TCBO controllers)
for physics-informed coherence monitoring. They are not directly exposed
to Python in the current version but are wired into the kernel's internal
scoring path.

---

## File Reference

| Item                      | Path                                          |
|---------------------------|-----------------------------------------------|
| FFI bindings              | `backfire-kernel/crates/backfire-ffi/src/lib.rs` |
| Core scorer               | `backfire-kernel/crates/backfire-core/src/scorer.rs` |
| Safety/streaming kernel   | `backfire-kernel/crates/backfire-core/src/kernel.rs` |
| Compute accelerators      | `backfire-kernel/crates/backfire-core/src/compute/` (facade: `compute.rs`) |
| Statistical primitives    | `backfire-kernel/crates/backfire-core/src/stats.rs` (bindings validate, core computes) |
| Verification signals      | `backfire-kernel/crates/backfire-core/src/signals.rs` |
| Shared types              | `backfire-kernel/crates/backfire-types/`       |
| Python integration tests  | `tests/test_rust_pipeline_integration.py`      |
| Python FFI tests          | `tests/test_ffi_bindings.py`                  |
| Python backend registry   | `src/director_ai/core/scoring/backends.py`    |
| Python verified scorer    | `src/director_ai/core/scoring/verified_scorer.py` |
