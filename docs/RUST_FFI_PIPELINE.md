# Director-AI Rust FFI Pipeline (Backfire Kernel)

> **Crate**: `backfire-kernel` | **Version**: 3.11.1 | **License**: GNU AGPL v3
>
> © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
> © Code 2020–2026 Miroslav Šotek. All rights reserved.

---

## Overview

The Backfire Kernel is a Rust implementation of Director-AI's core scoring,
streaming safety, and verification signal functions. It exposes a Python
API via PyO3 (FFI), providing a 14.2× speedup over the equivalent Python
code for heuristic scoring operations.

The Rust path is not a replacement for the Python stack — it is an
**accelerator**. The Python CoherenceScorer delegates to the Rust backend
when `scorer_backend="rust"` is specified and `backfire_kernel` is installed.
When Rust is unavailable, the system falls back to the Python implementation
transparently.

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

# Falls back to Python if Rust not installed
scorer = CoherenceScorer(scorer_backend="rust", threshold=0.5)
# → logs warning, uses Python heuristic scorer
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
    _RUST_SIGNALS = False
```

When `_RUST_SIGNALS is True`, the verified scorer dispatches individual
signal computations to Rust. The Python fallbacks are used otherwise.

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
- Parametrised thresholds, streaming lengths, input variations
- Performance: review latency <100 µs, throughput <50 µs/token

---

## Error Handling

| Scenario                     | Behaviour                              |
|------------------------------|----------------------------------------|
| Rust not installed           | Python fallback, warning logged        |
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
| Verification signals      | `backfire-kernel/crates/backfire-core/src/signals.rs` |
| Shared types              | `backfire-kernel/crates/backfire-types/`       |
| Python integration tests  | `tests/test_rust_pipeline_integration.py`      |
| Python FFI tests          | `tests/test_ffi_bindings.py`                  |
| Python backend registry   | `src/director_ai/core/scoring/backends.py`    |
| Python verified scorer    | `src/director_ai/core/scoring/verified_scorer.py` |
