# Rust Acceleration

*Added in v3.11.0*

Director-AI uses the Backfire Kernel for selected scoring, retrieval and
streaming operations. Availability depends on the operation. In-memory
retrieval has a Python fallback. Mandatory operations such as streaming trend
checks require the kernel and report how to install it if it is absent.

## Install

```bash
pip install director-ai[rust]
```

Or build from source:

```bash
cd backfire-kernel
pip install -e crates/backfire-ffi  # requires maturin + Rust toolchain
```

## What Gets Accelerated

### Verification Signals (VerifiedScorer)

| Function | Python | Rust | Speedup | Called |
|----------|--------|------|---------|-------|
| `entity_overlap` | ~0.5 ms | ~0.1 ms | 5x | per claim × source match |
| `numerical_consistency` | ~0.05 ms | ~0.01 ms | 5x | per claim × source match |
| `negation_flip` | ~0.1 ms | ~0.02 ms | 5x | per claim pair |
| `traceability` | ~2 ms | ~0.5 ms | 4x | per claim |

These run inside `VerifiedScorer.verify()` for each claim. With 5 claims and 3 source matches each, the aggregate savings are 5-15 ms per response in heuristic mode.

### Streaming Kernel

| Function | Python | Rust | Speedup | Called |
|----------|--------|------|---------|-------|
| `trend_drop` | ~0.1 ms | ~0.02 ms | 5x | per trend check in streaming |

### BM25 Retrieval

The `HybridBackend` ranks BM25 terms in Python and uses a Rust integer sum
when the kernel is installed. The underlying `InMemoryBackend` can call Rust
for word overlap on the same normalised tokens as the Python path. Standalone
BM25 benchmarks exercise the separate `RustBM25` interface. [Signal benchmarks](rust-benchmarks.md)

Use that standalone interface as follows.

```python
from backfire_kernel import RustBM25

bm25 = RustBM25(k1=1.2, b=0.75)
bm25.add_document("d1", "Water boils at 100 degrees Celsius.")
bm25.add_document("d2", "The speed of light is 299792 km/s.")

results = bm25.query("water temperature", n_results=3)
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

### PII Detection

`RegexPIIDetector` (and `PIIRedactor`) batch all PII patterns behind a single
Rust `RegexSet` via `backfire_kernel.PiiScanner`, scanning each text in one pass.

| Function | Python | Rust | Speedup | Called |
|----------|--------|------|---------|-------|
| PII scan (8 categories) | ~76k texts/s | ~322k texts/s | 4.2x | per redaction / moderation check |

Measured by `benchmarks/pii_redaction.py` (`prefer_rust=True` vs `False`): the
Rust and Python paths produce identical `(category, start, end)` spans (parity),
and on the labelled corpus the detector scores precision 0.94 / recall 1.00
(span-level, across the eight categories). The same 21-sample regression corpus
also records the optional Presidio comparison when installed; the 2026-06-18 run
reported Presidio precision 0.85 / recall 0.65 / F1 0.73 after entity-label
normalisation. Unlike simple reductions — where FFI marshalling can make Rust
slower — the multi-pattern `RegexSet` scan is a genuine win. The detector falls
back to the pure-Python regex path transparently when the kernel is not
installed; choose the path explicitly with `RegexPIIDetector(prefer_rust=...)`.

## When Rust Helps

**Model-backed review** delegates NLI to the configured inference backend.
Assess latency with retrieval, inference and review enabled on the same
workload. The `rust` scorer is a heuristic backend and is refused by profiles
that require model-backed NLI. [Production profile](../deployment/production.md)

## Architecture

```
backfire-kernel/crates/
├── backfire-core/src/
│   ├── signals.rs     ← verification signal functions
│   ├── bm25.rs        ← BM25 retrieval engine
│   ├── scorer.rs      ← coherence scorer
│   ├── kernel.rs      ← safety/streaming kernel
│   └── knowledge.rs   ← ground truth store
├── backfire-ffi/src/
│   └── lib.rs         ← PyO3 bindings (rust_entity_overlap, RustBM25, etc.)
└── Cargo.toml         ← workspace config
```

## Verification

Check the installed kernel version in the runtime environment.

```bash
python -m pip show backfire-kernel
```

## Running Rust Tests

```bash
cd backfire-kernel
cargo test                    # all crates
cargo test -p backfire-core   # core only (signals + bm25)
cargo bench -p backfire-core  # scoring benchmarks
```
