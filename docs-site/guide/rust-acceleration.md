# Rust Acceleration

*Added in v3.11.0*

Director-AI optionally accelerates hot-path functions via the Rust-based Backfire Kernel. When installed, several hot-path functions dispatch to compiled Rust code automatically. When not installed, Python fallbacks run transparently.

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

| Function | Python | Rust | Speedup | Called |
|----------|--------|------|---------|-------|
| BM25 query (1000 docs) | ~30 ms | ~10 ms | 3x | per retrieval in hybrid mode |

The `RustBM25` class is available directly:

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

**NLI mode (production):** Model inference is 19-200 ms. Python signal extraction adds ~3 ms. Rust saves ~2 ms — marginal improvement.

**Heuristic/fallback mode:** No model inference. Python signal extraction is 20-50 ms of the total 50-100 ms. Rust saves 15-30 ms — meaningful improvement.

**Hybrid retrieval:** BM25 inner loop on 1000+ docs benefits from Rust's tight arithmetic loops and no-GIL execution.

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

Python auto-dispatch (no code changes needed):

```python
# verified_scorer.py — automatic
try:
    from backfire_kernel import rust_entity_overlap, ...
    _RUST_SIGNALS = True
except ImportError:
    _RUST_SIGNALS = False

def _entity_overlap(text_a, text_b):
    if _RUST_SIGNALS:
        return rust_entity_overlap(text_a, text_b)
    # Python fallback...
```

## Verification

Check if Rust acceleration is active:

```python
from director_ai.core.scoring.verified_scorer import _RUST_SIGNALS
from director_ai.core.runtime.streaming import _RUST_TREND

print(f"Rust signals: {_RUST_SIGNALS}")
print(f"Rust trend:   {_RUST_TREND}")
```

## Running Rust Tests

```bash
cd backfire-kernel
cargo test                    # all crates
cargo test -p backfire-core   # core only (signals + bm25)
cargo bench -p backfire-core  # scoring benchmarks
```
