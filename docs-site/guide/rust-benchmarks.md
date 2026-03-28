# Rust vs Python Signal Benchmarks

The optional `backfire-kernel` Rust extension (PyO3/maturin) accelerates five scoring signals and BM25 retrieval. All numbers measured on a single core, 5,000 iterations, median microseconds.

## Signal Performance

| Signal | Python (us) | Rust (us) | Speedup | Notes |
|--------|------------|-----------|---------|-------|
| **BM25 query** (100 docs) | 110.2 | 10.8 | **10.2x** | Largest absolute gain — dominates retrieval-heavy pipelines |
| **trend_drop** | 6.2 | 0.3 | **20.7x** | Linear regression for streaming halt detection |
| **entity_overlap** | 14.7 | 3.7 | **4.0x** | Named entity set intersection |
| **numerical_consistency** | 14.8 | 2.3 | **6.4x** | Number extraction + comparison |
| **negation_flip** | 11.8 | 14.4 | 0.8x | Python wins — regex overhead in Rust FFI |
| **traceability** | 12.2 | 22.2 | 0.5x | Python wins — tokenisation overhead in Rust FFI |

Rust wins decisively on computation-heavy signals (BM25, trend_drop, entity, numerical). Python wins on signals dominated by regex and tokenisation where the FFI boundary overhead exceeds the compute savings.

## When to Use Rust

- **RAG pipelines**: BM25 at 10.2x is the biggest win. If your pipeline does retrieval on every scoring call, Rust saves ~100 us per call.
- **Streaming halt**: trend_drop at 20.7x means the streaming kernel can check halt conditions at sub-microsecond cost.
- **High-throughput servers**: at 1,000 RPS, the 4-6x speedup on entity + numerical saves ~25 ms/s of CPU time.

## Installation

```bash
pip install backfire-kernel   # pre-built wheels for Linux/macOS/Windows
```

Director-AI auto-detects the Rust extension and uses it when available. No code changes needed — the same `CoherenceScorer` and `StreamingKernel` APIs transparently dispatch to Rust.

## Reproducing

```bash
python -m benchmarks.rust_signals_bench --iterations 5000
```

Results are saved to `benchmarks/results/rust_signals_bench.json`.

## Methodology

- **Platform**: measured on Linux x86_64 (CI runner), Python 3.12, Rust 1.82
- **Warmup**: 500 iterations discarded before measurement
- **Iterations**: 5,000 timed iterations per signal
- **Metric**: median and p95 wall-clock time in microseconds
- **Data**: fixed synthetic input (100-word claim, 500-word source, 100-doc corpus for BM25)
