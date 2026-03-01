# Rust Backfire-Kernel FFI Guide

The safety interlock has a Rust implementation (`backfire-kernel/`) that
provides zero-copy token processing at <0.02 ms/token. Python binds via
PyO3 + Maturin.

## Architecture

```
backfire-kernel/
├── crates/
│   ├── backfire-types/       # Shared types (BackfireConfig, CoherenceScore, StreamSession)
│   ├── backfire-core/        # SafetyKernel, StreamingKernel, CoherenceScorer, InMemoryKnowledge
│   ├── backfire-physics/     # UPDE stepper, L16 controller, SEC functional
│   ├── backfire-observers/   # TCBO observer/controller, PGBO engine
│   ├── backfire-ssgf/        # SSGF geometry engine (Rust port)
│   └── backfire-ffi/         # PyO3 Python bindings → `backfire_kernel` module
├── benches/                  # Criterion benchmarks
├── Cargo.toml                # Workspace root
└── BENCHMARKS.md             # Measured results
```

## Build from source

```bash
cd backfire-kernel
pip install maturin
maturin develop --release -m crates/backfire-ffi/Cargo.toml
```

This compiles `backfire_kernel.so` (Linux) / `backfire_kernel.pyd` (Windows)
and installs it into your Python environment.

## Python usage

```python
from backfire_kernel import RustSafetyKernel, RustStreamingKernel, BackfireConfig

# Basic safety kernel
kernel = RustSafetyKernel(hard_limit=0.5)
result = kernel.stream_output(
    ["Hello ", "world ", "!"],
    lambda token: 0.85,  # coherence callback
)

# Streaming kernel with full session tracking
streaming = RustStreamingKernel(
    hard_limit=0.5,
    window_size=10,
    window_threshold=0.55,
)
session = streaming.stream_tokens(
    ["The ", "sky ", "is ", "blue."],
    lambda token: 0.9,
)
print(session.halted, session.avg_coherence)
```

## Auto-detection in CoherenceAgent

`CoherenceAgent` auto-detects the Rust kernel at import time:

```python
from director_ai import CoherenceAgent

agent = CoherenceAgent()
# If backfire_kernel is installed, agent uses Rust kernel.
# Otherwise, falls back to pure-Python kernel.
print(agent._rust_available)  # True if Rust kernel loaded
```

## Exposed Python classes

| Class | Description |
|-------|-------------|
| `BackfireConfig` | Configuration (thresholds, window sizes, weights) |
| `RustSafetyKernel` | Hard-limit safety interlock |
| `RustStreamingKernel` | Sliding-window streaming oversight |
| `RustCoherenceScorer` | Full dual-entropy scorer (NLI + RAG) |
| `UPDEStepper` | SCPN phase dynamics solver |
| `L16Controller` | Director-layer cybernetic closure |
| `SSGFEngine` | Stochastic geometry engine |
| `TCBOObserver` | Topological consciousness boundary |
| `PGBOEngine` | Phase-geometry bridge operator |

## FFI safety guarantees

- GIL acquired via `Python::with_gil` before every Python callback
- Python exceptions map to safe Rust defaults (0.0 for scores)
- No borrowed references escape the GIL lock scope
- All config validated before storage (`BackfireConfig::validate()`)

## Benchmarks

From `backfire-kernel/BENCHMARKS.md`:

| Operation | Rust | Python | Speedup |
|-----------|------|--------|---------|
| Token processing | 0.02 ms | 0.15 ms | 7.5x |
| Streaming session (100 tokens) | 1.8 ms | 15 ms | 8.3x |
| UPDE step (16 layers) | 0.003 ms | 0.05 ms | 16x |

Run benchmarks: `cd backfire-kernel && cargo bench`

## Pre-built wheels

Pre-built wheels are available on PyPI for:

- `manylinux2014_x86_64` (Linux x86-64)
- `manylinux2014_aarch64` (Linux ARM64)

Windows and macOS require building from source via `maturin develop`.
