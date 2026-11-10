<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI — streaming runtime comparison -->
# Streaming runtime comparison

The comparison runs the real Rust batch interface through Python FFI,
Python streaming orchestration with native statistics, and the generated
Node/WASM per-token interface. Every backend must consume the same complete
workload; a missing runtime, halted session or malformed report fails the run.

From the repository root, install the project and its native extension, then
build the separate Node package with wasm-pack 0.15.1 and Node 26.13.0:

```bash
maturin build --release --locked --manifest-path backfire-kernel/crates/backfire-ffi/Cargo.toml --out dist/backfire-kernel
python -m pip install --no-deps dist/backfire-kernel/backfire_kernel-*.whl
make wasm-build-node
python -m benchmarks.wasm_overhead_bench \
  --wasm-module backfire-kernel/crates/backfire-wasm/pkg-nodejs/backfire_wasm.js \
  --output benchmarks/results/wasm_streaming_comparison.json
```

The default is **1,000 tokens × 50 independent sessions per backend**, with
a supplied coherence score of 0.8 and the same streaming thresholds. Token
strings and kernels are prepared before each timed session. Module load,
kernel construction, report validation and artefact hashing are outside the
timed interval. No inference model is loaded or timed.

The interfaces measure different work:

| Interface | Timed work |
| --- | --- |
| Rust FFI | One Python-to-Rust batch call, with a Python callback per token |
| Python | Accumulated text, events, tracing and mandatory native statistics |
| Node/WASM | A JS-to-WASM call returning a full session snapshot per token |

These are **shared-host functional measurements**, explicitly marked
non-isolated. They do not establish pure boundary overhead, scoring accuracy,
production latency, a speedup guarantee or fairness between different APIs.
Python coverage instrumentation also changes timings; use a separate plain
CLI invocation for recorded comparisons.

JSON retains all raw millisecond samples, nearest-rank p95, median, extrema,
microseconds per token, configuration, host context and input hashes. Hashes
identify bytes; retain the build command and source identities separately to
prove a generated binary's provenance. Existing output files are refused.

For a bounded functional check, pass `--tokens 3 --iterations 2`. The full
comparison in CI keeps the default workload. `--node` selects the real Node
executable and `--timeout` bounds and reaps that child; all native failures
propagate.

The Python API in `wasm_overhead_bench` exposes:

- `bench_native_rust(tokens=1000, iterations=50, *, config_json=CONFIG_JSON)`
  and `bench_python(...)`: raw session milliseconds. The optional complete
  native configuration is validated and its streaming fields are shared.
- `bench_wasm(module, tokens=1000, iterations=50, *, node="node", timeout=120)`:
  validated samples and the complete original Node report.
- `parse_wasm_report(payload, tokens, iterations)`: the same decoder used
  for subprocess output, also suitable for a saved Node report. Invalid
  samples or workload identities are refused.
- `report(name, timings, tokens=1000)`: a summary without rounding or dropping
  observations.
- `run_comparison(module, ...)`: all three real backends and host context.

The required benchmark job builds the actual FFI and Node package, checks
strict Python and JavaScript types and documentation, exercises the real
cross-runtime CLI, and enforces complete authored-file coverage. The Rust
job separately executes wasm32 tests in Node; host-only Rust tests cannot
replace that lane. Web package builds still write `pkg/`; Node builds write
`pkg-nodejs/` and preserve an existing web package.
