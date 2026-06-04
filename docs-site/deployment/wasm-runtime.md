<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — WASM runtime deployment guide
-->

# WASM Runtime

The WASM runtime is the priority path for browser, Worker, and offline hosts
that cannot run Python services. It is a halt-decision runtime only: the host
supplies one coherence score per token, and `backfire-wasm` applies the same
hard-limit, window-average, downward-trend, and irrevocable-halt checks used by
the streaming kernel.

Use the Python quickstart for server deployments. Use WASM when the deployment
must stay inside a browser, a local desktop bundle, or an edge function with no
Python process.

## Scope

| Included | Host-owned |
|----------|------------|
| `WasmStreamingKernel` session state | NLI or heuristic scorer |
| Hard-limit halt | Tokenisation cadence |
| Sliding-window halt | Retrieval or local fact store |
| Downward-trend halt | Trace export and host telemetry |
| Irrevocable halt after first failure | UI rendering and redaction |

The crate does not ship a neural scorer, tenant routing, auth, or audit sinks.
Those remain host responsibilities.

## Build And Test

From the repo root:

```bash
make test-wasm
make wasm-build
```

Direct crate commands:

```bash
cd backfire-kernel/crates/backfire-wasm
wasm-pack test --node
wasm-pack build --target web --release
```

CI already builds the web target in `.github/workflows/wheels.yml` and uploads
the `pkg/` directory as the `wasm-edge-runtime` artefact.

## Readiness Evidence

Before claiming an edge or mobile release, generate the R14 readiness packet:

```bash
PYTHONPATH=src python -m benchmarks.edge_mobile_evidence
```

The packet records whether the current checkout is ready for a local
low-latency trial or for release. Release readiness requires:

- local `wasm-pack` artefacts in `backfire-kernel/crates/backfire-wasm/pkg/`
- a quantised ONNX model artefact
- browser or Web Worker smoke evidence
- mobile or embedded-device smoke evidence
- Rust Python accelerator import evidence

If those artefacts are absent, the packet remains useful but must be treated as
local-trial evidence only.

## Release Priority

`requirements/wasm_release_plan.toml` is the tracked source for the host
targets and release gates.

| Phase | Status | Goal |
|-------|--------|------|
| `p0-local-artefact` | active | Ship browser and Worker artefacts from CI for offline trials. |
| `p1-package-publish` | next | Publish JS glue plus `.wasm` as an operator-triggered package release. |
| `p2-edge-scoring-bridge` | later | Document host scoring bridges for browser and Worker runtimes. |

## Target Matrix

| Target id | Runtime | `wasm-pack` target | Priority |
|-----------|---------|--------------------|----------|
| `browser-worker` | Browser Web Worker | `web` | `p0` |
| `service-worker` | Service Worker | `web` | `p0` |
| `node-edge` | Node edge process | `nodejs` | `p1` |
| `worker-runtime` | Worker runtime | `web` | `p1` |
| `wasmtime-host` | Wasmtime host | `wasm32-unknown-unknown` | `p2` |

## Host Contract

The host must feed monotonic token order and a finite score in `[0.0, 1.0]`:

```ts
import init, { WasmStreamingKernel } from "./pkg/backfire_wasm.js";

await init();
const kernel = new WasmStreamingKernel(JSON.stringify(config));

for (const token of tokens) {
  const score = await scoreToken(token);
  const session = kernel.process_token(token, score);
  if (session.halted) break;
}
```

For browser use, run scoring and `process_token` in a Web Worker so the UI
thread does not stall on long streams.

## Links

- [Runtime Boundaries](../guide/runtime-boundaries.md)
- [Streaming Halt](../cookbook/streaming-halt-guide.md)
- [Rust FFI](../guide/rust-ffi.md)
- `backfire-kernel/crates/backfire-wasm/README.md`
