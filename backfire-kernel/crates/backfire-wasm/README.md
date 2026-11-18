<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI — WebAssembly streaming halt runtime -->

# backfire-wasm

WebAssembly bindings for the Director-Class AI streaming halt
kernel. Deploys the safety check to any runtime that can load a
WASM module with its generated JavaScript bindings, such as browsers or Node.

The WASM path provides **halt decisions only**: the kernel
decides whether the next token clears `hard_limit`, fires window
and trend checks, and returns the updated session. The host (browser, edge
function, Node process) must produce a coherence score per token and feed it
to the kernel. The bundled Python/Rust stack in the repo keeps
the full scoring path; this crate contains no model or retrieval backend.

## Layout

```
backfire-kernel/crates/backfire-wasm/
├── Cargo.toml          # cdylib + rlib, wasm-bindgen-test dev dep
├── src/lib.rs          # WasmStreamingKernel JS-facing API
├── tests/kernel.rs     # real Node/WASM integration tests
├── example/index.html  # minimal browser demo
└── pkg/                # wasm-pack build output (gitignored)
```

## Building

Use wasm-pack 0.15.1. From the repository root:

```bash
make wasm-build       # web module in pkg/
make wasm-build-node  # Node module in pkg-nodejs/
```

The separate Node output preserves an existing web package. Direct crate
commands are equivalent:

```bash
wasm-pack build --target web --release -- --locked
wasm-pack build --target nodejs --release --out-dir pkg-nodejs -- --locked
```

Set `CARGO_TARGET_DIR` to an owned build directory when needed. The Make
targets otherwise use the repository's `backfire-kernel/target/`. Release
builds use wasm-pack's wasm-opt step; package size depends on the toolchain
and target, so retain the generated artefact identities for each build.

## Testing

```bash
wasm-pack test --node -- --locked
```

The integration tests in `tests/kernel.rs` execute a real WASM build under Node.
They cover construction, canonical configuration validation, missing fields,
malformed JSON, all three halt reasons and their precedence, strict threshold
boundaries, window eviction, recent-history trend checks, nonfinite scores,
independent snapshots and permanent halt state. A host-target `cargo test`
does not execute these `wasm32` tests.

## API

```ts
import init, { WasmStreamingKernel } from "./pkg/backfire_wasm.js";
await init();

const config = JSON.stringify({
    coherence_threshold: 0.6,
    hard_limit: 0.5,
    soft_limit: 0.7,
    w_logic: 0.6,
    w_fact: 0.4,
    window_size: 10,
    window_threshold: 0.55,
    trend_window: 5,
    trend_threshold: 0.15,
    history_window: 5,
    deadline_ms: 50,
    logit_entropy_limit: 1.2,
});
const kernel = new WasmStreamingKernel(config);

try {
    for (const token of tokens) {
        const score = await coherenceFromEdgeNli(token); // host-supplied
        const session = kernel.process_token(token, score);
        if (session.halted) {
            console.log("halted:", session.halt_reason);
            break;
        }
    }
} finally {
    kernel.free();
}
```

`process_token` returns the full session object (serialised from
the Rust `StreamSession`) so the host can decide what to render.
`kernel.is_active()` is a cheap getter for the same flag.

The constructor requires the complete `BackfireConfig` shown above and applies
the native configuration validator before allocating the stream window.
`{threshold: 0.65}` is not that configuration. A malformed or invalid
configuration throws before an instance is returned.

Hosts should supply finite coherence scores in `[0, 1]`. As in the native Rust
streaming kernel, NaN and either infinity become zero before recording the
token event or making a halt decision. Finite values are not clamped. The hard
limit uses `<`, the window average uses `<`, and the downward drop uses `>`;
equality alone does not trigger that check. A zero score can remain active
when the configured limits allow it.

The downward trend is the projected drop `-slope * (N - 1)` from a least-squares
fit to the recent `N = trend_window` scores. Rust, WASM and the Python streaming
paths use the same native regression function.

The token that triggers a halt is included in the trace. After any halt,
further `process_token` calls leave the original session unchanged. Create a
new kernel for each stream, then call `free()` when finished. Returned snapshots
are independent JavaScript values; event timestamps are zero because this
crate does not implement a host clock.

## Hosting the browser demo

```bash
wasm-pack build --target web --release -- --locked
cd example
python3 -m http.server 8000
# open http://localhost:8000 in a browser
```

`example/index.html` is a self-contained page — no bundler, no
framework. Useful as a starting template when embedding the
kernel into an existing front-end.

## Publishing

The generated `pkg/package.json` has `name: "backfire-wasm"`. Push
to npm with `wasm-pack publish --access public` once the upstream
NLI edge runtime is ready. The crate does **not** auto-publish on
version bumps — the publish step is operator-triggered.

## Known limitations

- **No scoring model.** The kernel expects a coherence score per
  token. Edge deployments typically run a distilled or quantised
  NLI model via Transformers.js / ONNX-WebRuntime and feed the
  score in from JavaScript.
- **No factuality guarantee.** These generic coherence checks do not implement
  the Python evidence-bound completed-claim contradiction interlock. Retrieval,
  calibration and the meaning of a supplied score remain host responsibilities.
- **JavaScript host bindings.** The generated web and Node packages use
  `wasm-bindgen`; the raw module is not a standalone WASI executable.
- **No OpenTelemetry bridge.** OTEL spans are the Python host's
  concern. JavaScript callers that want token-level traces should
  set up `@opentelemetry/api` on their side and instrument
  `process_token` at the JS layer.
- **Single-threaded.** WASM on the browser main thread blocks on
  CPU work. For long sessions, host the kernel in a Web Worker.

## Not in this crate

- Python bindings (those are in `backfire-ffi` via PyO3).
- GPU acceleration — WASM SIMD is enough for the halt checks;
  the dense NLI model is not shipped here.
- Tenant routing, auth, audit — those remain in the Python or Go
  stacks that front the deployment.
