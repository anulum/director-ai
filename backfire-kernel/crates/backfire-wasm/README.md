# backfire-wasm

WebAssembly bindings for the Director-Class AI streaming halt
kernel. Deploys the safety check to any runtime that can load a
WASM module — browsers, Cloudflare Workers, Deno, Node, Wasmtime.

The WASM path is intended for **edge scoring only**: the kernel
decides whether the next token clears `hard_limit`, fires window
and trend checks, and returns the updated session. The actual NLI
coherence number must come from somewhere — the host (browser, edge
function, Node process) is responsible for producing one per token
and feeding it in. The bundled Python/Rust stack in the repo keeps
the full scoring path; this crate is the minimum viable halt
check for environments that cannot run PyTorch.

## Layout

```
backfire-kernel/crates/backfire-wasm/
├── Cargo.toml          # cdylib + rlib, wasm-bindgen-test dev dep
├── src/lib.rs          # WasmStreamingKernel JS-facing API
├── tests/kernel.rs     # wasm-bindgen-test — 7 node-executable cases
├── example/index.html  # minimal browser demo
└── pkg/                # wasm-pack build output (gitignored)
```

## Building

```bash
# from the crate directory
wasm-pack build --target web --release

# other targets
wasm-pack build --target nodejs --release
wasm-pack build --target bundler --release     # Webpack/Vite
wasm-pack build --target no-modules --release  # classic <script>
```

The default artefact weighs ~110 KB uncompressed (`*.wasm`) plus
~12 KB of JS glue. `wasm-opt -Oz` is run by `wasm-pack` during a
release build.

Cargo cache lives on NTFS per the project convention — set
`CARGO_TARGET_DIR=/media/anulum/724AA8E84AA8AA75/linux_data/rust-target`
before running to avoid hammering the root partition.

## Testing

```bash
wasm-pack test --node
```

The seven integration tests in `tests/kernel.rs` run against a real
WASM build under Node: construction, all-pass path, sub-threshold
hard halt, window-average halt, downward-trend halt, irrevocable
halt, and malformed config rejection.

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

for (const token of tokens) {
    const score = await coherenceFromEdgeNli(token); // host-supplied
    const session = kernel.process_token(token, score);
    if (session.halted) {
        console.log("halted:", session.halt_reason);
        break;
    }
}
```

`process_token` returns the full session object (serialised from
the Rust `StreamSession`) so the host can decide what to render.
`kernel.is_active()` is a cheap getter for the same flag.

## Hosting the browser demo

```bash
wasm-pack build --target web --release
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
