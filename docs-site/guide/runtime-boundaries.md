<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-Class AI - Runtime Boundary Guide -->

# Runtime Boundaries

Director-AI has one supported default path and several optional runtime paths.
The default path is intentionally Python-only so new users, contributors, and
small deployments can get a working guard without installing the full research
and acceleration stack.

## Supported Default

Use this path first:

```bash
pip install director-ai[server,vector]
director-ai quickstart --run
director-ai doctor
```

This path includes:

| Component | Status | Purpose |
|-----------|--------|---------|
| Python package | Supported default | CLI, scorer, config, server, proxy |
| FastAPI service | Supported default | HTTP API and local deployment surface |
| Local Chroma persistence | Supported default | Simple persistent vector store |
| Fact file ingestion | Supported default | One-file knowledge base bootstrap |
| `director-ai doctor` | Supported default | Runtime stack audit and config warnings |

The default must not require Rust, Go, Julia, Lean, TensorRT, or WASM.

## Optional Runtime Paths

Enable these only when the deployment needs the specific capability.

| Runtime | Enable When | Entry Point | Support Boundary |
|---------|-------------|-------------|------------------|
| Rust kernel | Low-latency local scoring is required | `pip install director-ai[rust]` | Optional acceleration; Python fallback remains supported |
| ONNX | Local NLI inference is required | `pip install director-ai[nli,onnx]` | Optional model runtime; requires exported model artefacts |
| TensorRT | NVIDIA GPU throughput is required | `pip install director-ai[tensorrt]` | Optional GPU optimisation path |
| Go gateway | Dedicated gateway/risk-router deployment is required | gateway binary or container | Enterprise/research deployment path |
| Julia tuner | Research threshold sweeps are required | Julia scripts | Research path, not onboarding |
| Lean verifier | Formal proof checks are required | Lean backend | Research/formal-verification path |
| WASM | Browser or edge halt checks are required | WASM package | Deferred edge/offline path |

## Contributor Boundary

Good first issues and routine fixes should stay in the Python-only surface:

- `src/director_ai/`
- `tests/`
- `docs-site/`
- CLI, config, server, proxy, scorer, and ingestion code

Changes to Rust, Go, Julia, Lean, TensorRT, or WASM should be labelled as
advanced runtime work and should include contract tests against the Python
path. Optional runtime changes must not make the default install heavier.

## Operational Boundary

Production operators should choose one runtime stack per deployment:

| Deployment Need | Recommended Stack |
|-----------------|-------------------|
| First production trial | Python + FastAPI + Chroma |
| Local NLI scoring | Python + FastAPI + Chroma + ONNX |
| GPU throughput | Python + ONNX/TensorRT container |
| Gateway risk routing | Python API + Go gateway |
| Browser/edge halt checks | WASM once the runtime is published |
| Formal proof checks | Python verifier + Lean backend |

Run `director-ai doctor` after setting `DIRECTOR_*` environment variables. It
warns when config requests optional components that are not installed, for
example `DIRECTOR_SCORER_BACKEND=onnx` without `onnxruntime` or an ONNX path.

## Links

- [Installation](../installation.md)
- [Quickstart](../quickstart.md)
- [Docker Deployment](../deployment/docker.md)
- [Rust FFI](rust-ffi.md)
- [Rust Acceleration](rust-acceleration.md)
- [Streaming Halt WASM Notes](../cookbook/streaming-halt-guide.md)
