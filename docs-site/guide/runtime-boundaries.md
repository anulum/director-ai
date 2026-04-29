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
| WASM | Browser or edge halt checks are required | `make wasm-build` / `backfire-wasm` | Prioritised edge/offline path; host supplies scores |

For package extras and common combinations, use the
[advanced backend matrix](../installation.md#advanced-backend-matrix).

## Research Hook Boundary

Research hooks are not part of the supported default path. Load them through
the explicit namespace and gate:

```python
from director_ai import experimental

experimental.enable_experimental_hooks()
trajectory = experimental.load_hook("trajectory")
```

Non-interactive deployments may use:

```bash
export DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS=1
```

This boundary covers trajectory simulation, trace-safe checks, causal
verification, irreversibility forecasting, symbolic and ontology checks,
multimodal guards, self-monitoring/adaptation modules, swarm risk modules,
provenance/privacy research backends, and sustainability policy modules. Direct
`director_ai.core.<module>` imports remain for compatibility and internal tests,
but new public examples should use `director_ai.experimental`.

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
| Browser/edge halt checks | WASM halt kernel + host-owned scorer |
| Formal proof checks | Python verifier + Lean backend |

Run `director-ai doctor` after setting `DIRECTOR_*` environment variables. It
warns when config requests optional components that are not installed, for
example `DIRECTOR_SCORER_BACKEND=onnx` without `onnxruntime` or an ONNX path.

## Links

- [Installation](../installation.md)
- [Quickstart](../quickstart.md)
- [Docker Deployment](../deployment/docker.md)
- [Runtime Extraction Evaluation](runtime-extraction-evaluation.md)
- [Rust FFI](rust-ffi.md)
- [Rust Acceleration](rust-acceleration.md)
- [WASM Runtime](../deployment/wasm-runtime.md)
- [Streaming Halt WASM Notes](../cookbook/streaming-halt-guide.md)
