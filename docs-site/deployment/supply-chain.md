<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — supply-chain deployment notes
-->

# Supply-Chain Notes

The base package stays small. The heavier optional paths pull native runtimes,
model loaders, vector stores, or local web UI packages. Treat those extras as
deployment choices rather than default onboarding requirements.

`requirements/heavy_optional_dependency_policy.toml` is the tracked source for
this page. It records the package, owning extra, risk class, and required
controls.

## Required Controls

| Control | Rule |
|---------|------|
| `upper-bound` | Top-level optional packages keep a major cap in `pyproject.toml`. |
| `uv-lock` | `uv.lock` remains the resolved graph for repository installs. |
| `runtime-isolation` | Heavy extras run in a selected image, venv, or service tier. |
| `sbom` | Release builds keep SBOM output for deployed stacks. |
| `audit` | CI keeps dependency audit and static checks enabled. |
| `fallback` | Operators keep a rules, heuristic, or Python-only path available. |
| `hash-pin` | Export-only wheels use `--require-hashes`. |
| `isolated-build-stage` | Export-only tooling stays out of runtime images. |

## Package Notes

| Package | Extra | Risk class | Required controls |
|---------|-------|------------|-------------------|
| `torch` | `[nli]` | native-code | upper-bound, uv-lock, runtime-isolation, audit, fallback |
| `transformers` | `[nli]` | model-loader | upper-bound, uv-lock, runtime-isolation, audit, fallback |
| `onnxruntime` | `[onnx]` | native-runtime | upper-bound, uv-lock, runtime-isolation, audit, fallback |
| `onnx` | `[onnx]` | model-graph-serialization | upper-bound, uv-lock, runtime-isolation, audit, fallback |
| `onnxruntime-gpu` | `[tensorrt]` | native-gpu-runtime | upper-bound, uv-lock, runtime-isolation, audit, fallback |
| `mujoco` | `[physical]` | native-simulation-runtime | upper-bound, uv-lock, runtime-isolation, audit, fallback |
| `chromadb` | `[vector]` | local-store | upper-bound, uv-lock, runtime-isolation, audit, fallback |
| `sentence-transformers` | `[vector]` | embedding-model-loader | upper-bound, uv-lock, runtime-isolation, audit, fallback |
| `gradio` | `[ui]` | web-ui | upper-bound, uv-lock, runtime-isolation, audit |

## External Runtimes

Some supported adapters are installed outside PyPI. Keep them out of the base
API image and run them behind a selected boundary:

| Runtime | Source | Risk class | Required controls |
|---------|--------|------------|-------------------|
| `rclpy` | ROS 2 distribution | robotics-middleware | runtime-isolation, audit, fallback |
| `carla` | CARLA vendor package | native-simulation-runtime | runtime-isolation, audit, fallback |
| `arkworks` | operator-supplied zk adapter | proof-backend | runtime-isolation, audit, fallback |
| `gnark` | operator-supplied zk adapter | proof-backend | runtime-isolation, audit, fallback |
| `snarkjs` | operator-supplied zk adapter | proof-backend | runtime-isolation, audit, fallback |

For proof adapters, pin the prover, verifier, circuit artefacts, and proving
key by immutable release or digest in the adapter service manifest. Treat a
circuit change as a protocol migration: reject passports made under an
unknown circuit id and keep the commitment backend available as a fallback.

For physical adapters, pin the simulator package, world assets, robot model
files, and driver container together. Run the simulator or robotics bridge in
its own service account, cap CPU and memory, and expose only the local action
gateway needed by `GroundingHook`.

## Export-Only Tooling

ONNX export uses the same current `torch` and `transformers` major lines as
runtime NLI. The build-time wheel file is limited to graph/runtime packages
that are safe to keep hash-pinned outside the default service image:

| Package | File | Controls |
|---------|------|----------|
| `onnx` | `requirements/docker-gpu-export.txt` | hash-pin, isolated-build-stage |
| `onnxruntime` | `requirements/docker-gpu-export.txt` | hash-pin, isolated-build-stage |

Legacy exporter packages that force `transformers<5` are not installed. That
keeps the model-loader stack on the audited `transformers>=5.0.0rc3,<6` line.
`Dockerfile.gpu` installs the export wheels only in the model-builder stage,
exports the ONNX directory, then copies the artefact into the runtime stage.

## Deployment Guidance

Start with the Python-only path, then add one heavy extra at a time:

```bash
uv sync --locked --extra server --extra vector
uv sync --locked --extra nli
uv sync --locked --extra onnx
uv sync --locked --extra physical
```

For GPU serving, use a separate image or venv so native runtime drift does not
affect the base service. For local vector stores, bind persistence to a chosen
directory and keep tenant separation at the application layer.

When a heavy runtime is unavailable, keep one fallback active:

- `use_model=False` for heuristic-only scoring.
- `director-ai[server,vector]` for Python-only API plus local Chroma.
- WASM halt kernel for browser or offline hosts where the host owns scoring.

## Related Files

- `requirements/uv_extra_lock_policy.toml`
- `requirements/OPTIONAL_EXTRA_LOCKS.md`
- `requirements/onnx_wheel_targets.toml`
- `requirements/docker-gpu-export.txt`
- `SECURITY.md`
