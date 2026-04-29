<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — ONNX artefact deployment guide
-->

# ONNX Artefacts

Use ONNX when the deployment needs local NLI scoring without loading the
PyTorch backend at request time. The Python-only quickstart still remains the
default path; ONNX is an opt-in runtime path.

## Export Path

Install the runtime stack, add the hash-pinned export wheels from the repo, and
write the model into the quickstart path:

```bash
pip install "director-ai[nli,onnx]"
pip install --no-deps --require-hashes -r requirements/docker-gpu-export.txt
director-ai export --format onnx --output models/factcg-onnx
```

The GPU Docker build uses the same export wheel file during its model-builder
stage, then copies only the exported model into the runtime image.

The expected directory is:

```text
models/factcg-onnx/
  model.onnx
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

For CPU-only serving, install the ONNX Runtime extra:

```bash
pip install "director-ai[nli,onnx]"
```

For NVIDIA CUDA serving, install the GPU runtime extra:

```bash
pip install "director-ai[nli,tensorrt]"
```

TensorRT execution also requires matching NVIDIA TensorRT libraries on the
host or image. The package extra supplies `onnxruntime-gpu`; it does not vendor
driver or TensorRT system libraries.

## Quickstart Compose

`director-ai quickstart` creates `models/factcg-onnx/` and an opt-in Compose
profile. Export or copy the files into that directory, then run:

```bash
docker compose --profile onnx up director-proxy-onnx
```

The container mounts the model directory read-only and sets:

```bash
DIRECTOR_SCORER_BACKEND=onnx
DIRECTOR_ONNX_PATH=/models/factcg-onnx
```

## Wheel Target Matrix

`requirements/onnx_wheel_targets.toml` is the tracked source for these targets.
`requirements/docker-gpu-export.txt` pins the export-only wheels.

| Target id | Platform | Extra | Runtime package | Execution provider | Status |
|-----------|----------|-------|-----------------|--------------------|--------|
| `linux-cpu-x86_64` | Linux x86_64 | `[onnx]` | `onnxruntime` | `CPUExecutionProvider` | covered |
| `linux-nvidia-cuda-x86_64` | Linux x86_64 with NVIDIA driver | `[tensorrt]` | `onnxruntime-gpu` | `CUDAExecutionProvider` | covered |
| `linux-nvidia-tensorrt-x86_64` | Linux x86_64 with NVIDIA TensorRT libraries | `[tensorrt]` | `onnxruntime-gpu` | `TensorrtExecutionProvider` | covered with system TensorRT libraries |
| `macos-cpu-arm64` | macOS arm64 | `[onnx]` | `onnxruntime` | `CPUExecutionProvider` | covered |
| `windows-cpu-amd64` | Windows amd64 | `[onnx]` | `onnxruntime` | `CPUExecutionProvider` | covered |

ROCm, DirectML, and other provider wheels are not exposed as project extras
until their install path is verified in CI. Use CPU ONNX on those hosts unless
a deployment owns the provider wheel and system library stack.

## Runtime Check

After installing the target extra, verify the provider list:

```bash
director-ai doctor
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The expected provider should appear before the CPU fallback for GPU targets.
If it does not, keep the deployment on `[onnx]` until the driver and runtime
libraries match.
