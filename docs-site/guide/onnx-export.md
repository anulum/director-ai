<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-AI — ONNX export guide
-->

# ONNX Export & Custom Models

## Export FactCG to ONNX

Director-AI ships with `export_onnx()` in `director_ai.core.nli`:

```python
from director_ai.core.nli import export_onnx

export_onnx(
    model_name="yaxili96/FactCG-DeBERTa-v3-Large",
    output_dir="models/factcg_onnx",
)
```

This uses `torch.onnx.export` to convert the PyTorch model + tokenizer to ONNX
format while staying on the same audited `transformers>=5.0.0rc3,<6` model
loader line as runtime NLI. The export-only graph/runtime wheels are pinned in
`requirements/docker-gpu-export.txt`.

For deployable directory layout and wheel target coverage, see
[ONNX Artefacts](../deployment/onnx-artefacts.md).

## Use the ONNX model

```python
from director_ai.core.nli import NLIScorer

scorer = NLIScorer(
    backend="onnx",
    onnx_path="models/factcg_onnx",
    device="cuda",  # or "cpu"
)
```

The ONNX backend selects execution providers automatically:

| Provider | Env / Condition | Runtime requirement |
|----------|----------------|---------|
| TensorrtExecutionProvider | `DIRECTOR_ENABLE_TRT=1` + libnvinfer | Matching TensorRT libraries |
| CUDAExecutionProvider | `onnxruntime-gpu` installed | Matching NVIDIA driver and CUDA libraries |
| CPUExecutionProvider | Fallback | CPU ONNX Runtime |

To pre-build the TensorRT engine cache for a local ONNX export, point the CLI at
the ONNX directory explicitly:

```bash
director-ai export --format tensorrt \
  --onnx-dir models/factcg_onnx \
  --output models/factcg_onnx/trt_cache
```

The export rejects non-positive `max_batch`, `max_seq_len`, and `warmup_pairs`
profile values before it touches ONNX Runtime, which keeps configuration errors
separate from optional GPU-provider availability.

## GPU Docker image (pre-exported)

The `Dockerfile.gpu` multi-stage build exports the model at build time:

```bash
docker build -f Dockerfile.gpu -t director-ai:gpu .
docker run --gpus all -p 8080:8080 director-ai:gpu
```

The ONNX model is baked into `/app/models/onnx/` — no HuggingFace
downloads at runtime.

## Custom models

Any HuggingFace `AutoModelForSequenceClassification` works:

```python
from director_ai.core.nli import export_onnx, NLIScorer

# Export your fine-tuned model
export_onnx(
    model_name="your-org/your-nli-model",
    output_dir="models/custom_onnx",
)

# Use it
scorer = NLIScorer(
    backend="onnx",
    onnx_path="models/custom_onnx",
)
```

The scorer auto-detects 2-class (FactCG-style) vs 3-class (standard NLI)
models and adjusts the entailment probability extraction.

## Graph optimization

Set `ORT_ENABLE_ALL=1` (default in Director-AI) for operator fusion,
constant folding, and layout optimization. This is already configured
in the `NLIScorer` ONNX path.

## Pinned dependencies

Install the selected runtime from the repository lock, then add the export
wheels resolved against the same runtime:

```bash
uv sync --locked --extra nli --extra onnx
uv pip install --no-deps --require-hashes -r requirements/docker-gpu-export.txt
```

`requirements/docker-gpu-export.in` declares the exporter packages;
`requirements/docker-gpu-export.txt` records their native resolver output and
hashes. It includes FlatBuffers, an ONNX Runtime dependency absent from the
base NLI runtime. Shared packages keep compatible versions, including
`protobuf<7`; no legacy `optimum` pin is needed by the direct PyTorch exporter.
Measure latency on the selected model, provider, batch size and hardware.
