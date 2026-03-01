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

This uses [Optimum](https://huggingface.co/docs/optimum) to convert the
PyTorch model + tokenizer to ONNX format. Requires `pip install optimum[onnxruntime]`.

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

| Provider | Env / Condition | Latency |
|----------|----------------|---------|
| TensorrtExecutionProvider | `DIRECTOR_ENABLE_TRT=1` + libnvinfer | Sub-10 ms target |
| CUDAExecutionProvider | `onnxruntime-gpu` installed | 14.6 ms/pair (GTX 1060) |
| CPUExecutionProvider | Fallback | 383 ms/pair |

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

For reproducible exports:

```
onnxruntime-gpu==1.19.2
optimum==1.23.1
torch>=2.2.0
transformers>=4.40.0
```
