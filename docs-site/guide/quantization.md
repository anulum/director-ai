# Quantization Guide

## 8-bit quantization (bitsandbytes)

Director-AI supports 8-bit quantization via `bitsandbytes` for PyTorch
backends. This halves VRAM usage with minimal accuracy loss.

```python
from director_ai import CoherenceScorer

scorer = CoherenceScorer(
    use_nli=True,
    nli_quantize_8bit=True,
    nli_device="cuda",
)
```

Requires: `pip install director-ai[quantize]` (installs `bitsandbytes`).

## Performance trade-offs

Measured on GTX 1060 6GB, 16-pair batch, FactCG-DeBERTa-v3-Large:

| Backend | Precision | VRAM | Latency/pair | Bal. Acc |
|---------|-----------|------|-------------|----------|
| ONNX GPU (FP32) | FP32 | 1.2 GB | 14.6 ms | 75.8% |
| PyTorch GPU (FP32) | FP32 | 1.4 GB | 19.0 ms | 75.8% |
| PyTorch GPU (FP16) | FP16 | 0.7 GB | ~10 ms | 75.8% |
| PyTorch GPU (INT8) | INT8 | ~0.4 GB | ~8 ms | ~75.1% |

!!! note
    ONNX Runtime does not use bitsandbytes. For ONNX quantization, use
    `optimum` static quantization or ONNX Runtime's built-in quantizer.

## FP16 (half precision)

No extra dependencies. Supported on GPUs with compute capability >= 7.0
(Volta and newer):

```python
scorer = CoherenceScorer(
    use_nli=True,
    nli_torch_dtype="float16",
    nli_device="cuda",
)
```

GTX 1060 (compute 6.1) auto-skips FP16 — the scorer falls back to FP32.

## When to quantize

- **INT8**: CPU-only deployments where latency > 200 ms is acceptable
  but VRAM is constrained.
- **FP16**: Default for GPU deployments. Free speedup on Volta+.
- **ONNX GPU**: Fastest path overall. Prefer this over PyTorch quantization
  when `onnxruntime-gpu` is available.

## TensorRT (experimental)

For sub-10 ms inference on Ada/Hopper GPUs:

```bash
export DIRECTOR_ENABLE_TRT=1
```

Requires `libnvinfer` (NVIDIA TensorRT runtime). The scorer auto-detects
TensorRT availability and builds an engine cache in
`<onnx_path>/trt_cache/`.
