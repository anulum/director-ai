# ---------------------------------------------------------------------
# Director-Class AI — Cross-GPU Latency Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ---------------------------------------------------------------------
"""
Cross-GPU benchmark: PyTorch FP32/FP16, ONNX CUDA, ONNX TensorRT FP32/FP16.
Auto-detects available backends and skips unsupported ones.

Usage:
    python -m benchmarks.gpu_bench
    python -m benchmarks.gpu_bench --onnx-path benchmarks/results/factcg_onnx
    python -m benchmarks.gpu_bench --backends onnx_trt_fp16,pytorch_fp16
    python -m benchmarks.gpu_bench --batch-sizes 1,8,16,32 --iterations 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from .latency_bench import BENCH_PAIRS

ALL_BACKENDS = [
    "pytorch_fp32",
    "pytorch_fp16",
    "onnx_cuda",
    "onnx_trt_fp32",
    "onnx_trt_fp16",
]


def _gpu_info() -> dict:
    """Collect GPU metadata via torch.cuda."""
    import torch

    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / (1024**3), 1),
        "cuda_version": torch.version.cuda or "N/A",
    }


def _gpu_slug(name: str) -> str:
    """GPU name → filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _compute_capability() -> float:
    """Return GPU compute capability as float (e.g. 8.0), or 0.0 if no GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return props.major + props.minor * 0.1
    except ImportError:
        return 0.0


def _detect_backends() -> list[str]:
    """Return list of backend IDs the current hardware supports."""
    available = []
    cc = _compute_capability()

    try:
        import torch

        if torch.cuda.is_available():
            available.append("pytorch_fp32")
            if cc >= 7.0:
                available.append("pytorch_fp16")
    except ImportError:
        pass

    try:
        import onnxruntime as ort

        provs = ort.get_available_providers()
        if "CUDAExecutionProvider" in provs:
            available.append("onnx_cuda")
        if "TensorrtExecutionProvider" in provs:
            available.append("onnx_trt_fp32")
            if cc >= 7.0:
                available.append("onnx_trt_fp16")
    except ImportError:
        pass

    return available


def _make_batch(batch_size: int) -> list[tuple[str, str]]:
    """Replicate BENCH_PAIRS to fill batch_size."""
    n = len(BENCH_PAIRS)
    if batch_size <= n:
        return BENCH_PAIRS[:batch_size]
    reps = (batch_size // n) + 1
    return (BENCH_PAIRS * reps)[:batch_size]


# ── Backend runners ──────────────────────────────────────────────


def _run_pytorch(
    pairs: list[tuple[str, str]],
    iterations: int,
    warmup: int,
    fp16: bool = False,
) -> list[float]:
    """Batched PyTorch inference, returns list of total-batch times (ms)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from director_ai.core.nli import _FACTCG_TEMPLATE, _DEFAULT_MODEL

    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(_DEFAULT_MODEL)
    model = model.to("cuda").eval()

    dtype = torch.float16 if fp16 else torch.float32
    if fp16:
        model = model.half()

    texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=512,
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    for _ in range(warmup):
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=fp16):
            model(**inputs)
        torch.cuda.synchronize()

    times: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=fp16):
            model(**inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return times


def _run_onnx(
    pairs: list[tuple[str, str]],
    iterations: int,
    warmup: int,
    onnx_path: str,
    provider: str,
    trt_fp16: bool = False,
) -> list[float]:
    """Batched ONNX Runtime inference, returns list of total-batch times (ms)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from director_ai.core.nli import _FACTCG_TEMPLATE

    tokenizer = AutoTokenizer.from_pretrained(onnx_path)

    model_file = str(Path(onnx_path) / "model.onnx")
    if not Path(model_file).exists():
        for f in Path(onnx_path).iterdir():
            if f.suffix == ".onnx":
                model_file = str(f)
                break

    providers: list = []
    if provider == "TensorrtExecutionProvider":
        trt_opts = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(Path(onnx_path) / "trt_cache"),
            "trt_fp16_enable": trt_fp16,
        }
        providers.append(("TensorrtExecutionProvider", trt_opts))
    providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    session = ort.InferenceSession(model_file, opts, providers=providers)
    active = session.get_providers()[0]
    print(f"    active provider: {active}")

    if provider == "TensorrtExecutionProvider" and active != "TensorrtExecutionProvider":
        raise RuntimeError(f"TRT requested but got {active} (libnvinfer not installed?)")

    texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
    inputs = tokenizer(
        texts, return_tensors="np", truncation=True, padding=True, max_length=512,
    )
    expected = {i.name for i in session.get_inputs()}
    feed = {
        k: v.astype(np.int64) if v.dtype != np.int64 else v
        for k, v in inputs.items()
        if k in expected
    }

    for _ in range(warmup):
        session.run(None, feed)

    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)

    return times


# ── Stats ────────────────────────────────────────────────────────


def _stats(times: list[float], batch_size: int) -> dict:
    a = np.array(times)
    return {
        "mean_ms": round(float(np.mean(a)), 2),
        "median_ms": round(float(np.median(a)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "p99_ms": round(float(np.percentile(a, 99)), 2),
        "std_ms": round(float(np.std(a)), 2),
        "per_pair_ms": round(float(np.median(a)) / batch_size, 2),
    }


def _gpu_mem_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    except ImportError:
        pass
    return 0.0


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Director-AI cross-GPU benchmark")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--onnx-path", type=str,
        default=str(Path(__file__).parent / "results" / "factcg_onnx"),
    )
    parser.add_argument(
        "--backends", type=str, default=None,
        help=f"Comma-separated subset of: {','.join(ALL_BACKENDS)}",
    )
    parser.add_argument(
        "--batch-sizes", type=str, default="1,8,16,32",
        help="Comma-separated batch sizes (default: 1,8,16,32)",
    )
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    detected = _detect_backends()
    if args.backends:
        requested = [b.strip() for b in args.backends.split(",")]
        backends = [b for b in requested if b in detected]
        skipped = [b for b in requested if b not in detected]
        if skipped:
            print(f"Skipping unavailable: {', '.join(skipped)}")
    else:
        backends = detected

    if not backends:
        print("No GPU backends available. Need CUDA-capable GPU + torch/onnxruntime-gpu.")
        sys.exit(1)

    info = _gpu_info()
    print("\n" + "=" * 72)
    print("Director-AI Cross-GPU Benchmark")
    print(f"  GPU: {info.get('gpu', 'N/A')}")
    print(f"  Compute: {info.get('compute_capability', 'N/A')}")
    print(f"  VRAM: {info.get('vram_gb', 'N/A')} GB")
    print(f"  CUDA: {info.get('cuda_version', 'N/A')}")
    print(f"  Backends: {', '.join(backends)}")
    print(f"  Batches: {batch_sizes}")
    print(f"  Iterations: {args.iterations}, warmup: {args.warmup}")

    try:
        import torch
        info["torch_version"] = torch.__version__
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        info["onnxruntime_version"] = ort.__version__
    except ImportError:
        pass

    print("=" * 72)

    all_results = []

    for backend in backends:
        for bs in batch_sizes:
            pairs = _make_batch(bs)
            label = f"{backend} batch={bs}"
            print(f"\n  [{label}]", flush=True)

            try:
                if backend == "pytorch_fp32":
                    times = _run_pytorch(pairs, args.iterations, args.warmup, fp16=False)
                elif backend == "pytorch_fp16":
                    times = _run_pytorch(pairs, args.iterations, args.warmup, fp16=True)
                elif backend == "onnx_cuda":
                    times = _run_onnx(
                        pairs, args.iterations, args.warmup,
                        args.onnx_path, "CUDAExecutionProvider",
                    )
                elif backend == "onnx_trt_fp32":
                    times = _run_onnx(
                        pairs, args.iterations, args.warmup,
                        args.onnx_path, "TensorrtExecutionProvider", trt_fp16=False,
                    )
                elif backend == "onnx_trt_fp16":
                    times = _run_onnx(
                        pairs, args.iterations, args.warmup,
                        args.onnx_path, "TensorrtExecutionProvider", trt_fp16=True,
                    )
                else:
                    print(f"    unknown backend: {backend}")
                    continue
            except Exception as exc:
                print(f"    FAILED: {exc}")
                continue

            s = _stats(times, bs)
            s["backend"] = backend
            s["batch_size"] = bs
            s["gpu_mem_mb"] = _gpu_mem_mb()
            all_results.append(s)

            print(
                f"    median={s['median_ms']:.1f} ms  "
                f"per_pair={s['per_pair_ms']:.1f} ms  "
                f"p95={s['p95_ms']:.1f} ms  "
                f"mem={s['gpu_mem_mb']:.0f} MB"
            )

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  {'Backend':20s} {'Batch':>5s} {'Median':>9s} {'Per-pair':>9s} {'P95':>9s} {'Mem':>8s}")
    print(f"  {'-' * 62}")
    for r in all_results:
        print(
            f"  {r['backend']:20s} {r['batch_size']:5d} "
            f"{r['median_ms']:8.1f}ms {r['per_pair_ms']:8.1f}ms "
            f"{r['p95_ms']:8.1f}ms {r['gpu_mem_mb']:7.0f}MB"
        )
    print(f"{'=' * 72}\n")

    # ── Save JSON ────────────────────────────────────────────────
    slug = _gpu_slug(info.get("gpu", "unknown"))
    output = {
        **info,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": all_results,
    }
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"gpu_bench_{slug}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
