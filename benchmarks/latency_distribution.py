# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Latency Distribution Visualiser
"""Generate latency distribution histograms with p50/p95/p99 percentile markers.

Supports CPU heuristic, streaming kernel, and GPU NLI (CUDA/ROCm) benchmarks.

Usage:
    python -m benchmarks.latency_distribution --run                    # CPU only
    python -m benchmarks.latency_distribution --run --nli              # + NLI GPU
    python -m benchmarks.latency_distribution --run --nli --device cuda:0
    python -m benchmarks.latency_distribution --run --iterations 500
    python -m benchmarks.latency_distribution                          # from saved JSON

Outputs:
    benchmarks/results/latency_raw.json          — raw times per benchmark
    benchmarks/results/latency_distribution.png   — histogram with percentiles
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .latency_bench import BENCH_PAIRS


def _detect_gpu_backend() -> dict:
    """Detect GPU backend (CUDA or ROCm) and return metadata."""
    try:
        import torch
    except ImportError:
        return {"available": False}

    if not torch.cuda.is_available():
        return {"available": False, "torch": torch.__version__}

    info: dict = {
        "available": True,
        "torch": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }

    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)

    if hip_version:
        info["backend"] = "ROCm"
        info["rocm_version"] = hip_version
    elif cuda_version:
        info["backend"] = "CUDA"
        info["cuda_version"] = cuda_version
    else:
        info["backend"] = "unknown"

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append(
            {
                "index": i,
                "name": props.name,
                "vram_gb": round(props.total_memory / (1024**3), 1),
            }
        )

    return info


def run_fresh_benchmark(iterations: int, warmup: int) -> dict:
    """Run lightweight + streaming benchmarks and return raw times."""
    from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel

    facts = {
        "capital_france": "Paris is the capital of France.",
        "sky_color": "The sky appears blue due to Rayleigh scattering.",
        "water_formula": "Water has the chemical formula H2O.",
    }
    pairs = BENCH_PAIRS[:4]

    store = GroundTruthStore()
    for k, v in facts.items():
        store.add(k, v)
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store, use_nli=False)

    # Warmup
    for _ in range(warmup):
        scorer.review(pairs[0][0], pairs[0][1])

    # Heuristic review
    review_times: list[float] = []
    for i in range(iterations):
        q, r = pairs[i % len(pairs)]
        t0 = time.perf_counter()
        scorer.review(q, r)
        review_times.append((time.perf_counter() - t0) * 1000)

    # Streaming kernel
    scores = [0.9, 0.85, 0.88, 0.92, 0.87, 0.90, 0.86, 0.91, 0.89]
    tokens = [
        "The",
        " sky",
        " is",
        " blue",
        " due",
        " to",
        " Rayleigh",
        " scattering",
        ".",
    ]

    sk = StreamingKernel()
    for _ in range(warmup):
        idx = [0]

        def _cb(_t: str, _idx: list[int] = idx, _s: list[float] = scores) -> float:
            s = _s[_idx[0] % len(_s)]
            _idx[0] += 1
            return s

        sk.stream_tokens(iter(tokens), _cb)

    stream_times: list[float] = []
    for _ in range(iterations):
        sk = StreamingKernel()
        idx = [0]

        def _cb2(_t: str, _idx: list[int] = idx, _s: list[float] = scores) -> float:
            s = _s[_idx[0] % len(_s)]
            _idx[0] += 1
            return s

        t0 = time.perf_counter()
        sk.stream_tokens(iter(tokens), _cb2)
        stream_times.append((time.perf_counter() - t0) * 1000)

    return {
        "review_no_nli": review_times,
        "streaming_session": stream_times,
    }


def run_nli_benchmark(
    iterations: int,
    warmup: int,
    device: str | None = None,
) -> dict:
    """Run NLI scoring benchmarks on GPU (CUDA or ROCm).

    Returns dict with raw times for sequential and batched scoring.
    """
    from director_ai.core.scoring.nli import NLIScorer

    gpu_info = _detect_gpu_backend()
    backend_name = gpu_info.get("backend", "GPU")
    device_name = ""

    if gpu_info.get("devices"):
        dev_idx = 0
        if device and ":" in device:
            dev_idx = int(device.split(":")[1])
        if dev_idx < len(gpu_info["devices"]):
            device_name = gpu_info["devices"][dev_idx]["name"]

    label_prefix = f"NLI {backend_name}"
    if device_name:
        label_prefix = f"NLI {backend_name} ({device_name})"

    print(f"  Loading NLI model on {device or 'auto'}...", end=" ", flush=True)
    t0 = time.perf_counter()
    nli = NLIScorer(use_model=True, device=device)
    if not nli.model_available:
        print("FAILED (model unavailable)")
        return {}
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"done ({load_ms:.0f} ms)")

    data: dict[str, list[float]] = {}

    # Sequential: score 16 pairs one by one
    print(f"  Benchmarking sequential (16 pairs × {iterations} iters)...")
    for _ in range(warmup):
        for p, h in BENCH_PAIRS:
            nli.score(p, h)

    seq_times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        for p, h in BENCH_PAIRS:
            nli.score(p, h)
        seq_times.append((time.perf_counter() - t0) * 1000)
    data[f"{label_prefix} seq (16 pairs)"] = seq_times

    # Batched: score_batch 16 pairs in one call
    print(f"  Benchmarking batched (16 pairs × {iterations} iters)...")
    for _ in range(warmup):
        nli.score_batch(BENCH_PAIRS)

    batch_times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        nli.score_batch(BENCH_PAIRS)
        batch_times.append((time.perf_counter() - t0) * 1000)
    data[f"{label_prefix} batch (16 pairs)"] = batch_times

    # Per-pair latency (batch / 16)
    per_pair_times = [t / 16 for t in batch_times]
    data[f"{label_prefix} per-pair (from batch)"] = per_pair_times

    # Full scorer pipeline with NLI
    print(f"  Benchmarking full scorer.review() with NLI ({iterations} iters)...")
    from director_ai.core import CoherenceScorer, GroundTruthStore

    store = GroundTruthStore()
    store.add("capital_france", "Paris is the capital of France.")
    store.add("sky_color", "The sky appears blue due to Rayleigh scattering.")

    scorer = CoherenceScorer(
        threshold=0.6,
        ground_truth_store=store,
        use_nli=True,
        nli_device=device,
    )

    for _ in range(warmup):
        scorer.review(BENCH_PAIRS[0][0], BENCH_PAIRS[0][1])

    review_nli_times: list[float] = []
    for i in range(iterations):
        q, r = BENCH_PAIRS[i % len(BENCH_PAIRS)]
        t0 = time.perf_counter()
        scorer.review(q, r)
        review_nli_times.append((time.perf_counter() - t0) * 1000)
    data[f"{label_prefix} scorer.review()"] = review_nli_times

    return data


def plot_distribution(data: dict, output_path: Path) -> None:
    """Generate latency distribution histograms with percentile markers."""
    import matplotlib.pyplot as plt

    n_plots = len(data)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

    axes_flat = [axes] if n_plots == 1 else list(np.array(axes).flat)

    for ax, (name, times) in zip(axes_flat[:n_plots], data.items(), strict=False):
        arr = np.array(times)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean = float(np.mean(arr))

        n_bins = min(80, max(20, len(arr) // 10))
        ax.hist(
            arr,
            bins=n_bins,
            color="#10B981",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        for val, label, colour, ls in [
            (p50, f"p50={p50:.3f}ms", "#3B82F6", "--"),
            (p95, f"p95={p95:.3f}ms", "#F59E0B", "--"),
            (p99, f"p99={p99:.3f}ms", "#DC2626", "-"),
        ]:
            ax.axvline(x=val, color=colour, linestyle=ls, linewidth=1.5, label=label)

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{name}\nn={len(arr)}  mean={mean:.3f}ms  p99={p99:.3f}ms",
            fontweight="bold",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Distribution plot saved to {output_path}")
    plt.close()


def print_percentile_table(data: dict) -> None:
    """Print full percentile table to stdout."""
    print(f"\n{'=' * 80}")
    print("Latency Distribution Report")
    print(f"{'=' * 80}")
    print(
        f"  {'Benchmark':35s}  {'n':>5s}  {'mean':>8s}  {'p50':>8s}  "
        f"{'p90':>8s}  {'p95':>8s}  {'p99':>8s}  {'max':>8s}"
    )
    print(f"  {'-' * 78}")

    for name, times in data.items():
        arr = np.array(times)
        print(
            f"  {name:35s}  {len(arr):5d}  {np.mean(arr):7.3f}ms  "
            f"{np.percentile(arr, 50):7.3f}ms  {np.percentile(arr, 90):7.3f}ms  "
            f"{np.percentile(arr, 95):7.3f}ms  {np.percentile(arr, 99):7.3f}ms  "
            f"{np.max(arr):7.3f}ms"
        )
    print(f"{'=' * 80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency distribution visualiser")
    parser.add_argument("--run", action="store_true", help="Run fresh benchmark")
    parser.add_argument(
        "--nli",
        action="store_true",
        help="Include NLI GPU benchmarks (requires torch + transformers)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for NLI (auto, cpu, cuda, cuda:0, cuda:1, etc.)",
    )
    parser.add_argument(
        "--iterations", type=int, default=500, help="Iterations (default: 500)"
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup (default: 50)")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to latency_raw.json (default: benchmarks/results/latency_raw.json)",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    if args.run:
        print(
            f"Running latency benchmark: {args.iterations} iterations, "
            f"{args.warmup} warmup"
        )

        # GPU info
        gpu_info = _detect_gpu_backend()
        if gpu_info.get("available"):
            backend = gpu_info["backend"]
            version = gpu_info.get("rocm_version") or gpu_info.get("cuda_version", "")
            devices = gpu_info.get("devices", [])
            print(f"  GPU backend: {backend} {version}")
            for dev in devices:
                print(f"    [{dev['index']}] {dev['name']} ({dev['vram_gb']} GB)")
        else:
            print("  GPU: not available")

        # CPU benchmarks
        print("\n--- CPU Heuristic + Streaming ---")
        data = run_fresh_benchmark(args.iterations, args.warmup)

        # NLI GPU benchmarks
        if args.nli:
            print("\n--- NLI GPU Benchmarks ---")
            nli_data = run_nli_benchmark(
                iterations=min(args.iterations, 100),  # NLI is slower — cap at 100
                warmup=min(args.warmup, 10),
                device=args.device,
            )
            data.update(nli_data)

        # Save raw times
        raw_path = out_dir / "latency_raw.json"
        save_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "python": sys.version.split()[0],
            "gpu_info": gpu_info if gpu_info.get("available") else None,
            "raw_times_ms": data,
        }
        raw_path.write_text(
            json.dumps(save_payload, indent=2),
            encoding="utf-8",
        )
        print(f"\nRaw times saved to {raw_path}")
    else:
        input_path = Path(args.input) if args.input else out_dir / "latency_raw.json"
        if not input_path.exists():
            # Fallback to summary-only JSON
            input_path = out_dir / "latency.json"
        if not input_path.exists():
            print("No results file found. Run with --run or provide --input.")
            sys.exit(1)
        results = json.loads(input_path.read_text(encoding="utf-8"))
        if "raw_times_ms" in results:
            data = results["raw_times_ms"]
        else:
            print(
                "Warning: only summary data available (no raw times). "
                "Run with --run for full distribution."
            )
            data = {
                r["name"]: [r["mean_ms"]] * r["n"] for r in results.get("results", [])
            }

    print_percentile_table(data)

    try:
        import matplotlib  # noqa: F401

        plot_distribution(data, out_dir / "latency_distribution.png")
    except ImportError:
        print("matplotlib not available — skipping histogram plot.")


if __name__ == "__main__":
    main()
