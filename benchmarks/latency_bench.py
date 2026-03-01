# ---------------------------------------------------------------------
# Director-Class AI — End-to-End Latency Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ---------------------------------------------------------------------
"""
Measure wall-clock latency: sequential vs batched, PyTorch vs ONNX.

Usage:
    python -m benchmarks.latency_bench                # lightweight only
    python -m benchmarks.latency_bench --nli          # + PyTorch seq/batch
    python -m benchmarks.latency_bench --nli --onnx   # + ONNX export + seq/batch
    python -m benchmarks.latency_bench --nli --onnx --iterations 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LatencyResult:
    name: str
    times_ms: list[float] = field(default_factory=list, repr=False)

    @property
    def mean(self) -> float:
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.times_ms)) if self.times_ms else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.times_ms, 95)) if self.times_ms else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.times_ms, 99)) if self.times_ms else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.times_ms)) if self.times_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n": len(self.times_ms),
            "mean_ms": round(self.mean, 2),
            "median_ms": round(self.median, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "std_ms": round(self.std, 2),
        }


# ── Test data ──────────────────────────────────────────────────

FACTS = {
    "capital_france": "Paris is the capital of France.",
    "capital_germany": "Berlin is the capital of Germany.",
    "sky_color": "The sky appears blue due to Rayleigh scattering.",
    "water_formula": "Water has the chemical formula H2O.",
    "earth_sun": "The Earth orbits the Sun at 149.6 million km.",
}

# 16 pairs for batch benchmarking (mix of entailing, contradicting, neutral)
BENCH_PAIRS = [
    ("Paris is the capital of France.", "France's capital city is Paris."),
    ("Paris is the capital of France.", "London is the capital of France."),
    ("The sky is blue due to Rayleigh scattering.", "The sky appears blue."),
    ("Water has the formula H2O.", "Water is composed of hydrogen and oxygen."),
    ("The Earth orbits the Sun.", "The Sun orbits the Earth."),
    ("Dogs are mammals.", "Dogs are reptiles."),
    ("Python is a programming language.", "Python is interpreted."),
    ("The speed of light is 3e8 m/s.", "Light travels at 300,000 km/s."),
    ("Einstein developed relativity.", "Newton developed relativity."),
    ("DNA carries genetic information.", "RNA carries genetic information."),
    ("Photosynthesis uses sunlight.", "Plants produce oxygen via photosynthesis."),
    ("The Pacific is the largest ocean.", "The Atlantic is the largest ocean."),
    ("Gravity attracts mass.", "Gravity is a fundamental force."),
    ("Iron has atomic number 26.", "Iron is a transition metal."),
    ("Shakespeare wrote Hamlet.", "Marlowe wrote Hamlet."),
    ("The Nile is the longest river.", "The Nile flows through Africa."),
]

# Long document for chunked scoring test
LONG_SOURCE = ". ".join([
    "The global mean surface temperature has increased by approximately 1.1 degrees "
    "Celsius since the pre-industrial era",
    "Carbon dioxide concentrations have risen from 280 ppm to over 420 ppm",
    "Arctic sea ice extent has declined by roughly 13 percent per decade",
    "Sea levels have risen about 20 centimeters since 1900",
    "Ocean acidification has increased by 30 percent due to CO2 absorption",
    "Extreme weather events have become more frequent and intense",
    "The Paris Agreement aims to limit warming to 1.5 degrees Celsius",
    "Renewable energy capacity has doubled in the past decade",
    "Methane emissions from agriculture contribute significantly to warming",
    "Permafrost thaw releases stored carbon into the atmosphere",
    "Coral reef bleaching events have tripled in frequency since 1980",
    "Global ice sheet mass loss has accelerated to 150 billion tonnes per year",
]) + "."

LONG_CLAIM = "Climate change has caused significant environmental impacts."


# ── Lightweight / streaming ────────────────────────────────────

def bench_lightweight(iterations: int, warmup: int) -> LatencyResult:
    from director_ai.core import CoherenceScorer, GroundTruthStore

    store = GroundTruthStore()
    for k, v in FACTS.items():
        store.add(k, v)
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store, use_nli=False)

    for _ in range(warmup):
        scorer.review(BENCH_PAIRS[0][0], BENCH_PAIRS[0][1])

    result = LatencyResult("review (no NLI)")
    for i in range(iterations):
        q, r = BENCH_PAIRS[i % len(BENCH_PAIRS)]
        t0 = time.perf_counter()
        scorer.review(q, r)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def bench_streaming(iterations: int, warmup: int) -> LatencyResult:
    from director_ai.core import StreamingKernel

    scores = [0.9, 0.85, 0.88, 0.92, 0.87, 0.90, 0.86, 0.91, 0.89]
    tokens = ["The", " sky", " is", " blue", " due",
              " to", " Rayleigh", " scattering", "."]

    def make_callback():
        idx = [0]
        def cb(_token):
            s = scores[idx[0] % len(scores)]
            idx[0] += 1
            return s
        return cb

    sk = StreamingKernel()
    for _ in range(warmup):
        sk.stream_tokens(iter(tokens), make_callback())

    result = LatencyResult("streaming session")
    for _ in range(iterations):
        sk = StreamingKernel()
        t0 = time.perf_counter()
        sk.stream_tokens(iter(tokens), make_callback())
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


# ── NLI: sequential vs batched ─────────────────────────────────

def bench_nli_sequential(nli, iterations: int, warmup: int) -> LatencyResult:
    """Time N sequential score() calls on all 16 pairs."""
    for _ in range(warmup):
        for p, h in BENCH_PAIRS:
            nli.score(p, h)

    result = LatencyResult(f"PyTorch seq (16 pairs)")
    for _ in range(iterations):
        t0 = time.perf_counter()
        for p, h in BENCH_PAIRS:
            nli.score(p, h)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def bench_nli_batched(nli, iterations: int, warmup: int) -> LatencyResult:
    """Time 1 score_batch() call with all 16 pairs."""
    for _ in range(warmup):
        nli.score_batch(BENCH_PAIRS)

    result = LatencyResult(f"PyTorch batch (16 pairs)")
    for _ in range(iterations):
        t0 = time.perf_counter()
        nli.score_batch(BENCH_PAIRS)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def bench_nli_chunked_sequential(nli, iterations: int, warmup: int) -> LatencyResult:
    """Chunked scoring: manually score each chunk sequentially."""
    for _ in range(warmup):
        chunks = nli._build_chunks(
            nli._split_sentences(LONG_SOURCE),
            int(nli.max_length * 0.6),
        )
        for chunk in chunks:
            nli.score(LONG_SOURCE[:200], chunk)

    result = LatencyResult("PyTorch chunked-seq")
    for _ in range(iterations):
        chunks = nli._build_chunks(
            nli._split_sentences(LONG_SOURCE),
            int(nli.max_length * 0.6),
        )
        t0 = time.perf_counter()
        for chunk in chunks:
            nli.score(LONG_SOURCE[:200], chunk)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def bench_nli_chunked_batched(nli, iterations: int, warmup: int) -> LatencyResult:
    """Chunked scoring via score_chunked() (uses score_batch internally)."""
    for _ in range(warmup):
        nli.score_chunked(LONG_SOURCE[:200], LONG_SOURCE)

    result = LatencyResult("PyTorch chunked-batch")
    for _ in range(iterations):
        t0 = time.perf_counter()
        nli.score_chunked(LONG_SOURCE[:200], LONG_SOURCE)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def bench_onnx_sequential(nli, iterations: int, warmup: int) -> LatencyResult:
    """Time N sequential score() calls on all 16 pairs (ONNX)."""
    for _ in range(warmup):
        for p, h in BENCH_PAIRS:
            nli.score(p, h)

    result = LatencyResult(f"ONNX seq (16 pairs)")
    for _ in range(iterations):
        t0 = time.perf_counter()
        for p, h in BENCH_PAIRS:
            nli.score(p, h)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def bench_onnx_batched(nli, iterations: int, warmup: int) -> LatencyResult:
    """Time 1 score_batch() call with all 16 pairs (ONNX)."""
    for _ in range(warmup):
        nli.score_batch(BENCH_PAIRS)

    result = LatencyResult(f"ONNX batch (16 pairs)")
    for _ in range(iterations):
        t0 = time.perf_counter()
        nli.score_batch(BENCH_PAIRS)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


# ── Output ─────────────────────────────────────────────────────

def print_result(r: LatencyResult) -> None:
    print(f"  {r.name:28s}  mean={r.mean:8.2f} ms  "
          f"median={r.median:8.2f} ms  p95={r.p95:8.2f} ms  "
          f"(n={len(r.times_ms)})")


def print_comparison(label: str, a: LatencyResult, b: LatencyResult) -> None:
    if a.median > 0 and b.median > 0:
        if a.median >= b.median:
            ratio = a.median / b.median
            print(f"  >> {label}: {b.name} is {ratio:.1f}x faster "
                  f"({a.median:.1f} ms vs {b.median:.1f} ms)")
        else:
            ratio = b.median / a.median
            print(f"  >> {label}: {a.name} is {ratio:.1f}x faster "
                  f"({a.median:.1f} ms vs {b.median:.1f} ms)")


def main():
    parser = argparse.ArgumentParser(description="Director-AI latency benchmark")
    parser.add_argument(
        "--nli", action="store_true",
        help="PyTorch NLI benchmarks (requires torch + transformers)",
    )
    parser.add_argument(
        "--onnx", action="store_true",
        help="ONNX Runtime benchmarks (exports model, requires optimum)",
    )
    parser.add_argument(
        "--onnx-path", type=str, default=None,
        help="Path to pre-exported ONNX model (skip export step)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device for PyTorch backend (cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--iterations", type=int, default=30,
        help="Iterations per benchmark (default: 30)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup iterations (default: 5)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("Director-AI Latency Benchmark (v1.4.0)")
    print(f"  iterations={args.iterations}  warmup={args.warmup}")
    print(f"  python={sys.version.split()[0]}")

    try:
        import torch
        cuda_info = (f"cuda={torch.cuda.is_available()}"
                     + (f" ({torch.cuda.get_device_name(0)})"
                        if torch.cuda.is_available() else ""))
        print(f"  torch={torch.__version__}  {cuda_info}")
    except ImportError:
        print("  torch=N/A")

    try:
        import onnxruntime as ort
        print(f"  onnxruntime={ort.__version__}  "
              f"providers={ort.get_available_providers()}")
    except ImportError:
        print("  onnxruntime=N/A")

    print("=" * 72)

    results = []

    # ── Lightweight baselines ──────────────────────────────────
    print("\n--- Lightweight / Streaming ---")
    r = bench_lightweight(args.iterations, args.warmup)
    print_result(r)
    results.append(r)

    r = bench_streaming(args.iterations, args.warmup)
    print_result(r)
    results.append(r)

    # ── PyTorch NLI ────────────────────────────────────────────
    pt_seq = pt_batch = pt_chunk_seq = pt_chunk_batch = None

    if args.nli:
        from director_ai.core.nli import NLIScorer

        print("\n--- PyTorch NLI (FactCG-DeBERTa-v3-Large) ---")
        print("  Loading model...", end=" ", flush=True)
        t0 = time.perf_counter()
        nli = NLIScorer(use_model=True, device=args.device)
        if not nli.model_available:
            print("FAILED (model unavailable)")
            sys.exit(1)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"done ({load_ms:.0f} ms)")

        pt_seq = bench_nli_sequential(nli, args.iterations, args.warmup)
        print_result(pt_seq)
        results.append(pt_seq)

        pt_batch = bench_nli_batched(nli, args.iterations, args.warmup)
        print_result(pt_batch)
        results.append(pt_batch)

        print_comparison("batch vs seq (16 pairs)", pt_seq, pt_batch)

        print()
        pt_chunk_seq = bench_nli_chunked_sequential(nli, args.iterations, args.warmup)
        print_result(pt_chunk_seq)
        results.append(pt_chunk_seq)

        pt_chunk_batch = bench_nli_chunked_batched(nli, args.iterations, args.warmup)
        print_result(pt_chunk_batch)
        results.append(pt_chunk_batch)

        print_comparison("chunked batch vs seq", pt_chunk_seq, pt_chunk_batch)

    # ── ONNX ──────────────────────────────────────────────────
    onnx_seq = onnx_batch = None

    if args.onnx:
        from director_ai.core.nli import NLIScorer, export_onnx

        onnx_path = args.onnx_path
        if not onnx_path:
            print("\n--- ONNX Export ---")
            onnx_dir = Path(__file__).parent / "results" / "factcg_onnx"
            if (onnx_dir / "model.onnx").exists() or any(
                f.endswith(".onnx") for f in (onnx_dir.iterdir() if onnx_dir.exists() else [])
            ):
                print(f"  Using cached export: {onnx_dir}")
                onnx_path = str(onnx_dir)
            else:
                print("  Exporting FactCG to ONNX...", end=" ", flush=True)
                t0 = time.perf_counter()
                onnx_path = export_onnx(output_dir=str(onnx_dir))
                export_ms = (time.perf_counter() - t0) * 1000
                print(f"done ({export_ms:.0f} ms)")

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            has_cuda = "CUDAExecutionProvider" in providers
            label = "ONNX GPU" if has_cuda else "ONNX CPU"
        except ImportError:
            label = "ONNX"
        print(f"\n--- {label} ---")
        print("  Loading session...", end=" ", flush=True)
        t0 = time.perf_counter()
        nli_onnx = NLIScorer(
            use_model=True, backend="onnx", onnx_path=onnx_path,
        )
        if not nli_onnx.model_available:
            print("FAILED (ONNX session unavailable)")
        else:
            load_ms = (time.perf_counter() - t0) * 1000
            print(f"done ({load_ms:.0f} ms)")

            onnx_seq = bench_onnx_sequential(nli_onnx, args.iterations, args.warmup)
            print_result(onnx_seq)
            results.append(onnx_seq)

            onnx_batch = bench_onnx_batched(nli_onnx, args.iterations, args.warmup)
            print_result(onnx_batch)
            results.append(onnx_batch)

            print_comparison("ONNX batch vs seq", onnx_seq, onnx_batch)

    # ── Cross-backend comparison ───────────────────────────────
    if pt_batch and onnx_batch:
        print(f"\n--- Cross-Backend (16-pair batch) ---")
        print_comparison("ONNX vs PyTorch (batch)", pt_batch, onnx_batch)
    if pt_seq and onnx_seq:
        print_comparison("ONNX vs PyTorch (seq)", pt_seq, onnx_seq)

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  {'Benchmark':28s}  {'mean':>8s}  {'median':>8s}  {'p95':>8s}  {'per-pair':>8s}")
    print(f"  {'-' * 64}")
    for r in results:
        n_pairs = 16 if "16 pairs" in r.name else 1
        per_pair = r.median / n_pairs
        print(f"  {r.name:28s}  {r.mean:7.1f}ms  {r.median:7.1f}ms  "
              f"{r.p95:7.1f}ms  {per_pair:7.1f}ms")
    print(f"{'=' * 72}\n")

    # ── Save results ──────────────────────────────────────────
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "python": sys.version.split()[0],
        "results": [r.to_dict() for r in results],
    }

    try:
        import torch
        out["torch_version"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        import onnxruntime as ort
        out["onnxruntime_version"] = ort.__version__
        out["onnx_providers"] = ort.get_available_providers()
    except ImportError:
        pass

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "latency.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
