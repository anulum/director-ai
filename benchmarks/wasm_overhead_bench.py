# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native, Python and Node/WASM streaming comparison
"""Compare real streaming interfaces without treating host costs as pure FFI overhead."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

TOKENS = 1000
ITERATIONS = 50
SCORE = 0.8
ROOT = Path(__file__).resolve().parents[1]
CONFIG_JSON = json.dumps(
    {
        "coherence_threshold": 0.6,
        "hard_limit": 0.5,
        "soft_limit": 0.7,
        "w_logic": 0.6,
        "w_fact": 0.4,
        "window_size": 10,
        "window_threshold": 0.55,
        "trend_window": 5,
        "trend_threshold": 0.15,
        "history_window": 5,
        "deadline_ms": 50,
        "logit_entropy_limit": 1.2,
    }
)


def _counts(tokens: int, iterations: int) -> None:
    """Refuse empty, negative or Boolean workloads before invoking a runtime."""
    if (
        isinstance(tokens, bool)
        or isinstance(iterations, bool)
        or tokens < 1
        or iterations < 1
    ):
        raise ValueError("tokens and iterations must be positive integers")


def bench_native_rust(
    tokens: int = TOKENS,
    iterations: int = ITERATIONS,
    *,
    config_json: str = CONFIG_JSON,
) -> list[float]:
    """Time the actual Rust batch interface with a Python score callback.

    Parameters
    ----------
    tokens, iterations : int
        Positive stream size and independent session count.
    config_json : str
        Complete shared native configuration; applied before each timed session.

    Returns
    -------
    list of float
        Milliseconds per completed session, excluding kernel construction.

    Raises
    ------
    ImportError, RuntimeError, ValueError
        If the native backend is missing, the stream is incomplete, or counts are invalid.
    """
    _counts(tokens, iterations)
    from backfire_kernel import BackfireConfig, RustStreamingKernel

    inputs = [f"token_{i}" for i in range(tokens)]
    timings: list[float] = []
    for _ in range(iterations):
        kernel = RustStreamingKernel(config=BackfireConfig.from_json(config_json))
        started = time.perf_counter()
        session = kernel.stream_tokens(inputs, lambda _token: SCORE)
        timings.append((time.perf_counter() - started) * 1000)
        if session.halted or session.token_count() != tokens:
            raise RuntimeError(
                "Native streaming benchmark did not consume its full workload"
            )
    return timings


def bench_python(
    tokens: int = TOKENS,
    iterations: int = ITERATIONS,
    *,
    config_json: str = CONFIG_JSON,
) -> list[float]:
    """Time Python orchestration, including its mandatory native statistics.

    Parameters
    ----------
    tokens, iterations : int
        Positive stream size and independent session count.
    config_json : str
        The same validated native configuration used by the other backends.

    Returns
    -------
    list of float
        Milliseconds per completed session, excluding kernel construction.

    Raises
    ------
    ImportError, RuntimeError, ValueError
        If mandatory statistics are missing, a stream is incomplete, or counts are invalid.
    """
    _counts(tokens, iterations)
    from backfire_kernel import BackfireConfig

    from director_ai.core.runtime.streaming import StreamingKernel

    BackfireConfig.from_json(config_json)
    config = json.loads(config_json)

    inputs = [f"token_{i}" for i in range(tokens)]
    timings: list[float] = []
    for _ in range(iterations):
        kernel = StreamingKernel(
            hard_limit=float(config["hard_limit"]),
            soft_limit=float(config["soft_limit"]),
            window_size=int(config["window_size"]),
            window_threshold=float(config["window_threshold"]),
            trend_window=int(config["trend_window"]),
            trend_threshold=float(config["trend_threshold"]),
        )
        started = time.perf_counter()
        session = kernel.stream_tokens(inputs, lambda _text: SCORE)
        timings.append((time.perf_counter() - started) * 1000)
        if session.halted or session.token_count != tokens:
            raise RuntimeError(
                "Python streaming benchmark did not consume its full workload"
            )
    return timings


def bench_wasm(
    module: Path,
    tokens: int = TOKENS,
    iterations: int = ITERATIONS,
    *,
    node: str = "node",
    timeout: float = 120.0,
) -> tuple[list[float], dict[str, object]]:
    """Run the actual Node/WASM helper and retain its complete native samples.

    Parameters
    ----------
    module : Path
        Actual nodejs-target wasm-bindgen JavaScript entry point.
    tokens, iterations : int
        Positive stream size and independent session count.
    node : str
        Node executable, inherited by the child without a shell.
    timeout : float
        Positive finite deadline in seconds; an expired child is killed and reaped.

    Returns
    -------
    tuple of list of float and dict of str to object
        Validated samples and the helper's runtime and artefact identities.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If inputs, real module, native child or its complete report cannot qualify.
    """
    _counts(tokens, iterations)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    module = module.resolve(strict=True)
    completed = subprocess.run(
        [
            node,
            str(ROOT / "benchmarks/wasm_overhead_node.mjs"),
            str(module),
            str(tokens),
            str(iterations),
            CONFIG_JSON,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    return parse_wasm_report(completed.stdout, tokens, iterations)


def parse_wasm_report(
    payload: str, tokens: int, iterations: int
) -> tuple[list[float], dict[str, object]]:
    """Decode the Node report without admitting incomplete or invalid measurements.

    Parameters
    ----------
    payload : str
        JSON emitted by the actual Node helper or read from its saved output.
    tokens, iterations : int
        Expected positive workload sizes.

    Returns
    -------
    tuple of list of float and dict of str to object
        Validated positive session samples and the complete original measurement packet.

    Raises
    ------
    ValueError
        If JSON, packet shape, counts or samples cannot represent the expected workload.
    """
    _counts(tokens, iterations)
    value: object = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("WASM benchmark report must be an object")
    result: dict[str, object] = value
    samples = result.get("timings_ms")
    if (
        type(result.get("tokens")) is not int
        or type(result.get("iterations")) is not int
        or result["tokens"] != tokens
        or result["iterations"] != iterations
        or not isinstance(samples, list)
        or len(samples) != iterations
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item <= 0
            for item in samples
        )
    ):
        raise ValueError("WASM benchmark report does not match the complete workload")
    return [float(item) for item in samples], result


def report(
    name: str, timings: list[float], tokens: int = TOKENS
) -> dict[str, str | int | float]:
    """Summarise positive raw session samples using nearest-rank p95.

    Parameters
    ----------
    name : str
        Actual implementation label.
    timings : list of float
        Nonempty positive finite millisecond samples, retained separately.
    tokens : int
        Positive token count in every session.

    Returns
    -------
    dict of str to str or int or float
        Sample count, median, nearest-rank p95, extrema and microseconds per token.

    Raises
    ------
    ValueError
        If counts or measured samples cannot produce a meaningful comparison.
    """
    _counts(tokens, len(timings))
    if any(not math.isfinite(sample) or sample <= 0 for sample in timings):
        raise ValueError("timings must be positive and finite")
    median = statistics.median(timings)
    return {
        "name": name,
        "tokens": tokens,
        "iterations": len(timings),
        "median_ms": median,
        "p95_ms": sorted(timings)[math.ceil(len(timings) * 0.95) - 1],
        "min_ms": min(timings),
        "max_ms": max(timings),
        "us_per_token": median * 1000 / tokens,
    }


def run_comparison(
    module: Path,
    tokens: int = TOKENS,
    iterations: int = ITERATIONS,
    *,
    node: str = "node",
    timeout: float = 120.0,
) -> dict[str, object]:
    """Measure all three interfaces and record the non-isolated host context.

    Parameters
    ----------
    module : Path
        Actual generated Node/WASM module; a missing counterpart refuses the comparison.
    tokens, iterations : int
        Positive stream size and independent session count, defaulting to 1000 and 50.
    node : str
        Node executable used by the real helper.
    timeout : float
        Positive finite Node child deadline in seconds.

    Returns
    -------
    dict of str to object
        All raw samples, summaries, boundary descriptions and host context.
    """
    load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
    wasm_times, wasm = bench_wasm(
        module, tokens, iterations, node=node, timeout=timeout
    )
    native_times = bench_native_rust(tokens, iterations)
    python_times = bench_python(tokens, iterations)
    import backfire_kernel

    native_module = Path(
        getattr(backfire_kernel, "backfire_kernel", backfire_kernel).__file__
    ).resolve(strict=True)
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    samples = [
        (
            "native_rust_ffi",
            native_times,
            "One Python-to-Rust batch call; Python callback per token",
        ),
        (
            "python_orchestration",
            python_times,
            "Python events, accumulated text, tracing and native statistics",
        ),
        (
            "node_wasm",
            wasm_times,
            "One JS-to-WASM call and full session snapshot per token",
        ),
    ]
    summaries = [report(name, timings, tokens) for name, timings, _boundary in samples]
    return {
        "schema": "director-ai.streaming-runtime-comparison.v1",
        "measured_at": datetime.now(UTC).isoformat(),
        "tokens": tokens,
        "iterations": iterations,
        "score": SCORE,
        "config": json.loads(CONFIG_JSON),
        "source_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in [
                "benchmarks/wasm_overhead_bench.py",
                "benchmarks/wasm_overhead_node.mjs",
                "src/director_ai/core/runtime/kernel.py",
                "src/director_ai/core/runtime/streaming.py",
                "backfire-kernel/Cargo.lock",
                "backfire-kernel/crates/backfire-core/src/kernel.rs",
                "backfire-kernel/crates/backfire-core/src/signals.rs",
                "backfire-kernel/crates/backfire-wasm/src/lib.rs",
            ]
        },
        "qualification": {
            "isolated": False,
            "purpose": "Shared-host functional/parity measurement; no production latency or pure FFI-overhead claim",
        },
        "implementations": [
            {"summary": summary, "timings_ms": timings, "boundary": boundary}
            for summary, (_name, timings, boundary) in zip(
                summaries, samples, strict=True
            )
        ],
        "python_native_median_ratio": statistics.median(python_times)
        / statistics.median(native_times),
        "wasm": wasm,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "python_executable": sys.executable,
            "cpu_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "load_before": load_before,
            "load_after": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "cpu0_governor": governor.read_text().strip()
            if governor.is_file()
            else None,
            "native_module": str(native_module),
            "native_module_sha256": hashlib.sha256(
                native_module.read_bytes()
            ).hexdigest(),
        },
    }


def main(argv: list[str] | None = None) -> None:
    """Run the comparison CLI and exclusively create its requested JSON result.

    Parameters
    ----------
    argv : list of str or None
        Explicit arguments or the current process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wasm-module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--node", default="node")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run_comparison(
        args.wasm_module,
        args.tokens,
        args.iterations,
        node=args.node,
        timeout=args.timeout,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Complete non-isolated comparison written to {args.output}")


if __name__ == "__main__":
    main()
