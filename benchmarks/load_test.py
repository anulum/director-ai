# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Load Testing Benchmark

"""Measure requests-per-second and latency percentiles under concurrent load.

Tests both the Python scorer directly and the FastAPI server (if running).

Usage::

    python -m benchmarks.load_test
    python -m benchmarks.load_test --rps-target 100 --duration 30
    python -m benchmarks.load_test --server http://localhost:8080 --concurrency 16
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from benchmarks._common import save_results

logger = logging.getLogger("DirectorAI.Benchmark.LoadTest")

_SAMPLE_PAIRS = [
    ("The capital of France is Paris.", "Paris is the capital of France."),
    ("Water boils at 100 degrees Celsius.", "Water boils at 50 degrees."),
    ("The Earth orbits the Sun.", "The Sun orbits the Earth."),
    ("Python is a programming language.", "Python is a snake species."),
    ("Gravity pulls objects toward Earth.", "Objects float in normal gravity."),
    ("DNA carries genetic information.", "DNA stores genetic data."),
    ("The speed of light is constant.", "Light speed varies with temperature."),
    ("Photosynthesis converts CO2 to oxygen.", "Plants absorb oxygen from air."),
]


@dataclass
class LoadTestResult:
    """Load test metrics."""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    duration_s: float = 0.0
    concurrency: int = 1
    latencies_ms: list[float] = field(default_factory=list, repr=False)

    @property
    def rps(self) -> float:
        return self.successful / self.duration_s if self.duration_s else 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def avg_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "duration_s": round(self.duration_s, 2),
            "concurrency": self.concurrency,
            "rps": round(self.rps, 1),
            "latency_p50_ms": round(self.p50_ms, 2),
            "latency_p95_ms": round(self.p95_ms, 2),
            "latency_p99_ms": round(self.p99_ms, 2),
            "latency_avg_ms": round(self.avg_ms, 2),
        }


def run_scorer_load_test(
    concurrency: int = 4,
    duration_s: float = 10.0,
    use_nli: bool = False,
    threshold: float = 0.5,
) -> LoadTestResult:
    """Load test CoherenceScorer.review() with concurrent threads."""
    from director_ai.core.scorer import CoherenceScorer

    scorer = CoherenceScorer(threshold=threshold, use_nli=use_nli)
    result = LoadTestResult(concurrency=concurrency)

    stop_time = time.monotonic() + duration_s
    pair_idx = 0

    def _score_one() -> float:
        nonlocal pair_idx
        idx = pair_idx % len(_SAMPLE_PAIRS)
        pair_idx += 1
        prompt, response = _SAMPLE_PAIRS[idx]
        t0 = time.perf_counter()
        scorer.review(prompt, response)
        return (time.perf_counter() - t0) * 1000

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        while time.monotonic() < stop_time:
            futures.append(pool.submit(_score_one))
            # Limit outstanding futures to prevent memory blow-up
            if len(futures) > concurrency * 100:
                for f in as_completed(futures[:concurrency]):
                    try:
                        lat = f.result(timeout=5)
                        result.latencies_ms.append(lat)
                        result.successful += 1
                    except Exception:
                        result.failed += 1
                    result.total_requests += 1
                futures = futures[concurrency:]

        for f in as_completed(futures):
            try:
                lat = f.result(timeout=10)
                result.latencies_ms.append(lat)
                result.successful += 1
            except Exception:
                result.failed += 1
            result.total_requests += 1

    result.duration_s = time.monotonic() - t_start
    return result


def run_server_load_test(
    server_url: str = "http://localhost:8080",
    concurrency: int = 8,
    duration_s: float = 10.0,
) -> LoadTestResult:
    """Load test the FastAPI /v1/review endpoint."""
    import requests

    result = LoadTestResult(concurrency=concurrency)
    stop_time = time.monotonic() + duration_s
    pair_idx = 0

    def _request_one() -> float:
        nonlocal pair_idx
        idx = pair_idx % len(_SAMPLE_PAIRS)
        pair_idx += 1
        prompt, response = _SAMPLE_PAIRS[idx]
        t0 = time.perf_counter()
        resp = requests.post(
            f"{server_url}/v1/review",
            json={"prompt": prompt, "response": response},
            timeout=10,
        )
        lat = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        return lat

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        while time.monotonic() < stop_time:
            futures.append(pool.submit(_request_one))
            if len(futures) > concurrency * 50:
                for f in as_completed(futures[:concurrency]):
                    try:
                        lat = f.result(timeout=10)
                        result.latencies_ms.append(lat)
                        result.successful += 1
                    except Exception:
                        result.failed += 1
                    result.total_requests += 1
                futures = futures[concurrency:]

        for f in as_completed(futures):
            try:
                lat = f.result(timeout=15)
                result.latencies_ms.append(lat)
                result.successful += 1
            except Exception:
                result.failed += 1
            result.total_requests += 1

    result.duration_s = time.monotonic() - t_start
    return result


def _print_load_results(r: LoadTestResult, label: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Concurrency:  {r.concurrency}")
    print(f"  Duration:     {r.duration_s:.1f}s")
    print(f"  Requests:     {r.total_requests} ({r.successful} ok, {r.failed} failed)")
    print(f"  RPS:          {r.rps:.1f}")
    print(f"  Latency P50:  {r.p50_ms:.2f} ms")
    print(f"  Latency P95:  {r.p95_ms:.2f} ms")
    print(f"  Latency P99:  {r.p99_ms:.2f} ms")
    print(f"  Latency Avg:  {r.avg_ms:.2f} ms")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Director-AI load test")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--nli", action="store_true")
    parser.add_argument("--server", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    results = {}

    if args.server:
        r = run_server_load_test(
            server_url=args.server,
            concurrency=args.concurrency,
            duration_s=args.duration,
        )
        _print_load_results(r, f"Server Load Test ({args.server})")
        results["server"] = r.to_dict()
    else:
        r = run_scorer_load_test(
            concurrency=args.concurrency,
            duration_s=args.duration,
            use_nli=args.nli,
            threshold=args.threshold,
        )
        _print_load_results(r, "Scorer Load Test (direct)")
        results["scorer"] = r.to_dict()

    save_results(
        {"benchmark": "load_test", **results},
        "load_test.json",
    )
