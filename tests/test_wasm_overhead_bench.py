# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native streaming comparison contracts
"""Exercise actual native sessions, comparison summaries and missing-runtime refusal."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.wasm_overhead_bench import (
    CONFIG_JSON,
    ITERATIONS,
    TOKENS,
    bench_native_rust,
    bench_python,
    bench_wasm,
    main,
    parse_wasm_report,
    report,
)


@pytest.mark.parametrize("backend", [bench_native_rust, bench_python])
def test_native_interfaces_complete_the_same_finite_workload(
    backend: Callable[[int, int], list[float]],
) -> None:
    """Both real Python-facing backends return a sample for every complete session."""
    assert TOKENS == 1000
    assert ITERATIONS == 50
    timings = backend(3, 2)
    assert len(timings) == 2
    assert all(sample > 0 for sample in timings)


@pytest.mark.parametrize(
    ("tokens", "iterations"), [(0, 1), (1, 0), (-1, 2), (True, 2), (2, True)]
)
def test_invalid_workloads_are_refused_before_any_runtime(
    tokens: int, iterations: int
) -> None:
    """No backend may silently execute an empty or Boolean workload."""
    for backend in [bench_native_rust, bench_python]:
        with pytest.raises(ValueError, match="positive integers"):
            backend(tokens=tokens, iterations=iterations)
    with pytest.raises(ValueError, match="positive integers"):
        bench_wasm(Path("missing.js"), tokens=tokens, iterations=iterations)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_node_execution_requires_a_positive_finite_deadline(timeout: float) -> None:
    """Invalid child deadlines are rejected before touching the missing module."""
    with pytest.raises(ValueError, match="positive and finite"):
        bench_wasm(Path("missing.js"), timeout=timeout)


def test_missing_wasm_cannot_be_replaced_with_a_two_backend_comparison(
    tmp_path: Path,
) -> None:
    """The actual CLI refuses an absent runtime without creating a result."""
    output = tmp_path / "result.json"
    with pytest.raises(FileNotFoundError):
        main(["--wasm-module", str(tmp_path / "missing.js"), "--output", str(output)])
    assert not output.exists()


def test_p95_uses_nearest_rank_and_retains_small_samples() -> None:
    """An exactly integral percentile selects its rank rather than the next sample."""
    result = report("measured", [float(i) for i in range(1, 21)], tokens=10)
    assert result["p95_ms"] == 19
    assert result["median_ms"] == 10.5
    assert result["us_per_token"] == 1050
    assert report("single", [0.125])["p95_ms"] == 0.125


@pytest.mark.parametrize("samples", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_empty_or_invalid_samples_cannot_be_reported_as_measurements(
    samples: list[float],
) -> None:
    """A missing or nonfinite observation is a refusal, never a zero-cost runtime."""
    with pytest.raises(ValueError):
        report("invalid", samples)


@pytest.mark.parametrize("backend", [bench_native_rust, bench_python])
def test_a_real_halt_cannot_be_reported_as_a_completed_benchmark(
    backend: Callable[..., list[float]],
) -> None:
    """The same full-window gate halts both real native-backed interfaces."""
    config = json.loads(CONFIG_JSON)
    config.update(window_size=3, window_threshold=0.9)
    with pytest.raises(RuntimeError, match="full workload"):
        backend(3, 1, config_json=json.dumps(config))


@pytest.mark.parametrize(
    "payload",
    [
        "invalid JSON",
        "null",
        "[]",
        "{}",
        '{"tokens": true, "iterations": 1, "timings_ms": [1]}',
        '{"tokens": 1, "iterations": true, "timings_ms": [1]}',
        '{"tokens": 2, "iterations": 1, "timings_ms": [1]}',
        '{"tokens": 1, "iterations": 2, "timings_ms": [1]}',
        '{"tokens": 1, "iterations": 1, "timings_ms": null}',
        '{"tokens": 1, "iterations": 1, "timings_ms": []}',
        '{"tokens": 1, "iterations": 1, "timings_ms": [true]}',
        '{"tokens": 1, "iterations": 1, "timings_ms": ["fast"]}',
        '{"tokens": 1, "iterations": 1, "timings_ms": [NaN]}',
        '{"tokens": 1, "iterations": 1, "timings_ms": [Infinity]}',
        '{"tokens": 1, "iterations": 1, "timings_ms": [0]}',
        '{"tokens": 1, "iterations": 1, "timings_ms": [-1]}',
    ],
)
def test_saved_node_reports_refuse_malformed_or_incomplete_measurements(
    payload: str,
) -> None:
    """Rejected format fixtures never become positive native measurement evidence."""
    with pytest.raises(ValueError):
        parse_wasm_report(payload, 1, 1)
