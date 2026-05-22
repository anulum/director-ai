# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — tests for rust_python_e2e_compare benchmark

from __future__ import annotations

from pathlib import Path

from benchmarks import rust_python_e2e_compare as bench


def test_run_benchmark_returns_dual_mode_payload() -> None:
    payload = bench.run_benchmark(iterations=1, warmup=0)
    assert payload["benchmark"] == "rust_python_e2e_compare"
    assert "rust" in payload["modes"]
    assert "python" in payload["modes"]
    assert payload["scenario_order"]
    first = payload["scenario_order"][0]
    assert first in payload["modes"]["rust"]
    assert first in payload["modes"]["python"]


def test_render_markdown_includes_side_by_side_table(tmp_path: Path) -> None:
    payload = {
        "generated_utc": "2026-05-22T00:00:00+00:00",
        "git_commit": "abc123",
        "python_version": "3.12.3",
        "platform": "linux-test",
        "iterations": 2,
        "warmup": 1,
        "scenario_order": ["scenario_x"],
        "modes": {
            "rust": {
                "scenario_x": {
                    "latency_ms_median": 1.0,
                    "latency_ms_p95": 1.5,
                    "checksum": 10.0,
                }
            },
            "python": {
                "scenario_x": {
                    "latency_ms_median": 2.0,
                    "latency_ms_p95": 2.5,
                    "checksum": 10.0,
                }
            },
        },
    }
    md = bench._render_markdown(
        payload, output_json=tmp_path / "rust_python_e2e_compare.json"
    )
    assert "Rust vs Python E2E Benchmark" in md
    assert "Median speedup (Py/Rust)" in md
    assert "`scenario_x`" in md
    assert "2.000x" in md
    assert "yes" in md
