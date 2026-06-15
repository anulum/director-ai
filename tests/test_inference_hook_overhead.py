# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — inference hook overhead benchmark tests

"""Offline smoke + statistics tests for the logits-adapter overhead benchmark."""

from __future__ import annotations

from benchmarks.inference_hook_overhead import _percentiles, run_benchmark


def test_percentiles_basic() -> None:
    s = _percentiles([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s["n"] == 5
    assert s["p50_us"] == 3.0
    assert s["mean_us"] == 3.0
    assert s["p95_us"] == 5.0


def test_run_benchmark_smoke() -> None:
    result = run_benchmark(repeats=3)
    assert result["benchmark"] == "inference_hook_overhead"
    assert result["pass_through_per_token"]["n"] == 3
    assert result["allow_at_boundary"]["n"] == 3
    # EOS mask measured for each configured vocab size.
    assert set(result["eos_mask_by_vocab"]) == {"32000", "128256"}
    # Steady-state pass-through must be cheaper than scoring at a boundary.
    assert (
        result["pass_through_per_token"]["mean_us"]
        <= result["allow_at_boundary"]["mean_us"]
    )
