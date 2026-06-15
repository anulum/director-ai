# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — competitor latency benchmark harness tests

"""Offline tests for the competitor guard-latency benchmark.

The statistics, the cross-venv subprocess runner, and the in-process Director-AI
timing are exercised without any competitor library: the subprocess paths are
driven through ``sys.executable`` with self-contained snippets, so absence,
non-zero exit, and unparsable output all resolve to a recorded ``available:
false`` rather than a crash or a fabricated number.
"""

from __future__ import annotations

import sys

from benchmarks.competitor_latency_bench import (
    WORKLOAD,
    _run_competitor,
    run_benchmark,
    summarise,
    time_director_ai,
)

# --------------------------------------------------------------------------- #
# summarise                                                                   #
# --------------------------------------------------------------------------- #


def test_summarise_empty_is_zero_n() -> None:
    assert summarise([]) == {"n": 0}


def test_summarise_single_value() -> None:
    s = summarise([4.0])
    assert s["n"] == 1
    assert s["p50_ms"] == 4.0
    assert s["min_ms"] == s["max_ms"] == 4.0


def test_summarise_orders_and_computes_percentiles() -> None:
    s = summarise([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s["n"] == 5
    assert s["p50_ms"] == 3.0
    assert s["min_ms"] == 1.0
    assert s["max_ms"] == 5.0
    assert s["mean_ms"] == 3.0
    assert s["p95_ms"] == 5.0  # index min(4, int(0.95*5)=4) -> last element


# --------------------------------------------------------------------------- #
# _run_competitor (cross-venv subprocess runner)                              #
# --------------------------------------------------------------------------- #


def test_run_competitor_parses_json_from_snippet() -> None:
    snippet = 'import json; print(json.dumps({"available": True, "n": 7}))'
    out = _run_competitor(sys.executable, snippet)
    assert out == {"available": True, "n": 7}


def test_run_competitor_missing_python_is_unavailable() -> None:
    out = _run_competitor("/no/such/python/binary", "print(1)")
    assert out["available"] is False
    assert "not found" in out["reason"]


def test_run_competitor_nonzero_exit_is_unavailable() -> None:
    out = _run_competitor(sys.executable, "import sys; sys.exit(3)")
    assert out["available"] is False
    assert "exit 3" in out["reason"]


def test_run_competitor_unparsable_output_is_unavailable() -> None:
    out = _run_competitor(sys.executable, 'print("definitely not json")')
    assert out["available"] is False
    assert "unparsable" in out["reason"]


def test_run_competitor_takes_last_stdout_line() -> None:
    snippet = (
        'print("warmup noise"); import json; print(json.dumps({"available": True}))'
    )
    out = _run_competitor(sys.executable, snippet)
    assert out == {"available": True}


# --------------------------------------------------------------------------- #
# time_director_ai (in-process, local, no model)                              #
# --------------------------------------------------------------------------- #


def test_time_director_ai_runs_locally() -> None:
    result = time_director_ai(repeats=2)
    assert result["available"] is True
    assert result["framework"] == "director-ai"
    assert result["makes_local_grounding_decision"] is True
    assert result["n"] == 2 * len(WORKLOAD)
    assert result["p50_ms"] >= 0.0


# --------------------------------------------------------------------------- #
# run_benchmark (Director real; competitors point at a missing venv)          #
# --------------------------------------------------------------------------- #


def test_run_benchmark_degrades_when_competitor_python_absent() -> None:
    result = run_benchmark(repeats=1, competitor_python="/no/such/python")
    by = {f["framework"]: f for f in result["frameworks"]}
    assert by["director-ai"]["available"] is True
    assert by["guardrails-ai"]["available"] is False
    assert by["nemo-guardrails"]["available"] is False
    # The honesty caveat is always recorded.
    assert "not LLM latency" in result["caveat"]
    assert by["nemo-guardrails"]["makes_local_grounding_decision"] is False
