# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Regression tests for retrieval benchmark quality reporting."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import retrieval_bench


def test_scoring_probe_summary_reports_downstream_quality():
    results = [
        retrieval_bench.ScoringProbeResult(
            query="q1",
            response_key="fact_a",
            label_supported=True,
            factual_divergence=0.1,
            accepted=True,
            latency_ms=1.0,
        ),
        retrieval_bench.ScoringProbeResult(
            query="q1",
            response_key="distractor_a",
            label_supported=False,
            factual_divergence=0.8,
            accepted=False,
            latency_ms=2.0,
        ),
        retrieval_bench.ScoringProbeResult(
            query="q2",
            response_key="fact_b",
            label_supported=True,
            factual_divergence=0.6,
            accepted=False,
            latency_ms=3.0,
        ),
    ]

    summary = retrieval_bench.summarize_scoring_probe(results)

    assert summary["total_cases"] == 3
    assert summary["scoring_accuracy"] == pytest.approx(2 / 3)
    assert summary["supported_accept_rate"] == pytest.approx(0.5)
    assert summary["unsupported_reject_rate"] == pytest.approx(1.0)
    assert summary["avg_supported_divergence"] == pytest.approx(0.35)
    assert summary["avg_unsupported_divergence"] == pytest.approx(0.8)
    assert summary["latency_ms_avg"] == pytest.approx(2.0)


def test_retrieval_benchmark_output_includes_downstream_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval_bench, "RESULTS_DIR", tmp_path)

    output = retrieval_bench.run_benchmark("inmemory")

    scoring = output["downstream_scoring"]
    assert scoring["total_cases"] == len(retrieval_bench.EVAL_SET) * 2
    assert 0.0 <= scoring["scoring_accuracy"] <= 1.0
    assert 0.0 <= scoring["supported_accept_rate"] <= 1.0
    assert 0.0 <= scoring["unsupported_reject_rate"] <= 1.0
    assert "scoring_threshold" in scoring
    assert (tmp_path / "retrieval_inmemory.json").exists()
