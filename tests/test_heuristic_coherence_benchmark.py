# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Heuristic coherence benchmark tests
"""Contract tests for the heuristic coherence local-regression benchmark."""

from __future__ import annotations

from tools.ci_workflow_inventory import object_mapping


def test_heuristic_coherence_benchmark_schema() -> None:
    """Pin the emitted no-model route labels and throughput counters."""
    from benchmarks.heuristic_coherence_pipeline import run

    result = run(repeats=3)

    assert result["benchmark"] == "heuristic_coherence_pipeline"
    assert result["evidence_grade"] == "non_isolated_local_regression"
    routes = object_mapping(result["routes"])
    throughput = object_mapping(result["throughput"])
    outputs = object_mapping(result["route_outputs"])
    assert routes["cases"] == 4
    assert throughput["reviews"] == 12
    rate = throughput["reviews_per_sec"]
    assert isinstance(rate, (int, float))
    assert rate > 0
    assert object_mapping(outputs["dialogue"])["route"] == "parallel_components"
    assert object_mapping(outputs["prompt_premise"])["route"] == "factual_only"
    assert object_mapping(outputs["factual_only"])["h_logical"] == 0.0
