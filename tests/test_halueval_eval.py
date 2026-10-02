# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — public benchmark range and evidence contracts
"""Verify public range validation without downloading or substituting a model."""

from typing import TypedDict

import pytest

from benchmarks.halueval_eval import ClassificationMetrics, run_halueval_benchmark


class BenchmarkArguments(TypedDict, total=False):
    """Keep the public benchmark's typed optional caller arguments."""

    tasks: list[str]
    sample_offset: int
    max_samples_per_task: int
    coherence_threshold: float


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tasks": []},
        {"tasks": ["qa", "qa"]},
        {"tasks": ["unknown"]},
        {"sample_offset": -1},
        {"sample_offset": True},
        {"max_samples_per_task": 0},
        {"max_samples_per_task": True},
        {"coherence_threshold": float("nan")},
        {"coherence_threshold": 2.0},
    ],
)
def test_public_benchmark_refuses_invalid_ranges(kwargs: BenchmarkArguments) -> None:
    """Reject invalid contracts before any data or inference side effect."""
    # The runtime boundary deliberately receives malformed decoded caller values.
    with pytest.raises(ValueError):
        run_halueval_benchmark(**kwargs)


def test_original_global_threshold_classification_math() -> None:
    """The additional range/evidence surface preserves all classification counts."""
    metrics = ClassificationMetrics(tp=3, tn=5, fp=1, fn=2)
    assert metrics.total == 11
    assert metrics.precision == 3 / 4
    assert metrics.recall == 3 / 5
    assert metrics.accuracy == 8 / 11
    assert metrics.f1 == pytest.approx(2 * (3 / 4) * (3 / 5) / (3 / 4 + 3 / 5))
