# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete model-backed HaluEval test ranges
"""Exercise every original HaluEval row with the real configured NLI model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.halueval_eval import _CACHE_DIR, run_halueval_benchmark
from tools.ci_workflow_inventory import load_ci_workflow_policy

POLICY = load_ci_workflow_policy(Path(__file__).resolve().parents[1])
RANGE_SIZE = POLICY.rows_per_task // POLICY.partitions


@pytest.mark.slow
def test_halueval_qa_sample() -> None:
    """Review both original responses for the existing 25-row QA smoke test."""
    result = run_halueval_benchmark(
        tasks=["qa"],
        use_nli=True,
        max_samples_per_task=25,
        require_complete=True,
    )
    assert result.overall.total == 50


@pytest.mark.slow
@pytest.mark.parametrize(
    "task,sample_offset",
    [
        (task, offset)
        for task in POLICY.dataset_sha256
        for offset in range(0, POLICY.rows_per_task, RANGE_SIZE)
    ],
)
def test_halueval_full(
    task: str,
    sample_offset: int,
    record_property: Callable[[str, object], None],
) -> None:
    """Review one complete original range and retain every real review identity.

    Parameters
    ----------
    task : str
        Original task selected by the complete three-task contract.
    sample_offset : int
        Contiguous range start within the original first 200 dataset rows.
    record_property : Callable
        Native pytest evidence recorder, retained by the partition plugin.
    """
    result = run_halueval_benchmark(
        tasks=[task],
        use_nli=True,
        max_samples_per_task=RANGE_SIZE,
        sample_offset=sample_offset,
        require_complete=True,
    )
    digest = hashlib.sha256(
        (_CACHE_DIR / f"halueval_{task}.parquet").read_bytes()
    ).hexdigest()
    assert digest == POLICY.dataset_sha256[task]
    assert result.overall.total == 2 * RANGE_SIZE
    assert len(result.evaluations) == 2 * RANGE_SIZE
    record_property(
        "halueval",
        json.dumps(
            {
                "task": task,
                "sample_offset": sample_offset,
                "dataset_sha256": digest,
                "model_required": True,
                "evaluations": [asdict(row) for row in result.evaluations],
            },
            allow_nan=False,
        ),
    )
