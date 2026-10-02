# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native test and model evidence refusals
"""Prevent incomplete native evidence from passing the full-suite aggregate."""

import json
from pathlib import Path

import pytest

from tools.aggregate_test_partitions import verify_test_partitions
from tools.ci_workflow_inventory import load_ci_workflow_policy


@pytest.mark.parametrize("lane", ["3.11", "3.12", "3.13"])
def test_missing_actual_workers_is_refused(lane: str) -> None:
    """No native workers means no complete test or real-model evidence."""
    with pytest.raises(ValueError, match="Every native test partition"):
        verify_test_partitions([], "1" * 40, lane)


@pytest.mark.parametrize(
    "raw",
    [
        '{"index": 0, "index": 0}',
        '{"index": NaN}',
        '{"index": true}',
        '{"index": 99}',
    ],
)
def test_ambiguous_or_invalid_native_evidence_is_refused(
    tmp_path: Path, raw: str
) -> None:
    """Ambiguous JSON and false native worker identities fail closed."""
    path = tmp_path / "partition.json"
    path.write_text(raw)
    count = load_ci_workflow_policy().partitions
    with pytest.raises((ValueError, KeyError)):
        verify_test_partitions([path] * count, "1" * 40, "3.11")


def test_a_small_real_collection_cannot_prove_all_required_tests(
    tmp_path: Path,
) -> None:
    """Eight empty success labels do not prove any executed native model review."""
    paths = []
    policy = load_ci_workflow_policy()
    for index in range(policy.partitions):
        path = tmp_path / f"{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "director-test-partition.v1",
                    "index": index,
                    "revision": "1" * 40,
                    "python_version": "3.11.16",
                    "exit_code": 0,
                    "count": policy.partitions,
                    "rows_per_task": policy.rows_per_task,
                    "targets": ["tests/"],
                    "collected": [],
                    "selected": [],
                    "results": [],
                }
            )
        )
        paths.append(path)
    with pytest.raises(ValueError, match="Native collections"):
        verify_test_partitions(paths, "1" * 40, "3.11")
