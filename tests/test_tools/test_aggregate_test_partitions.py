# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native test and model evidence refusals
"""Prevent incomplete native evidence from passing the full-suite aggregate."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from tools.aggregate_test_partitions import verify_test_partitions
from tools.ci_workflow_inventory import ROOT, load_ci_workflow_policy
from tools.pytest_partition_timings import COLD_SNAPSHOT, read_timing_snapshot


@pytest.fixture(scope="module")
def actual_partial_worker(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    """Execute real core tests with advisory weights, retaining the honest partial scope."""
    root = tmp_path_factory.mktemp("actual-native-partial-worker")
    snapshot = root / "snapshot.json"
    hints = json.loads(COLD_SNAPSHOT)
    hints["source_revision"] = "b" * 40
    for lane in hints["lanes"]:
        hints["lanes"][lane] = {"tests/test_types.py": 1.0}
    snapshot.write_text(json.dumps(hints))
    report = root / "partition.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_types.py",
            "-q",
            "-p",
            "tools.pytest_partition",
            "--partition-index",
            "0",
            "--partition-report",
            str(report),
            "--partition-revision",
            "a" * 40,
            "--partition-timings",
            str(snapshot),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(report.read_text())["targets"] == ["tests/test_types.py"]
    return report, snapshot


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("snapshot", "complete required run"),
        ("negative", "duration"),
        ("nonfinite", "Nonfinite"),
        ("integer", "duration"),
        ("boolean", "duration"),
        ("overflow", "overflow"),
        ("partial", "repeated native partition"),
    ],
)
def test_forged_partial_worker_and_corrupt_native_measurements_are_refused(
    tmp_path: Path, actual_partial_worker: tuple[Path, Path], mutation: str, reason: str
) -> None:
    """Adversarially alter a real partial report; it must never prove a complete run."""
    original, snapshot = actual_partial_worker
    report = json.loads(original.read_text())
    report["targets"] = ["tests/"]
    if mutation == "snapshot":
        report["timing_snapshot_sha256"] = "0" * 64
    elif mutation == "overflow":
        for phase in report["results"]:
            phase["duration"] = 1e308
    elif mutation != "partial":
        report["results"][0]["duration"] = {
            "negative": -1.0,
            "nonfinite": float("inf"),
            "integer": 1,
            "boolean": True,
        }[mutation]
    path = tmp_path / "forged-worker.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=reason):
        verify_test_partitions(
            [path] * load_ci_workflow_policy().partitions,
            "a" * 40,
            ".".join(platform.python_version().split(".")[:2]),
            timing_snapshot=snapshot,
        )


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
    """Empty success labels do not prove any executed native model review."""
    paths = []
    policy = load_ci_workflow_policy()
    for index in range(policy.partitions):
        path = tmp_path / f"{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "director-test-partition.v2",
                    "index": index,
                    "revision": "1" * 40,
                    "python_version": "3.11.16",
                    "exit_code": 0,
                    "count": policy.partitions,
                    "rows_per_task": policy.rows_per_task,
                    "targets": ["tests/"],
                    "timing_snapshot_sha256": read_timing_snapshot(None, "3.11")[1],
                    "collected": [],
                    "selected": [],
                    "results": [],
                }
            )
        )
        paths.append(path)
    with pytest.raises(ValueError, match="Native collections"):
        verify_test_partitions(paths, "1" * 40, "3.11")
