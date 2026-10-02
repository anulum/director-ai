# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native complete test partition selection
"""Partition the actual pytest case inventory and execute real core tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci_workflow_inventory import ROOT, load_ci_workflow_policy
from tools.pytest_partition import partition_for_nodeid


def test_every_original_halueval_range_has_exactly_one_worker() -> None:
    """Every real model case keeps its contiguous original task range."""
    policy = load_ci_workflow_policy()
    size = policy.rows_per_task // policy.partitions
    for task in policy.dataset_sha256:
        assert [
            partition_for_nodeid(
                f"tests/test_halueval_benchmark.py::test_halueval_full[{task}-{i * size}]",
                policy.partitions,
                policy.rows_per_task,
            )
            for i in range(policy.partitions)
        ] == list(range(policy.partitions))
    for nodeid in (
        "tests/test_halueval_benchmark.py::test_halueval_full",
        "tests/test_halueval_benchmark.py::test_halueval_full[qa-201]",
    ):
        with pytest.raises(ValueError):
            partition_for_nodeid(nodeid, policy.partitions, policy.rows_per_task)


def test_actual_core_tests_emit_native_collection_and_phase_results(
    tmp_path: Path,
) -> None:
    """Execute real core type tests through the actual plugin and pytest CLI."""
    policy = load_ci_workflow_policy()
    nodeid = "tests/test_types.py::TestClamp::test_within_range"
    index = partition_for_nodeid(nodeid, policy.partitions, policy.rows_per_task)
    report = tmp_path / "partition.json"
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
            str(index),
            "--partition-report",
            str(report),
            "--partition-revision",
            "1" * 40,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(report.read_text())
    assert nodeid in evidence["collected"]
    assert evidence["selected"] == evidence["collected"]
    assert evidence["exit_code"] == 0
    assert {row["phase"] for row in evidence["results"]} == {
        "setup",
        "call",
        "teardown",
    }


@pytest.mark.parametrize(
    "selection",
    [
        ["-k", "within"],
        ["-m", "not slow"],
        ["--ignore", "tests/test_types.py"],
        ["--ignore-glob", "*types*"],
        ["--deselect", "tests/test_types.py::TestClamp::test_within_range"],
        ["--lf"],
        ["--stepwise"],
        ["--collect-only"],
    ],
)
def test_partition_cli_refuses_filtered_collection(
    tmp_path: Path, selection: list[str]
) -> None:
    """Native selection options cannot masquerade as a complete test worker."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_types.py",
            "-q",
            *selection,
            "-p",
            "tools.pytest_partition",
            "--partition-index",
            "0",
            "--partition-report",
            str(tmp_path / "partition.json"),
            "--partition-revision",
            "1" * 40,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "cannot filter" in result.stderr


def test_plugin_preserves_an_ordinary_unpartitioned_native_run(tmp_path: Path) -> None:
    """Loading the optional plugin alone still executes every real core type test."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_types.py",
            "-q",
            "-p",
            "tools.pytest_partition",
            "--basetemp",
            str(tmp_path / "pytest"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "deselected" not in result.stdout


def test_a_terminal_native_report_is_never_overwritten(tmp_path: Path) -> None:
    """A second actual CLI run refuses the original report and preserves its bytes."""
    policy = load_ci_workflow_policy()
    nodeid = "tests/test_types.py::TestClamp::test_within_range"
    report = tmp_path / "partition.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_types.py",
        "-q",
        "-p",
        "tools.pytest_partition",
        "--partition-index",
        str(partition_for_nodeid(nodeid, policy.partitions, policy.rows_per_task)),
        "--partition-report",
        str(report),
        "--partition-revision",
        "1" * 40,
    ]
    first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stdout + first.stderr
    original = report.read_bytes()
    second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "fresh report path" in second.stderr
    assert report.read_bytes() == original


@pytest.mark.parametrize("count,rows", [(0, 200), (8, 0), (3, 200)])
def test_public_partition_refuses_incomplete_range_contracts(
    count: int, rows: int
) -> None:
    """An invalid range contract cannot silently assign any real test case."""
    with pytest.raises(ValueError, match="Invalid complete test partition"):
        partition_for_nodeid(
            "tests/test_types.py::TestClamp::test_within_range", count, rows
        )


@pytest.mark.parametrize("mutation", ["duplicate", "empty"])
def test_native_collection_refuses_duplicate_ids_and_empty_workers(
    tmp_path: Path, mutation: str
) -> None:
    """Actual collection failures cannot masquerade as successful native workers."""
    policy = load_ci_workflow_policy()
    owner = partition_for_nodeid(
        "tests/test_types.py::TestClamp::test_within_range",
        policy.partitions,
        policy.rows_per_task,
    )
    report = tmp_path / "partition.json"
    targets = ["tests/test_types.py"]
    if mutation == "duplicate":
        targets.append(targets[0])
        targets.append("--keep-duplicates")
    else:
        owner = (owner + 1) % policy.partitions
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *targets,
            "-q",
            "-p",
            "tools.pytest_partition",
            "--partition-index",
            str(owner),
            "--partition-report",
            str(report),
            "--partition-revision",
            "1" * 40,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert ("Duplicate native" if mutation == "duplicate" else "no native cases") in (
        result.stdout + result.stderr
    )
    evidence = json.loads(report.read_text())
    assert evidence["exit_code"] != 0
    assert evidence["results"] == []
