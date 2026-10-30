# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual indexed native coverage admission
"""Validate genuine worker-attempt artifacts without claiming a complete model run."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci_workflow_inventory import ROOT, load_ci_workflow_policy
from tools.pytest_partition_artifacts import partition_artifact_reports
from tools.pytest_partition_timings import COLD_SNAPSHOT


@pytest.fixture(scope="module")
def native_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    """Generate every declared partial worker and actual incompatible coverage."""
    root = tmp_path_factory.mktemp("native-indexed-coverage")
    bundles = root / "bundles"
    bundles.mkdir()
    lane = ".".join(platform.python_version().split(".")[:2])
    hints = json.loads(COLD_SNAPSHOT)
    hints["source_revision"] = "a" * 40
    for version in hints["lanes"]:
        hints["lanes"][version] = {"tests/test_types.py": 1.0}
    snapshot = root / "snapshot.json"
    snapshot.write_text(json.dumps(hints))
    branch_config = root / "branch.ini"
    line_config = root / "line.ini"
    branch_config.write_text("[run]\nbranch = true\n")
    line_config.write_text("[run]\nbranch = false\n")
    env = dict(os.environ)
    env.pop("COVERAGE_PROCESS_START", None)
    for index in range(load_ci_workflow_policy().partitions):
        bundle = bundles / f"test-partition-{lane}-{index}"
        bundle.mkdir()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(branch_config),
                "--data-file=" + str(bundle / ".coverage.native"),
                "--parallel-mode",
                "--include=" + str(ROOT / "src/director_ai/core/types.py"),
                "-m",
                "pytest",
                "tests/test_types.py",
                "-q",
                "-p",
                "tools.pytest_partition",
                "--partition-index",
                str(index),
                "--partition-report",
                str(bundle / "partition.json"),
                "--partition-revision",
                "b" * 40,
                "--partition-timings",
                str(snapshot),
                "--junitxml",
                str(bundle / "partition.xml"),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert (result.returncode == 0) == (index == 0), result.stdout + result.stderr
    line_data, empty_data = root / ".coverage.lines", root / ".coverage.unmeasured"
    commands = [
        [
            "--rcfile=" + str(line_config),
            "--data-file=" + str(line_data),
            "--include=" + str(ROOT / "src/director_ai/core/types.py"),
            "-m",
            "pytest",
            "tests/test_types.py",
            "-q",
        ],
        [
            "--rcfile=" + str(branch_config),
            "--data-file=" + str(empty_data),
            "--include=" + str(ROOT / "src/director_ai/core/types.py"),
            "-m",
            "json.tool",
            str(snapshot),
        ],
    ]
    for arguments in commands:
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "run", *arguments],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return bundles, line_data, empty_data


def test_actual_layout_is_readable_without_accepting_partial_execution(
    tmp_path: Path, native_artifacts: tuple[Path, Path, Path]
) -> None:
    """Require every real artifact; readable coverage alone cannot qualify partial tests."""
    root, _, _ = native_artifacts
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    lane = ".".join(platform.python_version().split(".")[:2])
    count = load_ci_workflow_policy().partitions
    reports = partition_artifact_reports(root, lane, count)
    assert [json.loads(p.read_text())["index"] for p in reports] == list(range(count))
    assert sum(json.loads(p.read_text())["exit_code"] == 0 for p in reports) == 1
    assert all(
        json.loads(p.read_text())["targets"] == ["tests/test_types.py"] for p in reports
    )
    assert {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    } == before
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.aggregate_test_partitions",
            "--input",
            str(root),
            "--revision",
            "b" * 40,
            "--python-version",
            lane,
            "--output",
            str(tmp_path / "complete.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert not (tmp_path / "complete.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-directory",
        "missing-last-directory",
        "extra-directory",
        "bundle-link",
        "bundle-file",
        "report-missing",
        "report-link",
        "report-empty",
        "junit-missing",
        "report-array",
        "wrong-index",
        "boolean-index",
        "coverage-missing",
        "coverage-link",
        "coverage-directory",
        "coverage-empty",
        "coverage-corrupt",
        "coverage-lines",
        "coverage-unmeasured",
    ],
)
def test_missing_wrong_or_unusable_native_coverage_cannot_be_combined(
    tmp_path: Path, native_artifacts: tuple[Path, Path, Path], mutation: str
) -> None:
    """Mutate copies of actual native outputs; every invalid indexed input must refuse."""
    original, line_data, empty_data = native_artifacts
    root = tmp_path / "bundles"
    shutil.copytree(original, root)
    lane = ".".join(platform.python_version().split(".")[:2])
    index = (
        load_ci_workflow_policy().partitions - 1
        if mutation == "missing-last-directory"
        else 0
    )
    bundle = root / f"test-partition-{lane}-{index}"
    report = bundle / "partition.json"
    coverage_file = next(bundle.glob(".coverage.*"))
    if mutation in {"missing-directory", "missing-last-directory"}:
        bundle.rename(tmp_path / "retained-bundle")
    elif mutation == "extra-directory":
        (root / "unindexed-extra").mkdir()
    elif mutation in {"bundle-link", "bundle-file"}:
        target = tmp_path / "retained-bundle"
        bundle.rename(target)
        if mutation == "bundle-link":
            bundle.symlink_to(target, target_is_directory=True)
        else:
            bundle.write_bytes(b"not a directory")
    elif mutation in {"report-missing", "report-link", "report-empty", "junit-missing"}:
        path = bundle / "partition.xml" if mutation == "junit-missing" else report
        target = tmp_path / "retained-report"
        path.rename(target)
        if mutation == "report-link":
            path.symlink_to(target)
        elif mutation == "report-empty":
            path.write_bytes(b"")
    elif mutation in {"wrong-index", "boolean-index", "report-array"}:
        value = json.loads(report.read_text())
        value["index"] = True if mutation == "boolean-index" else 1
        report.write_text(json.dumps([] if mutation == "report-array" else value))
    else:
        retained = tmp_path / "retained-coverage"
        coverage_file.rename(retained)
        if mutation == "coverage-link":
            coverage_file.symlink_to(retained)
        elif mutation == "coverage-directory":
            coverage_file.mkdir()
        elif mutation == "coverage-empty":
            coverage_file.write_bytes(b"")
        elif mutation == "coverage-corrupt":
            coverage_file.write_bytes(b"not a native coverage database")
        elif mutation == "coverage-lines":
            shutil.copyfile(line_data, coverage_file)
        elif mutation == "coverage-unmeasured":
            shutil.copyfile(empty_data, coverage_file)
    with pytest.raises(ValueError):
        partition_artifact_reports(root, lane, load_ci_workflow_policy().partitions)


@pytest.mark.parametrize("lane,count", [("3.14", 8), ("3.12", 0), ("3.12", True)])
def test_unknown_or_nonpositive_artifact_contract_is_refused(
    native_artifacts: tuple[Path, Path, Path], lane: str, count: int
) -> None:
    """An invalid artifact contract cannot admit a reduced native input set."""
    root, _, _ = native_artifacts
    with pytest.raises(ValueError):
        partition_artifact_reports(root, lane, count)
