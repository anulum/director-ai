# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — immutable pytest scheduling advice contracts
"""Exercise real timing commands and native worker consumption of scheduling hints."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci_workflow_inventory import ROOT
from tools.pytest_partition_timings import (
    COLD_SNAPSHOT,
    LANE_SCHEMA,
    PYTHON_LANES,
    TIMING_SCHEMA,
    merge_timing_reports,
    read_timing_snapshot,
)


@pytest.fixture
def timing_advice(tmp_path: Path) -> Path:
    """Write explicit scheduling inputs, never fabricated execution receipts."""
    root = tmp_path / "advice"
    for lane in PYTHON_LANES:
        target = root / lane / "timings.json"
        target.parent.mkdir(parents=True)
        target.write_text(
            json.dumps(
                {
                    "schema": LANE_SCHEMA,
                    "revision": "a" * 40,
                    "python_version": lane,
                    "weights": {"tests/test_types.py": 1.0},
                }
            )
        )
    return root


def test_cold_and_complete_warm_snapshot_identity(
    tmp_path: Path, timing_advice: Path
) -> None:
    """Read exact advice bytes across all lanes without authorising any test result."""
    cold = tmp_path / "cold.json"
    cold.write_bytes(COLD_SNAPSHOT)
    for lane in PYTHON_LANES:
        assert read_timing_snapshot(None, lane) == read_timing_snapshot(cold, lane)
    output = tmp_path / "warm.json"
    merge_timing_reports(timing_advice, "a" * 40, output)
    expected_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    for lane in PYTHON_LANES:
        assert read_timing_snapshot(output, lane) == (
            {"tests/test_types.py": 1.0},
            expected_hash,
        )
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        merge_timing_reports(timing_advice, "a" * 40, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b'{"schema":1,"schema":1}',
        b'{"schema":NaN}',
        b"{}",
        json.dumps({"schema": "old", "source_revision": None, "lanes": {}}).encode(),
        json.dumps(
            {"schema": TIMING_SCHEMA, "source_revision": True, "lanes": {}}
        ).encode(),
        json.dumps(
            {"schema": TIMING_SCHEMA, "source_revision": "bad", "lanes": {}}
        ).encode(),
        json.dumps(
            {"schema": TIMING_SCHEMA, "source_revision": None, "lanes": []}
        ).encode(),
        json.dumps(
            {"schema": TIMING_SCHEMA, "source_revision": None, "lanes": {"3.11": {}}}
        ).encode(),
    ],
)
def test_malformed_snapshot_is_not_a_cold_fallback(tmp_path: Path, raw: bytes) -> None:
    """Reject present invalid history instead of silently changing worker assignment."""
    path = tmp_path / "snapshot.json"
    path.write_bytes(raw)
    with pytest.raises((ValueError, KeyError)):
        read_timing_snapshot(path, "3.12")


@pytest.mark.parametrize(
    "weights",
    [
        [],
        {"": 1.0},
        {"tests/test_types.py": True},
        {"tests/test_types.py": "1"},
        {"tests/test_types.py": 0.0},
        {"tests/test_types.py": -1.0},
        {"tests/test_types.py": 1e309},
        {"tests/test_types.py": 10**1000},
    ],
)
def test_invalid_hint_in_another_lane_is_also_refused(
    tmp_path: Path, weights: object
) -> None:
    """Validate the entire shared snapshot before the current lane starts testing."""
    path = tmp_path / "snapshot.json"
    lanes: dict[str, object] = {lane: {} for lane in PYTHON_LANES}
    lanes["3.13"] = weights
    path.write_text(
        json.dumps(
            {"schema": TIMING_SCHEMA, "source_revision": "a" * 40, "lanes": lanes}
        )
    )
    with pytest.raises(ValueError):
        read_timing_snapshot(path, "3.11")


def test_unattributed_hints_absent_files_and_unknown_lanes_are_refused(
    tmp_path: Path,
) -> None:
    """Only explicit None means cold; supplied history needs actual provenance."""
    path = tmp_path / "missing.json"
    with pytest.raises(OSError):
        read_timing_snapshot(path, "3.12")
    with pytest.raises(ValueError, match="Unsupported Python"):
        read_timing_snapshot(None, "3.10")
    path.write_text(
        json.dumps(
            {
                "schema": TIMING_SCHEMA,
                "source_revision": None,
                "lanes": {lane: {"tests/test_types.py": 1.0} for lane in PYTHON_LANES},
            }
        )
    )
    with pytest.raises(ValueError, match="source revision"):
        read_timing_snapshot(path, "3.12")


@pytest.mark.parametrize(
    "mutation", ["missing", "revision", "schema", "duplicate", "unknown"]
)
def test_incomplete_or_mixed_lane_advice_is_refused(
    tmp_path: Path, timing_advice: Path, mutation: str
) -> None:
    """A retained snapshot requires every distinct lane from the same producer commit."""
    path = timing_advice / "3.13" / "timings.json"
    data = json.loads(path.read_text())
    if mutation == "missing":
        path.rename(path.with_suffix(".not-json"))
    else:
        key, value = {
            "revision": ("revision", "b" * 40),
            "schema": ("schema", "old"),
            "duplicate": ("python_version", "3.11"),
            "unknown": ("python_version", "3.14"),
        }[mutation]
        data[key] = value
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        merge_timing_reports(timing_advice, "a" * 40, tmp_path / "output.json")
    assert not (tmp_path / "output.json").exists()
    with pytest.raises(ValueError, match="exact timing producer"):
        merge_timing_reports(timing_advice, "invalid", tmp_path / "output.json")


def test_actual_timing_cli_and_native_worker_share_the_snapshot(
    tmp_path: Path, timing_advice: Path
) -> None:
    """Run all real timing commands, then execute real core tests with warm advice."""
    cold, warm = tmp_path / "cold.json", tmp_path / "warm.json"
    commands = [
        ["cold", "--output", str(cold)],
        ["validate", "--input", str(cold)],
        [
            "record",
            "--input",
            str(timing_advice),
            "--revision",
            "a" * 40,
            "--output",
            str(warm),
        ],
    ]
    for arguments in commands:
        result = subprocess.run(
            [sys.executable, "-m", "tools.pytest_partition_timings", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    original = cold.read_bytes()
    refused = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.pytest_partition_timings",
            "cold",
            "--output",
            str(cold),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 1
    assert "snapshot refused" in refused.stdout
    assert cold.read_bytes() == original
    report = tmp_path / "partition.json"
    native = subprocess.run(
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
            "b" * 40,
            "--partition-timings",
            str(warm),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stdout + native.stderr
    data = json.loads(report.read_text())
    assert data["selected"] == data["collected"]
    assert (
        data["timing_snapshot_sha256"] == hashlib.sha256(warm.read_bytes()).hexdigest()
    )
    assert all(
        type(row["duration"]) is float and row["duration"] >= 0
        for row in data["results"]
    )


@pytest.mark.parametrize("raw", [None, b"{broken history"])
def test_actual_worker_refuses_absent_or_corrupt_shared_history(
    tmp_path: Path, raw: bytes | None
) -> None:
    """An explicit bad snapshot cannot silently execute a differently assigned worker."""
    snapshot = tmp_path / "snapshot.json"
    if raw is not None:
        snapshot.write_bytes(raw)
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
    assert result.returncode != 0
    assert "Common test timing snapshot refused" in result.stderr
    assert not report.exists()
