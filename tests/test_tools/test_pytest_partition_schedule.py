# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native pytest scheduling contracts
"""Schedule cases collected from real core and model-backed test modules."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tools.ci_workflow_inventory import ROOT, load_ci_workflow_policy
from tools.pytest_partition_schedule import (
    HALUEVAL_PREFIX,
    partition_assignments,
    partition_for_nodeid,
)


@pytest.fixture(scope="module")
def native_cases() -> list[str]:
    """Collect the real installed pytest entry point without model execution."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "tests/test_types.py",
            "tests/test_config.py",
            "tests/test_halueval_benchmark.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cases = [line for line in result.stdout.splitlines() if line.startswith("tests/")]
    assert len(cases) == len(set(cases))
    return cases


def test_complete_native_row_collection_and_cold_assignment(
    native_cases: list[str],
) -> None:
    """Retain every real row, both-label test body and ordinary module owner."""
    policy = load_ci_workflow_policy()
    actual_rows = {case for case in native_cases if case.startswith(HALUEVAL_PREFIX)}
    assert actual_rows == {
        f"{HALUEVAL_PREFIX}[{task}-{index}]"
        for task in policy.dataset_sha256
        for index in range(policy.rows_per_task)
    }
    assignments = partition_assignments(native_cases, 8, 200, {})
    assert list(assignments) == native_cases
    for task in policy.dataset_sha256:
        for worker in range(8):
            assert (
                sum(
                    assignments[case] == worker
                    for case in actual_rows
                    if case.startswith(f"{HALUEVAL_PREFIX}[{task}-")
                )
                == 25
            )
    for module in ("tests/test_types.py", "tests/test_config.py"):
        owners = {
            assignments[case] for case in native_cases if case.startswith(module + "::")
        }
        assert owners == {partition_for_nodeid(module, 8, 200)}


def test_warm_schedule_is_deterministic_and_uses_every_native_case(
    native_cases: list[str],
) -> None:
    """Deliberate timing hints change placement, never the discovered inventory."""
    hints = {"tests/test_config.py": 1000.0, "tests/test_types.py": 1.0}
    assignments = partition_assignments(native_cases, 8, 200, hints)
    assert set(assignments) == set(native_cases)
    assert set(assignments.values()) == set(range(8))
    assert assignments == partition_assignments(
        list(reversed(native_cases)), 8, 200, hints
    )
    assert {
        assignments[case]
        for case in native_cases
        if case.startswith("tests/test_config.py::")
    } == {0}


def test_removed_timing_entries_never_change_membership_or_default_weights(
    native_cases: list[str],
) -> None:
    """Ignore stale units while automatically admitting real newly discovered units."""
    hints = {"tests/test_types.py": 4.0}
    current = partition_assignments(native_cases, 8, 200, hints)
    stale = partition_assignments(
        native_cases, 8, 200, {**hints, "tests/test_removed_module.py": 1e100}
    )
    assert current == stale
    assert "tests/test_removed_module.py" not in stale
    assert any(case.startswith("tests/test_config.py::") for case in stale)
    assert partition_assignments(
        native_cases, 8, 200, {"tests/test_removed_module.py": 1.0}
    ) == partition_assignments(native_cases, 8, 200, {})


@pytest.mark.parametrize(
    "count,rows",
    [(0, 200), (8, 0), (3, 200), (True, 200), (8, True), (1.5, 200), (8, "200")],
)
def test_invalid_native_partition_contract_is_refused(count: int, rows: int) -> None:
    """Refuse malformed runtime inputs at the public scheduling boundary."""
    with pytest.raises(ValueError, match="Invalid complete test partition"):
        partition_for_nodeid(
            "tests/test_types.py::TestClamp::test_within_range", count, rows
        )


@pytest.mark.parametrize(
    "nodeid",
    [
        "",
        "README.md",
        HALUEVAL_PREFIX,
        f"{HALUEVAL_PREFIX}[unknown-0]",
        f"{HALUEVAL_PREFIX}[qa-200]",
        f"{HALUEVAL_PREFIX}[qa-01]",
    ],
)
def test_undeclared_case_identity_is_refused(nodeid: str) -> None:
    """An invalid row identity cannot be assigned to a successful native worker."""
    with pytest.raises(ValueError):
        partition_for_nodeid(nodeid, 8, 200)


@pytest.mark.parametrize(
    "seconds", [True, "1", 0.0, -1.0, float("nan"), float("inf"), 10**1000]
)
def test_invalid_timing_weights_are_refused_before_scheduling(
    native_cases: list[str], seconds: float
) -> None:
    """Even stale hints must contain positive finite numeric measurements."""
    with pytest.raises(ValueError, match="positive finite"):
        partition_assignments(
            native_cases, 8, 200, {"tests/test_removed_module.py": seconds}
        )


@pytest.mark.parametrize("name", ["", False])
def test_unnamed_timing_weights_are_refused(native_cases: list[str], name: str) -> None:
    """Refuse malformed scheduling identifiers without touching native membership."""
    with pytest.raises(ValueError, match="positive finite"):
        partition_assignments(native_cases, 8, 200, {name: 1.0})


def test_duplicate_native_discovery_is_refused(native_cases: list[str]) -> None:
    """A repeated real test cannot produce overlapping execution receipts."""
    with pytest.raises(ValueError, match="unique"):
        partition_assignments([*native_cases, native_cases[0]], 8, 200, {})
    with pytest.raises(ValueError, match="nonempty"):
        partition_assignments([], 8, 200, {})


def test_worker_load_overflow_is_refused(native_cases: list[str]) -> None:
    """Finite individual hints cannot silently overflow into infinite worker loads."""
    cases = [case for case in native_cases if case.startswith(HALUEVAL_PREFIX)][:2]
    with pytest.raises(ValueError, match="overflow"):
        partition_assignments(cases, 1, 200, {case: 1e308 for case in cases})
