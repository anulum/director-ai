# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — deterministic complete pytest partitions
"""Partition real collected cases and record every selected test outcome."""

from __future__ import annotations

import json
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.ci_workflow_inventory import ROOT, load_ci_workflow_policy
from tools.pytest_partition_schedule import partition_assignments
from tools.pytest_partition_schedule import partition_for_nodeid as partition_for_nodeid
from tools.pytest_partition_timings import read_timing_snapshot


@dataclass
class PartitionState:
    """Keep the collected inventory and real results for one pytest process.

    Attributes
    ----------
    index, count, rows_per_task : int
        Declared complete partition contract.
    revision : str
        Exact checkout SHA qualified by this run.
    report : Path
        Fresh JSON report path, written only at terminal session completion.
    targets : list[str]
        Actual command-line collection roots.
    timings : dict[str, float]
        Validated scheduling advice shared by every worker in this lane.
    timing_snapshot_sha256 : str
        Exact bytes of the common snapshot used for the assignment.
    collected, selected : list[str]
        Complete and selected native node IDs.
    results : list[dict[str, object]]
        Native setup, call and teardown results with benchmark evidence.
    """

    index: int
    count: int
    rows_per_task: int
    revision: str
    report: Path
    targets: list[str]
    timings: dict[str, float]
    timing_snapshot_sha256: str
    collected: list[str] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    results: list[dict[str, object]] = field(default_factory=list)


_partition: PartitionState | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register explicit partition options on the actual pytest CLI.

    Parameters
    ----------
    parser : pytest.Parser
        Active pytest command-line parser.
    """
    group = parser.getgroup("complete-test-partitions")
    group.addoption("--partition-index", type=int, default=None)
    group.addoption("--partition-report", type=Path, default=None)
    group.addoption("--partition-revision", default=None)
    group.addoption("--partition-timings", type=Path, default=None)


def pytest_configure(config: pytest.Config) -> None:
    """Require explicit evidence paths and exact revision for partitioned runs.

    Parameters
    ----------
    config : pytest.Config
        Actual parsed pytest invocation.

    Raises
    ------
    pytest.UsageError
        If partition arguments are incomplete or outside the policy.
    """
    global _partition
    index: object = config.getoption("partition_index")
    report: object = config.getoption("partition_report")
    revision: object = config.getoption("partition_revision")
    timings: object = config.getoption("partition_timings")
    _partition = None
    if index is None and report is None and revision is None and timings is None:
        return
    policy = load_ci_workflow_policy(ROOT)
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < policy.partitions
        or not isinstance(report, Path)
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or report.exists()
        or (timings is not None and not isinstance(timings, Path))
    ):
        raise pytest.UsageError(
            "A valid partition requires a fresh report path and exact SHA"
        )
    selection_options = (
        "keyword",
        "markexpr",
        "ignore",
        "ignore_glob",
        "deselect",
        "lf",
        "stepwise",
        "collectonly",
    )
    if any(config.getoption(option, default=False) for option in selection_options):
        raise pytest.UsageError("Partitioned runs cannot filter the collected cases")
    try:
        weights, snapshot_digest = read_timing_snapshot(
            timings, ".".join(platform.python_version().split(".")[:2])
        )
    except (ValueError, KeyError, OSError) as exc:
        raise pytest.UsageError("Common test timing snapshot refused") from exc
    _partition = PartitionState(
        index,
        policy.partitions,
        policy.rows_per_task,
        revision,
        report,
        list(config.args),
        weights,
        snapshot_digest,
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Select a disjoint subset from the complete native collection.

    Parameters
    ----------
    config : pytest.Config
        Active pytest hooks and configuration.
    items : list[pytest.Item]
        Native collection, before any case is deselected by this plugin.

    Raises
    ------
    pytest.UsageError
        If case IDs repeat or the selected partition is empty.
    """
    if _partition is None:
        return
    state = _partition
    state.collected = [item.nodeid for item in items]
    if len(set(state.collected)) != len(state.collected):
        raise pytest.UsageError("Duplicate native pytest case IDs")
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    assignments = partition_assignments(
        state.collected, state.count, state.rows_per_task, state.timings
    )
    for item in items:
        owner = assignments[item.nodeid]
        (selected if owner == state.index else deselected).append(item)
    if not selected:
        raise pytest.UsageError("The selected test partition has no native cases")
    state.selected = [item.nodeid for item in selected]
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Capture native phases and the actual HaluEval review evidence.

    Parameters
    ----------
    report : pytest.TestReport
        Native setup, call or teardown result.
    """
    if _partition is None:
        return
    evidence = [value for key, value in report.user_properties if key == "halueval"]
    _partition.results.append(
        {
            "nodeid": report.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
            "duration": report.duration,
            "halueval": evidence,
        }
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write terminal collection and result evidence without overwriting history.

    Parameters
    ----------
    session : pytest.Session
        Terminal native pytest session.
    exitstatus : int
        Actual pytest exit code, including collection and teardown failures.
    """
    if _partition is None:
        return
    state = _partition
    state.report.parent.mkdir(parents=True, exist_ok=True)
    with state.report.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema": "director-test-partition.v2",
                "revision": state.revision,
                "python_version": platform.python_version(),
                "index": state.index,
                "count": state.count,
                "rows_per_task": state.rows_per_task,
                "exit_code": int(exitstatus),
                "targets": state.targets,
                "timing_snapshot_sha256": state.timing_snapshot_sha256,
                "collected": state.collected,
                "selected": state.selected,
                "results": state.results,
            },
            stream,
            indent=2,
            allow_nan=False,
        )


def pytest_unconfigure(config: pytest.Config) -> None:
    """Release process-local report state after the native session exits.

    Parameters
    ----------
    config : pytest.Config
        Completed pytest configuration.
    """
    global _partition
    _partition = None
