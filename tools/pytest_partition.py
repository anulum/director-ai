# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — deterministic complete pytest partitions
"""Partition real collected cases and record every selected test outcome."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.ci_workflow_inventory import ROOT, load_ci_workflow_policy


def partition_for_nodeid(nodeid: str, count: int, rows_per_task: int) -> int:
    """Assign a real case to one worker, preserving module fixture locality.

    Parameters
    ----------
    nodeid : str
        Exact pytest collection identifier.
    count, rows_per_task : int
        Positive partition count and complete HaluEval task row count.

    Returns
    -------
    int
        Zero-based exclusive worker index. Each worker receives one original
        HaluEval range per task; other cases stay with their owning module.

    Raises
    ------
    ValueError
        If the partition contract or a full HaluEval case ID is invalid.
    """
    if count <= 0 or rows_per_task <= 0 or rows_per_task % count:
        raise ValueError("Invalid complete test partition contract")
    if nodeid.startswith("tests/test_halueval_benchmark.py::test_halueval_full"):
        match = re.fullmatch(
            r"tests/test_halueval_benchmark\.py::test_halueval_full\[(qa|summarization|dialogue)-(\d+)\]",
            nodeid,
        )
        size = rows_per_task // count
        if match is None:
            raise ValueError("Full HaluEval case lacks a declared task range")
        offset = int(match.group(2))
        if offset >= rows_per_task or offset % size:
            raise ValueError("Full HaluEval case has an undeclared range")
        return offset // size
    module = nodeid.split("::", maxsplit=1)[0]
    return (
        int.from_bytes(hashlib.sha256(module.encode("utf-8")).digest(), "big") % count
    )


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
    _partition = None
    if index is None and report is None and revision is None:
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
    _partition = PartitionState(
        index,
        policy.partitions,
        policy.rows_per_task,
        revision,
        report,
        list(config.args),
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
    for item in items:
        owner = partition_for_nodeid(item.nodeid, state.count, state.rows_per_task)
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
                "schema": "director-test-partition.v1",
                "revision": state.revision,
                "python_version": platform.python_version(),
                "index": state.index,
                "count": state.count,
                "rows_per_task": state.rows_per_task,
                "exit_code": int(exitstatus),
                "targets": state.targets,
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
