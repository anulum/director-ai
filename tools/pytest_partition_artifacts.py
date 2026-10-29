# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — indexed native pytest coverage artifacts
"""Require every indexed coverage bundle before complete-run semantic verification."""

from __future__ import annotations

import json
from pathlib import Path

from coverage import CoverageData
from coverage.exceptions import CoverageException

from tools.ci_workflow_inventory import object_mapping
from tools.pytest_partition_timings import PYTHON_LANES


def partition_artifact_reports(
    directory: Path, python_version: str, count: int
) -> list[Path]:
    """Validate indexed native coverage inputs without accepting test outcomes.

    Parameters
    ----------
    directory : Path
        Fresh download root containing exactly one artifact per expected worker.
    python_version : str
        Supported Python major.minor lane.
    count : int
        Positive number of mandatory workers.

    Returns
    -------
    list of Path
        Indexed native JSON reports, ready for the strict complete-run verifier.
        Readable artifacts alone never prove successful or complete execution.

    Raises
    ------
    ValueError, KeyError, OSError
        If an indexed artifact, report, JUnit or native branch database is absent,
        unexpected, misidentified, empty, unreadable or contains no measured arcs.
    """
    if python_version not in PYTHON_LANES or type(count) is not int or count <= 0:
        raise ValueError("A supported lane and positive worker count are required")
    bundles = [directory / f"test-partition-{python_version}-{i}" for i in range(count)]
    if set(directory.iterdir()) != set(bundles):
        raise ValueError("Every exact indexed native artifact directory is required")
    reports: list[Path] = []
    for index, bundle in enumerate(bundles):
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError("An indexed native artifact must be a real directory")
        report = bundle / "partition.json"
        for required in (report, bundle / "partition.xml"):
            if (
                required.is_symlink()
                or not required.is_file()
                or not required.stat().st_size
            ):
                raise ValueError("Every native report and JUnit artifact is required")
        value: object = json.loads(report.read_bytes())
        identity = object_mapping(value)["index"]
        if type(identity) is not int or identity != index:
            raise ValueError("Native report index differs from its artifact directory")
        databases = sorted(bundle.glob(".coverage.*"))
        if not databases:
            raise ValueError("Every indexed worker requires native coverage data")
        for path in databases:
            if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                raise ValueError(
                    "Native coverage input must be a nonempty regular file"
                )
            try:
                data = CoverageData(basename=str(path))
                data.read()
                if not data.has_arcs() or not any(
                    data.arcs(filename) for filename in data.measured_files()
                ):
                    raise ValueError(
                        "Native coverage must contain measured branch arcs"
                    )
            except CoverageException as exc:
                raise ValueError("Native coverage data is unreadable") from exc
        reports.append(report)
    return reports
