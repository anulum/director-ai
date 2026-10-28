# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — original native typing report qualification tests
"""Qualify real MyPy reports and refuse malformed controls at the consumed parser."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.quality_ratchet_typing import ENVIRONMENT_CODE, TYPING_FLAGS
from tools.quality_ratchet_typing_protocol import (
    parse_typing_environment,
    parse_typing_result,
)


@dataclass(frozen=True)
class NativeReports:
    """Keep actual native inputs and complete process reports together.

    Attributes
    ----------
    root : Path
        Original compiler working directory.
    config : Path
        Exact original native configuration path.
    configuration : str
        Original INI content.
    sources : dict[str, str]
        Complete source inventory and byte digests.
    version : subprocess.CompletedProcess[str]
        Unaltered native version report.
    environment : subprocess.CompletedProcess[str]
        Unaltered isolated runtime report.
    result : subprocess.CompletedProcess[str]
        Unaltered native error and unused-section report.
    """

    root: Path
    config: Path
    configuration: str
    sources: dict[str, str]
    version: subprocess.CompletedProcess[str]
    environment: subprocess.CompletedProcess[str]
    result: subprocess.CompletedProcess[str]


@pytest.fixture(scope="module")
def native_reports(tmp_path_factory: pytest.TempPathFactory) -> NativeReports:
    """Capture the real pinned compiler, runtime and diagnostic reports once.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Runner-owned isolated allocation.

    Returns
    -------
    NativeReports
        Actual native reports; no compiler execution is replaced.
    """
    root = tmp_path_factory.mktemp("original-typing-reports")
    (root / "src").mkdir()
    source = root / "src/original.py"
    source.write_text("def original(value):\n    return value\n")
    configuration = (
        "[mypy]\nstrict = true\nwarn_unused_configs = true\n"
        "[mypy-never_imported]\nignore_missing_imports = true\n"
    )
    config = root / "mypy.ini"
    config.write_text(configuration)
    commands = (
        [sys.executable, "-B", "-m", "mypy", "--version"],
        [sys.executable, "-I", "-B", "-c", ENVIRONMENT_CODE],
        [
            sys.executable,
            "-B",
            "-m",
            "mypy",
            *TYPING_FLAGS,
            "--config-file",
            str(config),
            "--cache-dir",
            str(root / "cache"),
            "--",
            str(source),
        ],
    )
    version, environment, result = (
        subprocess.run(
            command, cwd=root, text=True, capture_output=True, check=False, timeout=30
        )
        for command in commands
    )
    assert version.returncode == environment.returncode == 0
    assert result.returncode == 1 and not result.stderr
    assert "[mypy-never_imported]" in result.stdout
    return NativeReports(
        root,
        config,
        configuration,
        {"src/original.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        version,
        environment,
        result,
    )


def test_original_reports_qualify(native_reports: NativeReports) -> None:
    """Accept actual absolute native source paths and separate configuration notes.

    Parameters
    ----------
    native_reports : NativeReports
        Original native process reports.
    """
    identity = parse_typing_environment(
        native_reports.version, native_reports.environment
    )
    assert identity == json.loads(native_reports.environment.stdout)
    findings, advisories = parse_typing_result(
        native_reports.result,
        native_reports.root,
        native_reports.sources,
        "a" * 64,
        native_reports.config,
        native_reports.configuration,
    )
    assert len(findings) == 1 and findings[0].path == "src/original.py"
    assert advisories == ("mypy-unused-override:never_imported",)


@pytest.mark.parametrize(
    "returncode,stdout,stderr",
    (
        (2, "mypy 1.20.2 (compiled: yes)", ""),
        (0, "mypy 0.0.0", ""),
        (0, "mypy 1.20.2 (compiled: yes)", "unexpected stderr"),
    ),
)
def test_version_controls_refuse(
    native_reports: NativeReports, returncode: int, stdout: str, stderr: str
) -> None:
    """Reject unqualified version records directly without substituting a compiler.

    Parameters
    ----------
    native_reports : NativeReports
        Original environment used for the independent version control.
    returncode : int
        Negative-control process status.
    stdout : str
        Negative-control version output.
    stderr : str
        Negative-control process stderr.
    """
    control = subprocess.CompletedProcess(
        native_reports.version.args, returncode, stdout, stderr
    )
    with pytest.raises(ValueError, match="version differs"):
        parse_typing_environment(control, native_reports.environment)


@pytest.mark.parametrize("returncode,stderr", ((2, ""), (0, "unexpected stderr")))
def test_environment_process_controls_refuse(
    native_reports: NativeReports, returncode: int, stderr: str
) -> None:
    """Reject incomplete runtime observations before decoding installed metadata.

    Parameters
    ----------
    native_reports : NativeReports
        Actual original runtime report.
    returncode : int
        Negative-control status.
    stderr : str
        Negative-control stderr.
    """
    control = subprocess.CompletedProcess(
        native_reports.environment.args,
        returncode,
        native_reports.environment.stdout,
        stderr,
    )
    with pytest.raises(ValueError, match="could not be identified"):
        parse_typing_environment(native_reports.version, control)


@pytest.mark.parametrize(
    "field,value",
    (
        ("extra", True),
        ("python", None),
        ("python", []),
        ("python", [3, 14, 0]),
        ("python", [3, 12, -1]),
        ("python", [True, 12, 0]),
        ("packages", None),
        ("packages", []),
        ("packages", [None]),
        ("packages", [["invalid"]]),
        ("packages", [["", "1"]]),
        ("packages", [[1, "1"]]),
        ("packages", [["other", "1"]]),
        ("packages", [["mypy", "1.20.2"], ["MyPy", "1.20.3"]]),
    ),
)
def test_environment_member_controls_refuse(
    native_reports: NativeReports, field: str, value: object
) -> None:
    """Retain strict runtime, unique distribution and compiler metadata qualification.

    Parameters
    ----------
    native_reports : NativeReports
        Actual unmodified native environment report.
    field : str
        Original report member to corrupt.
    value : object
        Explicit malformed value; it never qualifies native runtime success.
    """
    payload = json.loads(native_reports.environment.stdout)
    payload[field] = value
    control = subprocess.CompletedProcess(
        native_reports.environment.args, 0, json.dumps(payload), ""
    )
    with pytest.raises(ValueError):
        parse_typing_environment(native_reports.version, control)


@pytest.mark.parametrize(
    "field,value",
    (
        ("extra", True),
        ("severity", "warning"),
        ("file", 1),
        ("code", True),
        ("code", "BadCode"),
        ("code", None),
        ("message", 1),
        ("message", ""),
        ("message", "invalid\x00message"),
        ("hint", {}),
        ("line", True),
        ("end_column", -2),
        ("line", 0),
        ("file", "src/absent.py"),
    ),
)
def test_diagnostic_member_controls_refuse(
    native_reports: NativeReports, field: str, value: object
) -> None:
    """Refuse corrupted native members, positions and unbound source identities.

    Parameters
    ----------
    native_reports : NativeReports
        Actual complete compiler error report.
    field : str
        Original diagnostic member to corrupt.
    value : object
        Explicit malformed diagnostic value.
    """
    lines = native_reports.result.stdout.splitlines()
    item = json.loads(lines[0])
    item[field] = value
    control = subprocess.CompletedProcess(
        native_reports.result.args, 1, json.dumps(item), ""
    )
    with pytest.raises(ValueError):
        parse_typing_result(
            control,
            native_reports.root,
            native_reports.sources,
            "a" * 64,
            native_reports.config,
            native_reports.configuration,
        )


@pytest.mark.parametrize(
    "returncode,stdout,stderr",
    (
        (2, "", ""),
        (0, "", "unqualified stderr"),
        (1, "", ""),
        (0, "ORIGINAL", ""),
        (1, "REPEATED", ""),
        (1, "DUPLICATE", ""),
    ),
)
def test_result_process_and_structure_controls_refuse(
    native_reports: NativeReports,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    """Reject inconsistent statuses, duplicate findings and repeated JSON members.

    Parameters
    ----------
    native_reports : NativeReports
        Actual complete error report.
    returncode : int
        Explicit negative-control status.
    stdout : str
        Empty output or selector for a corruption of the actual diagnostic.
    stderr : str
        Explicit negative-control stderr.
    """
    original = native_reports.result.stdout.splitlines()[0]
    controls = {
        "": "",
        "ORIGINAL": original,
        "REPEATED": original + "\n" + original,
        "DUPLICATE": original.replace('"code":', '"code": null, "code":', 1),
    }
    control = subprocess.CompletedProcess(
        native_reports.result.args, returncode, controls[stdout], stderr
    )
    with pytest.raises(ValueError):
        parse_typing_result(
            control,
            native_reports.root,
            native_reports.sources,
            "a" * 64,
            native_reports.config,
            native_reports.configuration,
        )


@pytest.mark.parametrize(
    "before,after",
    (
        ("mypy.ini:", "foreign.ini:"),
        ("never_imported", "undeclared"),
        ("[mypy-never_imported]", ""),
        ("[mypy-never_imported]", "[mypy-never_imported], [mypy-never_imported]"),
        ("[mypy-never_imported]", "module=never_imported"),
        ("\n", "\nREPEAT"),
    ),
)
def test_advisory_controls_refuse(
    native_reports: NativeReports, before: str, after: str
) -> None:
    """Refuse an advisory detached from its actual INI path or declared module set.

    Parameters
    ----------
    native_reports : NativeReports
        Actual diagnostic and original unused-section advisory.
    before : str
        Native advisory substring selected for corruption.
    after : str
        Invalid replacement; REPEAT selects a repeated whole advisory.
    """
    original = native_reports.result.stdout.splitlines()[-1] + "\n"
    replacement = after.replace("REPEAT", original)
    control = subprocess.CompletedProcess(
        native_reports.result.args, 0, original.replace(before, replacement), ""
    )
    with pytest.raises(ValueError):
        parse_typing_result(
            control,
            native_reports.root,
            native_reports.sources,
            "a" * 64,
            native_reports.config,
            native_reports.configuration,
        )
