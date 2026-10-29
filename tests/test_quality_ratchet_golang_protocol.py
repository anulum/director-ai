# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — original native Go process report tests
"""Qualify actual Go/Revive reports and reject malformed records at consumed boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.test_quality_ratchet_golang import go_tools as go_tools
from tools.quality_ratchet_golang import REVIVE_CONFIGURATION
from tools.quality_ratchet_golang_protocol import (
    parse_go_parser_report,
    parse_go_tool_versions,
    parse_revive_report,
)


@dataclass(frozen=True)
class NativeGoReports:
    """Keep real process reports and the exact original source identities together.

    Attributes
    ----------
    version : subprocess.CompletedProcess[str]
        Actual Go compiler version report.
    lint_version : subprocess.CompletedProcess[str]
        Actual Revive version report.
    native : subprocess.CompletedProcess[str]
        Actual declaration diagnostic report.
    lint : subprocess.CompletedProcess[str]
        Actual convention diagnostic report.
    sources : dict[str, str]
        Original diagnostic source bytes.
    clean_native : subprocess.CompletedProcess[str]
        Actual clean declaration report.
    clean_lint : subprocess.CompletedProcess[str]
        Actual clean convention report.
    clean_sources : dict[str, str]
        Original clean source bytes.
    """

    version: subprocess.CompletedProcess[str]
    lint_version: subprocess.CompletedProcess[str]
    native: subprocess.CompletedProcess[str]
    lint: subprocess.CompletedProcess[str]
    sources: dict[str, str]
    clean_native: subprocess.CompletedProcess[str]
    clean_lint: subprocess.CompletedProcess[str]
    clean_sources: dict[str, str]


@pytest.fixture(scope="module")
def native_go_reports(
    tmp_path_factory: pytest.TempPathFactory, go_tools: tuple[str, str]
) -> NativeGoReports:
    """Build the actual parser and capture original compiler and convention reports.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Runner-owned isolated native report allocation.
    go_tools : tuple[str, str]
        Actual installed Go and Revive executables.

    Returns
    -------
    NativeGoReports
        Original reports from real processes, with no execution substitute.
    """
    root = tmp_path_factory.mktemp("original-native-reports")
    directory = root / "gateway/go"
    directory.mkdir(parents=True)
    (root / "go.mod").write_text("module native-reports\n\ngo 1.27.1\n")
    source = directory / "original.go"
    source.write_text(
        "// Package example supplies a value.\npackage example\n"
        "\t// Wrong documents the value.\nfunc Value() int {return 1}\n"
        "func hidden() int {return 2}\n"
    )
    clean = directory / "clean.go"
    clean.write_text("// Package clean supplies a count.\npackage clean\n")
    config = root / "revive.toml"
    config.write_text(REVIVE_CONFIGURATION)
    helper = Path(__file__).resolve().parents[1] / "tools/quality_ratchet_go"
    binary = root / "native-documentation"
    built = subprocess.run(
        [
            go_tools[0],
            "build",
            "-o",
            str(binary),
            str(helper / "main.go"),
            str(helper / "documentation.go"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert built.returncode == 0 and not built.stdout and not built.stderr
    commands = (
        [go_tools[0], "version"],
        [go_tools[1], "-version"],
        [str(binary), "--", "gateway/go/original.go"],
        [
            go_tools[1],
            "-config",
            str(config),
            "-formatter",
            "json",
            "-set_exit_status",
            "gateway/go/original.go",
        ],
        [str(binary), "--", "gateway/go/clean.go"],
        [
            go_tools[1],
            "-config",
            str(config),
            "-formatter",
            "json",
            "-set_exit_status",
            "gateway/go/clean.go",
        ],
    )
    version, lint_version, native, lint, clean_native, clean_lint = (
        subprocess.run(
            command, cwd=root, text=True, capture_output=True, check=False, timeout=30
        )
        for command in commands
    )
    assert (
        version.returncode,
        lint_version.returncode,
        native.returncode,
        lint.returncode,
        clean_native.returncode,
        clean_lint.returncode,
    ) == (0, 0, 1, 1, 0, 0), (
        native.stderr,
        lint.stderr,
        clean_native.stderr,
        clean_lint.stderr,
    )
    return NativeGoReports(
        version,
        lint_version,
        native,
        lint,
        {"gateway/go/original.go": hashlib.sha256(source.read_bytes()).hexdigest()},
        clean_native,
        clean_lint,
        {"gateway/go/clean.go": hashlib.sha256(clean.read_bytes()).hexdigest()},
    )


def test_original_native_reports_and_clean_results_qualify(
    native_go_reports: NativeGoReports,
) -> None:
    """Accept actual clean and diagnostic results with their original source bindings.

    Parameters
    ----------
    native_go_reports : NativeGoReports
        Actual compiler, parser and convention reports.
    """
    reports = native_go_reports
    assert (
        parse_go_tool_versions(reports.version, reports.lint_version)
        == reports.version.stdout.strip()
    )
    native = parse_go_parser_report(reports.native, reports.sources, "a" * 64)
    convention = parse_revive_report(reports.lint, reports.sources, "a" * 64)
    assert len(native) == len(convention) == 1
    assert native[0].rule == "declaration-documentation" and native[0].line == 5
    assert convention[0].rule == "revive-exported" and convention[0].column == 2
    assert not parse_go_parser_report(
        reports.clean_native, reports.clean_sources, "a" * 64
    )
    assert not parse_revive_report(reports.clean_lint, reports.clean_sources, "a" * 64)


@pytest.mark.parametrize(
    "tool,returncode,stdout,stderr",
    (
        ("go", 2, "go version go1.27.1 linux/amd64", ""),
        ("go", 0, "go version go1.27.1 linux/amd64", "unexpected stderr"),
        ("go", 0, "go version go0.0.0 linux/amd64", ""),
        ("go", 0, "go version go1.27.1 ", ""),
        ("revive", 2, "version 1.17.0", ""),
        ("revive", 0, "version 1.17.0", "unexpected stderr"),
        ("revive", 0, "version 0.0.0", ""),
    ),
)
def test_version_record_controls_refuse(
    native_go_reports: NativeGoReports,
    tool: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    """Reject explicit malformed version records without replacing a native tool.

    Parameters
    ----------
    native_go_reports : NativeGoReports
        Original native reports.
    tool : str
        Native version report selected for corruption.
    returncode : int
        Negative-control process status.
    stdout : str
        Negative-control version output.
    stderr : str
        Negative-control stderr.
    """
    reports = {
        "go": native_go_reports.version,
        "revive": native_go_reports.lint_version,
    }
    reports[tool] = subprocess.CompletedProcess(
        reports[tool].args, returncode, stdout, stderr
    )
    with pytest.raises(ValueError, match="version differs"):
        parse_go_tool_versions(reports["go"], reports["revive"])


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema", "other"),
        ("sources", {}),
        ("findings", None),
        ("extra", True),
    ),
)
def test_native_envelope_controls_refuse(
    native_go_reports: NativeGoReports,
    field: str,
    value: object,
) -> None:
    """Reject incomplete source scope and malformed original parser envelopes.

    Parameters
    ----------
    native_go_reports : NativeGoReports
        Actual complete parser report.
    field : str
        Original envelope member to corrupt.
    value : object
        Malformed replacement value.
    """
    payload = json.loads(native_go_reports.native.stdout)
    payload[field] = value
    control = subprocess.CompletedProcess(
        native_go_reports.native.args, 1, json.dumps(payload), ""
    )
    with pytest.raises(ValueError):
        parse_go_parser_report(control, native_go_reports.sources, "a" * 64)


@pytest.mark.parametrize(
    "field,value",
    (
        ("extra", True),
        ("rule", []),
        ("rule", "unknown"),
        ("line", 0),
        ("column", True),
        ("path", "gateway/go/absent.go"),
        ("message", ""),
    ),
)
def test_native_diagnostic_controls_refuse(
    native_go_reports: NativeGoReports,
    field: str,
    value: object,
) -> None:
    """Refuse unbound native findings and retain typed format-error refusals.

    Parameters
    ----------
    native_go_reports : NativeGoReports
        Actual native declaration finding.
    field : str
        Original finding member to corrupt.
    value : object
        Explicit malformed diagnostic value.
    """
    payload = json.loads(native_go_reports.native.stdout)
    payload["findings"][0][field] = value
    control = subprocess.CompletedProcess(
        native_go_reports.native.args, 1, json.dumps(payload), ""
    )
    with pytest.raises(ValueError):
        parse_go_parser_report(control, native_go_reports.sources, "a" * 64)


@pytest.mark.parametrize(
    "profile,returncode,stderr",
    (
        ("native", 2, ""),
        ("native", 1, "unexpected stderr"),
        ("native", 0, ""),
        ("revive", 2, ""),
        ("revive", 1, "unexpected stderr"),
        ("revive", 0, ""),
    ),
)
def test_original_process_status_controls_refuse(
    native_go_reports: NativeGoReports,
    profile: str,
    returncode: int,
    stderr: str,
) -> None:
    """Reject native failures and statuses that contradict original diagnostic output.

    Parameters
    ----------
    native_go_reports : NativeGoReports
        Original process reports.
    profile : str
        Native parser or Revive report under test.
    returncode : int
        Explicit negative-control status.
    stderr : str
        Explicit negative-control stderr.
    """
    reports = {"native": native_go_reports.native, "revive": native_go_reports.lint}
    original = reports[profile]
    control = subprocess.CompletedProcess(
        original.args, returncode, original.stdout, stderr
    )
    with pytest.raises(ValueError):
        {"native": parse_go_parser_report, "revive": parse_revive_report}[profile](
            control,
            native_go_reports.sources,
            "a" * 64,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("extra", True),
        ("Severity", "error"),
        ("Category", "unknown"),
        ("RuleName", []),
        ("RuleName", "unknown"),
        ("Failure", 1),
        ("Confidence", True),
        ("Confidence", "0.8"),
        ("Confidence", -1),
        ("Confidence", 2),
        ("ReplacementLine", []),
        ("Position", {}),
    ),
)
def test_revive_member_controls_refuse(
    native_go_reports: NativeGoReports,
    field: str,
    value: object,
) -> None:
    """Refuse malformed original convention metadata without dropping required members.

    Parameters
    ----------
    native_go_reports : NativeGoReports
        Actual original convention report.
    field : str
        Native convention member selected for corruption.
    value : object
        Invalid original member value.
    """
    payload = json.loads(native_go_reports.lint.stdout)
    payload[0][field] = value
    control = subprocess.CompletedProcess(
        native_go_reports.lint.args, 1, json.dumps(payload), ""
    )
    with pytest.raises(ValueError):
        parse_revive_report(control, native_go_reports.sources, "a" * 64)


@pytest.mark.parametrize(
    "part,field,value",
    (
        ("Start", "extra", True),
        ("Start", "Filename", 1),
        ("Start", "Offset", -1),
        ("Start", "Line", 0),
        ("Start", "Column", True),
        ("End", "Filename", "different.go"),
        ("End", "Offset", 0),
        ("End", "Line", 1),
        ("End", "Column", 1),
    ),
)
def test_revive_position_controls_refuse(
    native_go_reports: NativeGoReports,
    part: str,
    field: str,
    value: object,
) -> None:
    """Reject missing physical coordinates and inconsistent original source intervals.

    Parameters
    ----------
    native_go_reports : NativeGoReports
        Original native convention finding.
    part : str
        Start or End position selected for corruption.
    field : str
        Original coordinate member to alter.
    value : object
        Explicit invalid source coordinate or interval member.
    """
    payload = json.loads(native_go_reports.lint.stdout)
    payload[0]["Position"][part][field] = value
    control = subprocess.CompletedProcess(
        native_go_reports.lint.args, 1, json.dumps(payload), ""
    )
    with pytest.raises(ValueError):
        parse_revive_report(control, native_go_reports.sources, "a" * 64)


@pytest.mark.parametrize("profile", ("native", "revive"))
@pytest.mark.parametrize(
    "corruption", ("object", "scalar", "duplicate", "truncated", "missing-member")
)
def test_native_json_framing_controls_refuse(
    native_go_reports: NativeGoReports, profile: str, corruption: str
) -> None:
    """Reject malformed native framing and repeated original diagnostic members.

    Parameters
    ----------
    native_go_reports : NativeGoReports
        Actual native reports, with original source bindings.
    profile : str
        Actual parser or convention report selected for corruption.
    corruption : str
        Invalid root, duplicate member or incomplete JSON frame.
    """
    original = {"native": native_go_reports.native, "revive": native_go_reports.lint}[
        profile
    ]
    incomplete = json.loads(original.stdout)
    if profile == "native":
        del incomplete["schema"]
    else:
        del incomplete[0]["Failure"]
    member = {"native": '"schema":', "revive": '"Severity":'}[profile]
    stdout = {
        "object": "{}",
        "scalar": '"not an original report"',
        "duplicate": original.stdout.replace(
            member, member + '"duplicate",' + member, 1
        ),
        "truncated": original.stdout[: len(original.stdout) // 2],
        "missing-member": json.dumps(incomplete),
    }[corruption]
    control = subprocess.CompletedProcess(
        original.args, original.returncode, stdout, ""
    )
    with pytest.raises(ValueError):
        {"native": parse_go_parser_report, "revive": parse_revive_report}[profile](
            control, native_go_reports.sources, "a" * 64
        )
