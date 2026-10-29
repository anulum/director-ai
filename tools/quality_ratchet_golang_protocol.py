# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native Go documentation snapshot adapter
"""Qualify original Go parser and Revive reports against complete source identities."""

from __future__ import annotations

import math
import re
import subprocess

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import (
    SCHEMA,
    Finding,
    decode_quality_json,
    parse_snapshot,
)

GO_VERSION = "go1.27.1"
REVIVE_VERSION = "version 1.17.0"


def qualify_go_build_report(build: subprocess.CompletedProcess[str]) -> None:
    """Require successful silent completion of the actual native parser build.

    Parameters
    ----------
    build : subprocess.CompletedProcess[str]
        Original native compiler process report.

    Raises
    ------
    ValueError
        If compilation failed or emitted unexpected output.
    """
    if build.returncode != 0 or build.stdout or build.stderr:
        raise ValueError("Native Go documentation parser could not be built")


def parse_go_tool_versions(
    version: subprocess.CompletedProcess[str],
    lint_version: subprocess.CompletedProcess[str],
) -> str:
    """Qualify actual compiler and convention-checker version observations.

    Parameters
    ----------
    version : subprocess.CompletedProcess[str]
        Original native Go version process report.
    lint_version : subprocess.CompletedProcess[str]
        Original native Revive version process report.

    Returns
    -------
    str
        Qualified actual Go version including its target operating system and architecture.

    Raises
    ------
    ValueError
        If native status, stderr or exact tool identities cannot qualify.
    """
    if (
        version.returncode != 0
        or version.stderr
        or re.fullmatch(
            r"go version " + re.escape(GO_VERSION) + r" [a-z0-9]+/[a-z0-9]+",
            version.stdout.strip(),
        )
        is None
        or lint_version.returncode != 0
        or lint_version.stderr
        or lint_version.stdout.strip() != REVIVE_VERSION
    ):
        raise ValueError(
            "Native Go or Revive version differs from the qualified profile"
        )
    return version.stdout.strip()


def parse_go_parser_report(
    native: subprocess.CompletedProcess[str],
    sources: dict[str, str],
    identity: str,
) -> tuple[Finding, ...]:
    """Bind the complete original native declaration report to actual Go source bytes.

    Parameters
    ----------
    native : subprocess.CompletedProcess[str]
        Original native parser process status and output.
    sources : dict[str, str]
        Complete maintained source inventory and exact byte digests.
    identity : str
        Actual qualified compiler, helper and configuration fingerprint.

    Returns
    -------
    tuple[Finding, ...]
        Every validated original source-bound declaration or directive finding.

    Raises
    ------
    ValueError
        If process completion, source custody, schema, findings or status disagree.
    """
    if native.returncode not in (0, 1) or native.stderr:
        raise ValueError("Native Go parser did not completely analyse source")
    report = object_mapping(decode_quality_json(native.stdout.encode()))
    if (
        set(report) != {"schema", "sources", "findings"}
        or report["schema"] != "quality-ratchet.go-ast.v1"
        or report["sources"] != sources
        or not isinstance(report["findings"], list)
    ):
        raise ValueError(
            "Native Go parser report is not bound to complete source bytes"
        )
    findings = []
    for raw in report["findings"]:
        item = object_mapping(raw)
        if (
            set(item) != {"path", "rule", "line", "column", "message"}
            or not isinstance(item.get("rule"), str)
            or item["rule"]
            not in {
                "source-directive",
                "declaration-documentation",
                "member-documentation",
            }
        ):
            raise ValueError("Invalid original native Go parser diagnostic")
        findings.append({**item, "cell": None})
    if bool(findings) != bool(native.returncode):
        raise ValueError("Native Go parser status disagrees with its findings")
    return parse_snapshot(
        {
            "schema": SCHEMA,
            "profile": "go-docs",
            "identity": identity,
            "sources": sources,
            "findings": findings,
            "complete": True,
            "advisories": [],
        }
    ).findings


def _revive_position(value: object) -> tuple[str, int, int, int]:
    """Require original native file, byte offset and positive physical coordinates."""
    position = object_mapping(value)
    filename = position.get("Filename")
    if set(position) != {"Filename", "Offset", "Line", "Column"} or not isinstance(
        filename, str
    ):
        raise ValueError("Invalid original native Revive source position")
    values = []
    for name, minimum in (("Offset", 0), ("Line", 1), ("Column", 1)):
        number = position[name]
        if type(number) is not int or number < minimum:
            raise ValueError("Invalid original native Revive source coordinate")
        values.append(number)
    return filename, values[0], values[1], values[2]


def _revive_findings(data: str) -> list[dict[str, object]]:
    """Retain original native Revive locations, including its null clean output."""
    original = decode_quality_json(data.encode())
    if original is None:
        return []
    if not isinstance(original, list):
        raise ValueError("Invalid original native Revive diagnostic list")
    findings = []
    for value in original:
        item = object_mapping(value)
        if set(item) != {
            "Severity",
            "Failure",
            "RuleName",
            "Category",
            "Position",
            "Confidence",
            "ReplacementLine",
        }:
            raise ValueError("Invalid original native Revive diagnostic members")
        position = object_mapping(item["Position"])
        if set(position) != {"Start", "End"}:
            raise ValueError("Incomplete original native Revive source interval")
        start = _revive_position(position["Start"])
        end = _revive_position(position["End"])
        if (
            start[0] != end[0]
            or end[1] < start[1]
            or end[2] < start[2]
            or end[2] == start[2]
            and end[3] < start[3]
        ):
            raise ValueError("Inconsistent original native Revive source interval")
        confidence = item["Confidence"]
        rule, message = (
            item.get("RuleName"),
            item.get("Failure"),
        )
        if (
            item.get("Severity") != "warning"
            or item.get("Category") != "comments"
            or rule not in ("exported", "package-comments")
            or not isinstance(rule, str)
            or not isinstance(message, str)
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(item["ReplacementLine"], str)
        ):
            raise ValueError("Invalid original native Revive documentation finding")
        findings.append(
            {
                "path": start[0],
                "rule": "revive-" + rule,
                "line": start[2],
                "column": start[3],
                "message": message,
                "cell": None,
            }
        )
    return findings


def parse_revive_report(
    lint: subprocess.CompletedProcess[str],
    sources: dict[str, str],
    identity: str,
) -> tuple[Finding, ...]:
    """Qualify original native convention diagnostics without process interception.

    Parameters
    ----------
    lint : subprocess.CompletedProcess[str]
        Actual native Revive process report.
    sources : dict[str, str]
        Complete original source inventory and byte digests.
    identity : str
        Qualified native compiler, helper and configuration fingerprint.

    Returns
    -------
    tuple[Finding, ...]
        Original convention findings bound to maintained source bytes.

    Raises
    ------
    ValueError
        If process completion, original native records or source bindings fail.
    """
    if lint.returncode not in (0, 1) or lint.stderr:
        raise ValueError("Native Revive did not completely analyse source")
    original = _revive_findings(lint.stdout)
    if bool(original) != bool(lint.returncode):
        raise ValueError("Native Revive status disagrees with its findings")
    return parse_snapshot(
        {
            "schema": SCHEMA,
            "profile": "go-docs",
            "identity": identity,
            "sources": sources,
            "findings": original,
            "complete": True,
            "advisories": [],
        }
    ).findings
