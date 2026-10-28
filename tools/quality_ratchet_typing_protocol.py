# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — source-bound native strict Python typing snapshots
"""Validate original native MyPy process reports before accepting typing findings."""

from __future__ import annotations

import configparser
import re
import subprocess
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import (
    Finding,
    Snapshot,
    decode_quality_json,
    parse_snapshot,
    snapshot_payload,
)

MYPY_VERSION = "mypy 1.20.2 (compiled: yes)"
SUPPORTED_PYTHON_RUNTIMES = frozenset({(3, 11), (3, 12), (3, 13)})


def parse_typing_environment(
    version: subprocess.CompletedProcess[str],
    installed: subprocess.CompletedProcess[str],
) -> dict[str, object]:
    """Qualify actual compiler output and the complete installed runtime identity.

    Parameters
    ----------
    version : subprocess.CompletedProcess[str]
        Native MyPy version process report, without substituted execution.
    installed : subprocess.CompletedProcess[str]
        Isolated native interpreter's original package and runtime report.

    Returns
    -------
    dict[str, object]
        Validated original runtime and package identities. The interpreter may
        use any supported CI runtime; the compiler's language target is separate.

    Raises
    ------
    ValueError
        If a process, version, runtime or distribution report is unqualified.
    """
    if (
        version.returncode != 0
        or version.stderr
        or version.stdout.strip() != MYPY_VERSION
    ):
        raise ValueError("Native MyPy version differs from the qualified profile")
    if installed.returncode != 0 or installed.stderr:
        raise ValueError("Native typing environment could not be identified")
    identity = object_mapping(decode_quality_json(installed.stdout.encode()))
    runtime, packages = identity.get("python"), identity.get("packages")
    if (
        set(identity) != {"python", "packages"}
        or not isinstance(runtime, list)
        or len(runtime) != 3
        or any(type(part) is not int or part < 0 for part in runtime)
        or tuple(runtime[:2]) not in SUPPORTED_PYTHON_RUNTIMES
        or not isinstance(packages, list)
        or not packages
    ):
        raise ValueError("Invalid native typing environment identity")
    names = set()
    for package in packages:
        if (
            not isinstance(package, list)
            or len(package) != 2
            or any(not isinstance(part, str) or not part.strip() for part in package)
        ):
            raise ValueError("Invalid native typing distribution identity")
        name = re.sub(r"[-_.]+", "-", package[0]).lower()
        if name in names:
            raise ValueError("Repeated native typing distribution identity")
        names.add(name)
    if ["mypy", "1.20.2"] not in packages:
        raise ValueError("Native typing compiler metadata differs from its version")
    return identity


def _configuration_advisories(
    line: str, config: Path, configuration: str
) -> tuple[str, ...]:
    """Bind MyPy's original unused-section notice to its actual derived INI file."""
    prefix = f"{config}: note: unused section(s): "
    if not line.startswith(prefix):
        return ()
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(configuration)
    declared = {
        name
        for section in parser.sections()
        if section.startswith("mypy-")
        for name in section.removeprefix("mypy-").split(",")
    }
    modules = []
    for item in line.removeprefix(prefix).split(", "):
        matched = re.fullmatch(r"\[mypy-([A-Za-z_][A-Za-z0-9_.*]*|\*)\]", item)
        if matched is None or matched[1] not in declared:
            raise ValueError("Unbound native MyPy configuration advisory")
        modules.append(matched[1])
    if len(set(modules)) != len(modules):
        raise ValueError("Repeated native MyPy configuration advisory")
    return tuple(f"mypy-unused-override:{module}" for module in modules)


def parse_typing_result(
    result: subprocess.CompletedProcess[str],
    root: Path,
    sources: dict[str, str],
    identity: str,
    config: Path,
    configuration: str,
) -> tuple[tuple[Finding, ...], tuple[str, ...]]:
    """Qualify original error locations, notes and configuration advisories.

    Parameters
    ----------
    result : subprocess.CompletedProcess[str]
        Actual compiler process report to decode and qualify.
    root : Path
        Original repository root for native absolute diagnostic paths.
    sources : dict[str, str]
        Complete native source inventory and original byte digests.
    identity : str
        Fingerprint of the actual compiler, environment and configuration.
    config : Path
        Actual derived INI path named by native unused-section advisories.
    configuration : str
        Exact derived INI bytes supplied to the compiler.

    Returns
    -------
    tuple[tuple[Finding, ...], tuple[str, ...]]
        Validated source-bound errors and separate original advisories. Native
        information notes do not become debt; JSON mode still returns status one.

    Raises
    ------
    ValueError
        If status, stderr, a diagnostic, source binding or advisory is invalid.
    """
    if result.returncode not in (0, 1) or result.stderr:
        raise ValueError("Native typing process did not completely analyse source")
    findings_list: list[Finding] = []
    advisories: set[str] = set()
    diagnostics = 0
    lines = result.stdout.splitlines()
    for line in lines:
        information = _configuration_advisories(line, config, configuration)
        if information:
            if advisories.intersection(information):
                raise ValueError("Repeated native MyPy configuration advisory")
            advisories.update(information)
            continue
        item = object_mapping(decode_quality_json(line.encode()))
        if set(item) != {
            "file",
            "line",
            "column",
            "end_line",
            "end_column",
            "message",
            "hint",
            "code",
            "severity",
        }:
            raise ValueError("Native MyPy diagnostic lacks original required members")
        filename, code, message = item["file"], item["code"], item["message"]
        line_number, column = _coordinates(item)
        if (
            item["severity"] not in ("error", "note")
            or not isinstance(filename, str)
            or not (
                code is None
                or isinstance(code, str)
                and re.fullmatch(r"[a-z][a-z0-9-]*", code)
            )
            or not isinstance(message, str)
            or not message.strip()
            or "\x00" in message
            or not (item["hint"] is None or isinstance(item["hint"], str))
        ):
            raise ValueError("Invalid original native MyPy diagnostic")
        diagnostics += 1
        if item["severity"] == "note":
            continue
        if not isinstance(code, str) or line_number < 1:
            raise ValueError(
                "Native typing error lacks a source-bound rule or position"
            )
        path = Path(filename)
        relative = path.relative_to(root).as_posix() if path.is_absolute() else filename
        findings_list.append(
            Finding(
                relative,
                code,
                line_number,
                None if column == -1 else column + 1,
                message,
            )
        )
    # Validate once against the actual complete source inventory and profile.
    findings = parse_snapshot(
        snapshot_payload(
            Snapshot(
                "python-types",
                identity,
                sources,
                tuple(findings_list),
                True,
                tuple(sorted(advisories)),
            )
        )
    ).findings
    if bool(diagnostics) != bool(result.returncode):
        raise ValueError("Native typing status disagrees with original diagnostics")
    return findings, tuple(sorted(advisories))


def _coordinates(item: dict[str, object]) -> tuple[int, int]:
    """Require original native integers, including MyPy's unavailable-end sentinel."""
    values = []
    for name in ("line", "column", "end_line", "end_column"):
        value = item[name]
        if type(value) is not int or value < -1:
            raise ValueError("Invalid original native MyPy position")
        values.append(value)
    return values[0], values[1]
