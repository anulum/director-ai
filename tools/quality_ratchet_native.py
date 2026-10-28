# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual native documentation snapshot adapter
"""Run native Ruff against complete source bytes without documentation ignores."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import (
    SCHEMA,
    Snapshot,
    decode_quality_json,
    parse_snapshot,
)
from tools.quality_ratchet_inventory import (
    GENERATED_PYTHON,
    PYTHON_ROOTS,
    hash_sources,
    python_sources,
    source_path,
)

RUFF_VERSION = "ruff 0.16.10"
DOCUMENTATION_FLAGS = (
    "check",
    "--isolated",
    "--no-cache",
    "--ignore-noqa",
    "--select",
    "D",
    "--target-version",
    "py311",
    "--config",
    'lint.pydocstyle.convention="numpy"',
    "--output-format",
    "json",
)


def documentation_identity(root: Path, native_version: str = RUFF_VERSION) -> str:
    """Bind the native tool, effective flags, scopes and generator configuration.

    Parameters
    ----------
    root : Path
        Canonical selected source repository.
    native_version : str
        Actual native version output, or the declared profile for offline identity.

    Returns
    -------
    str
        Stable SHA-256 independent of installation and checkout absolute paths.

    Raises
    ------
    ValueError, OSError
        If the tool version differs or an identity-bearing configuration is absent.
    """
    if native_version != RUFF_VERSION:
        raise ValueError("Native Ruff version differs from the qualified profile")
    payload = {
        "profile": "python-docs",
        "snapshot_schema": SCHEMA,
        "native_version": RUFF_VERSION,
        "flags": DOCUMENTATION_FLAGS,
        "roots": PYTHON_ROOTS,
        "generated": GENERATED_PYTHON,
        "configurations": {
            relative: hashlib.sha256(
                source_path(root, relative).read_bytes()
            ).hexdigest()
            for relative in ("pyproject.toml", "requirements/ci-lint.txt")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _ruff_finding(value: object, root: Path) -> dict[str, object]:
    """Validate one actual Ruff diagnostic and retain its notebook cell identity."""
    item = object_mapping(value)
    if not {"filename", "code", "location", "message"}.issubset(item):
        raise ValueError("Native Ruff diagnostic lacks original required members")
    filename, code = item["filename"], item["code"]
    if not isinstance(filename, str) or not isinstance(code, str):
        raise ValueError("Invalid native Ruff diagnostic")
    if re.fullmatch(r"D[0-9]{3}", code) is None:
        raise ValueError(
            "Native documentation analysis did not completely parse source"
        )
    path = Path(filename)
    relative = path.relative_to(root).as_posix() if path.is_absolute() else filename
    location = object_mapping(item["location"])
    if set(location) != {"row", "column"}:
        raise ValueError("Invalid original native Ruff location")
    return {
        "path": relative,
        "rule": code,
        "line": location["row"],
        "column": location["column"],
        "message": item["message"],
        "cell": item.get("cell"),
    }


def capture_python_documentation(
    root: Path,
    python: str,
    *,
    timeout: float = 120.0,
) -> Snapshot:
    """Run the pinned native tool and verify unchanged source/configuration bytes.

    Parameters
    ----------
    root : Path
        Actual source repository root.
    python : str
        Interpreter with the pinned Ruff installed; no package installation occurs.
    timeout : float
        Positive finite per-process deadline in seconds.

    Returns
    -------
    Snapshot
        Complete original diagnostic identities bound to exact source bytes.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If tool, syntax, source, scope or configuration cannot qualify.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native quality deadline must be positive and finite")
    root = root.resolve(strict=True)
    version = subprocess.run(
        [python, "-m", "ruff", "--version"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout,
    )
    paths = python_sources(root)
    sources = hash_sources(root, paths)
    identity = documentation_identity(root, version.stdout.strip())
    result = subprocess.run(
        [python, "-m", "ruff", *DOCUMENTATION_FLAGS, "--", *paths],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if (
        python_sources(root) != paths
        or hash_sources(root, paths) != sources
        or documentation_identity(root) != identity
    ):
        raise ValueError(
            "Quality source or configuration changed during native analysis"
        )
    return parse_documentation_result(root, result, sources, identity)


def parse_documentation_result(
    root: Path,
    result: subprocess.CompletedProcess[str],
    sources: dict[str, str],
    identity: str,
) -> Snapshot:
    """Qualify original native Ruff diagnostics against their complete source scope.

    Parameters
    ----------
    root : Path
        Original repository root for native absolute diagnostic paths.
    result : subprocess.CompletedProcess[str]
        Actual native status and output report; malformed controls may be refused
        directly without replacing a compiler process.
    sources : dict[str, str]
        Complete original source paths and exact byte digests.
    identity : str
        Qualified native profile and configuration fingerprint.

    Returns
    -------
    Snapshot
        Complete source-bound native documentation findings.

    Raises
    ------
    ValueError
        If process status, stderr, output, source binding or diagnostics disagree.
    """
    if result.returncode not in (0, 1) or result.stderr:
        raise ValueError("Native documentation process did not complete")
    raw = decode_quality_json(result.stdout.encode("utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            "Native documentation output is not an original diagnostic list"
        )
    if bool(raw) != bool(result.returncode):
        raise ValueError("Native documentation status disagrees with diagnostics")
    return parse_snapshot(
        {
            "schema": SCHEMA,
            "profile": "python-docs",
            "identity": identity,
            "sources": sources,
            "findings": [_ruff_finding(item, root) for item in raw],
            "complete": True,
            "advisories": [],
        }
    )
