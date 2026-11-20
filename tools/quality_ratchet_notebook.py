# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native notebook typing profile
"""Capture original strict MyPy diagnostics at their notebook cell positions."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

from tools import (
    quality_ratchet_notebook_source,
    quality_ratchet_typing,
    quality_ratchet_typing_protocol,
)
from tools.quality_ratchet_contracts import (
    Finding,
    Snapshot,
    parse_snapshot,
    snapshot_payload,
)
from tools.quality_ratchet_inventory import hash_sources, python_sources, source_path
from tools.quality_ratchet_notebook_source import NotebookSource, project_notebook
from tools.quality_ratchet_typing import (
    MYPY_COMMAND,
    TYPING_CONFIGURATIONS,
    TYPING_FLAGS,
    _compiler_environment,
    _configuration,
    _environment,
    _run,
)
from tools.quality_ratchet_typing_protocol import parse_typing_result


def _digest(data: bytes) -> str:
    """Hash a native input for snapshot provenance."""
    return hashlib.sha256(data).hexdigest()


def _rebind(
    fields: tuple[Finding, ...],
    projections: dict[str, tuple[str, NotebookSource]],
    information: set[str],
) -> tuple[Finding, ...]:
    """Retain native imports and map generated coordinates to notebook inputs."""
    findings = []
    for finding in fields:
        if finding.path not in projections:
            findings.append(finding)
            continue
        part, projected = projections[finding.path]
        if finding.line > len(projected.coordinates):
            raise ValueError("Native notebook position has no original cell line")
        cell, line = projected.coordinates[finding.line - 1]
        if finding.rule == "top-level-await" and cell in projected.asynchronous_cells:
            information.add(f"notebook-native-top-level-await:{part}:{cell}:{line}")
            continue
        findings.append(
            Finding(part, finding.rule, line, finding.column, finding.message, cell)
        )
    return tuple(findings)


def capture_notebook_typing(
    root: Path, python: str, *, timeout: float = 120.0
) -> Snapshot:
    """Type-check the maintained Python notebook source with actual MyPy.

    Parameters
    ----------
    root : Path
        Git repository containing all authored notebook and imported sources.
    python : str
        Interpreter with the pinned MyPy compiler and typing dependencies.
    timeout : float
        Positive finite deadline for the complete native check in seconds.

    Returns
    -------
    Snapshot
        Source-bound native diagnostics with one-based notebook cell positions.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If native analysis, syntax, source custody or tool provenance is unqualified.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native notebook deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    root = root.resolve(strict=True)
    paths = python_sources(root)
    notebooks = tuple(path for path in paths if path.endswith(".ipynb"))
    if not notebooks:
        raise ValueError("Native notebook source inventory is empty")
    hashes = hash_sources(root, paths)
    config_hashes = hash_sources(root, TYPING_CONFIGURATIONS)
    configuration = _configuration(root)
    environment = _compiler_environment(root)
    installed = _environment(root, python, deadline, environment)
    engine_files = tuple(
        Path(path)
        for path in (
            __file__,
            quality_ratchet_notebook_source.__file__,
            quality_ratchet_typing.__file__,
            quality_ratchet_typing_protocol.__file__,
        )
    )
    engine = tuple(_digest(path.read_bytes()) for path in engine_files)
    identity = _digest(
        json.dumps(
            {
                "profile": "notebook-types",
                "environment": installed,
                "projection_runtime": tuple(sys.version_info[:3]),
                "configuration": configuration,
                "configuration_hashes": config_hashes,
                "flags": TYPING_FLAGS,
                "compiler_command": MYPY_COMMAND,
                "engine": engine,
            },
            sort_keys=True,
        ).encode()
    )
    information: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="quality-notebooks-") as directory:
        temporary = Path(directory).resolve(strict=True)
        config = temporary / "mypy.ini"
        config.write_text(configuration)
        projections: dict[str, tuple[str, NotebookSource]] = {}
        inputs: dict[str, str] = {}
        generated_hashes: dict[str, str] = {}
        aliases: dict[str, str] = {}
        for path in notebooks:
            original = source_path(root, path).read_bytes()
            if _digest(original) != hashes[path]:
                raise ValueError("Notebook source changed before native analysis")
            projection = project_notebook(original, path)
            name = temporary / f"notebook_{_digest(path.encode())}.py"
            encoded = projection.text.encode()
            name.write_bytes(encoded)
            file_id = f"notebook_quality_inputs/{name.name}"
            digest = _digest(encoded)
            inputs[str(name)] = digest
            aliases[str(name)] = file_id
            generated_hashes[file_id] = digest
            projections[file_id] = (path, projection)
            information.add(f"notebook-projection-sha256:{path}:{digest}")
        result = _run(
            [
                python,
                *MYPY_COMMAND,
                *TYPING_FLAGS,
                "--config-file",
                str(config),
                "--cache-dir",
                str(temporary / "cache"),
                "--",
                *inputs,
            ],
            root,
            deadline,
            environment,
        )
        fields, advisories = parse_typing_result(
            result,
            root,
            hashes | generated_hashes,
            identity,
            config,
            configuration,
            source_aliases=aliases,
        )
        information.update(advisories)
        findings = _rebind(fields, projections, information)
        if config.read_text() != configuration or any(
            _digest(Path(path).read_bytes()) != digest
            for path, digest in inputs.items()
        ):
            raise ValueError(
                "Generated notebook compiler inputs changed during analysis"
            )
    if (
        python_sources(root) != paths
        or hash_sources(root, paths) != hashes
        or hash_sources(root, TYPING_CONFIGURATIONS) != config_hashes
        or tuple(_digest(path.read_bytes()) for path in engine_files) != engine
        or _environment(root, python, deadline, environment) != installed
    ):
        raise ValueError(
            "Notebook source, generator, configuration or environment changed during analysis"
        )
    scope = set(notebooks) | {finding.path for finding in findings}
    sources = {path: hashes[path] for path in scope}
    snapshot = Snapshot(
        "notebook-types", identity, sources, findings, True, tuple(sorted(information))
    )
    return parse_snapshot(snapshot_payload(snapshot))
