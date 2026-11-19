# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native Lean documentation snapshot adapter
"""Compile unchanged Lean sources and bind authored documentation to native evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import (
    SCHEMA,
    Snapshot,
    decode_quality_json,
    parse_snapshot,
    repository_path,
)
from tools.quality_ratchet_inventory import hash_sources, source_path
from tools.quality_ratchet_lean_declarations import (
    LEAN_COMMIT,
    LEAN_VERSION,
    NATIVE_SCHEMA,
    lean_documentation_findings,
)

LEAN_PROJECT = "formal/HaltMonitor"
LEAN_HELPER_ROOT = "tools/quality_ratchet_lean"
LEAN_HELPER = LEAN_HELPER_ROOT + "/QualityRatchet.lean"
LEAN_CONFIGURATIONS = (
    LEAN_PROJECT + "/lakefile.toml",
    LEAN_PROJECT + "/lean-toolchain",
)
LEAN_ENGINE = (
    "tools/quality_ratchet_lean.py",
    "tools/quality_ratchet_lean_declarations.py",
    LEAN_HELPER,
)


def _compiler_findings(
    original: bytes, source: bytes, path: str, absolute: Path
) -> list[dict[str, object]]:
    """Retain each actual compiler warning at its original source coordinates."""
    findings: list[dict[str, object]] = []
    lines = source.decode("utf-8").split("\n")
    for line in original.splitlines():
        row = object_mapping(decode_quality_json(line))
        if row.get("severity") not in ("warning", "information"):
            raise ValueError("Native Lean compiler output has an unqualified severity")
        if row["severity"] != "warning":
            continue
        position = object_mapping(row.get("pos"))
        number, column, message = (
            position.get("line"),
            position.get("column"),
            row.get("data"),
        )
        if (
            row.get("fileName") != str(absolute)
            or type(number) is not int
            or type(column) is not int
            or not 1 <= number <= len(lines)
            or not 0 <= column <= len(lines[number - 1])
            or not isinstance(message, str)
            or not message.strip()
            or "\x00" in message
        ):
            raise ValueError(
                "Native Lean compiler warning lacks original source identity"
            )
        finding = {
            "path": path,
            "rule": "compiler-warning",
            "line": number,
            "column": column + 1,
            "message": message,
            "cell": None,
        }
        if finding not in findings:
            findings.append(finding)
    return findings


def lean_sources(root: Path) -> tuple[str, ...]:
    """Inventory every Git-visible Lean file and refuse undeclared source roots.

    Parameters
    ----------
    root : Path
        Actual canonical Git source root.

    Returns
    -------
    tuple[str, ...]
        Complete sorted maintained inventory, including the native reader.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If the root, source recipe or complete regular inventory cannot qualify.
    """
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    if Path(top.stdout.strip()).resolve() != root:
        raise ValueError("Lean inventory requires the actual Git root")
    native = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=20,
    )
    paths: set[str] = set()
    for raw in native.stdout.split(b"\0"):
        if not raw:
            continue
        relative = repository_path(raw.decode("utf-8"))
        if Path(relative).suffix != ".lean":
            continue
        if not any(
            relative.startswith(prefix + "/")
            for prefix in (LEAN_PROJECT, LEAN_HELPER_ROOT)
        ):
            raise ValueError("Lean source lies outside the declared native recipe")
        candidate = root / relative
        if not candidate.exists() and not candidate.is_symlink():
            continue
        source_path(root, relative)
        paths.add(relative)
    if not paths:
        raise ValueError("Native Lean source inventory is empty")
    return tuple(sorted(paths))


def capture_lean_documentation(
    root: Path, lean: str, *, timeout: float = 120.0
) -> Snapshot:
    """Measure every maintained source with the pinned native Lean frontend.

    Parameters
    ----------
    root : Path
        Actual selected Git source repository.
    lean : str
        Actual native Lean executable; no installation is performed.
    timeout : float
        Positive finite whole-analysis deadline in seconds.

    Returns
    -------
    Snapshot
        Complete original source-bound documentation findings. Each dependency
        is compiled from unchanged source in a fresh isolated import environment;
        hidden caller caches and search paths are not used. This profile checks
        documentation presence and compiler warnings, not proof adequacy or
        prose correctness. Repeated diagnostics retain their complete original
        advisory sequence and one finding per actual warning identity.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If source, import graph, compiler or whole native analysis cannot qualify.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native quality deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    root = root.resolve(strict=True)
    engine_root = Path(__file__).resolve().parents[1]
    engine = hash_sources(engine_root, LEAN_ENGINE)
    helper = source_path(engine_root, LEAN_HELPER)
    executable = shutil.which(lean)
    if executable is None:
        raise ValueError("Native Lean executable is unavailable")
    binary = Path(executable).resolve(strict=True)
    binary_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    paths = lean_sources(root)
    sources = hash_sources(root, paths)
    configurations = hash_sources(root, LEAN_CONFIGURATIONS)
    if (
        source_path(root, LEAN_CONFIGURATIONS[1]).read_text().strip()
        != "leanprover/lean4:v" + LEAN_VERSION
    ):
        raise ValueError("Lean project toolchain differs from the qualified recipe")
    project = tomllib.loads(source_path(root, LEAN_CONFIGURATIONS[0]).read_text())
    if project != {
        "name": "HaltMonitor",
        "defaultTargets": ["HaltMonitor"],
        "lean_lib": [{"name": "HaltMonitor"}],
    }:
        raise ValueError("Lean project dependencies differ from the qualified recipe")

    def run(
        arguments: list[str], directory: Path, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one actual native process within the remaining whole deadline."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("Native Lean documentation exhausted its whole deadline")
        return subprocess.run(
            [str(binary), *arguments],
            cwd=directory,
            env=environment,
            capture_output=True,
            check=False,
            timeout=remaining,
        )

    clean_environment = dict(os.environ)
    clean_environment.pop("LEAN_PATH", None)
    clean_environment.pop("LEAN_SYSROOT", None)
    version = run(["--version"], root, clean_environment)
    prefix = run(["--print-prefix"], root, clean_environment)
    expected_version = f"Lean (version {LEAN_VERSION}, x86_64-unknown-linux-gnu, commit {LEAN_COMMIT}, Release)"
    if (
        version.returncode
        or version.stderr
        or version.stdout.decode().strip() != expected_version
        or prefix.returncode
        or prefix.stderr
    ):
        raise ValueError("Native Lean differs from the qualified compiler recipe")
    sysroot = Path(prefix.stdout.decode().strip()).resolve(strict=True)
    library = sysroot / "lib/lean"
    standard = {
        name: hashlib.sha256((library / (name + ".olean")).read_bytes()).hexdigest()
        for name in ("Init", "Lean")
    }
    modules: dict[str, tuple[str, Path]] = {}
    for relative in paths:
        owner = (
            LEAN_PROJECT
            if relative.startswith(LEAN_PROJECT + "/")
            else LEAN_HELPER_ROOT
        )
        module = (
            relative.removeprefix(owner + "/").removesuffix(".lean").replace("/", ".")
        )
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", module)
            is None
        ):
            raise ValueError("Lean source module has an unqualified native identity")
        if module in modules:
            raise ValueError("Native Lean module inventory is ambiguous")
        modules[module] = relative, root / owner
    findings: list[dict[str, object]] = []
    advisories: list[str] = []
    with tempfile.TemporaryDirectory(prefix="quality-lean-") as temporary:
        output = Path(temporary)
        environment = dict(
            clean_environment, LEAN_PATH=str(output), LEAN_SYSROOT=str(sysroot)
        )
        imports: dict[str, tuple[str, ...]] = {}
        for module, (relative, _) in modules.items():
            native = run(
                [
                    "--run",
                    str(helper),
                    "header",
                    str(source_path(root, relative)),
                    module,
                    str(sysroot),
                ],
                root,
                environment,
            )
            row = object_mapping(decode_quality_json(native.stdout))
            expected = {
                "schema": NATIVE_SCHEMA,
                "compiler": LEAN_VERSION,
                "compiler_commit": LEAN_COMMIT,
                "mode": "header",
                "module": module,
                "source": source_path(root, relative).read_bytes().decode("utf-8"),
                "has_errors": False,
                "commands": [],
                "constants": [],
                "messages": [],
            }
            if (
                native.returncode
                or native.stderr
                or set(row) != set(expected) | {"header", "imports"}
                or any(row[key] != value for key, value in expected.items())
                or type(row["has_errors"]) is not bool
            ):
                raise ValueError("Native Lean header or original source cannot qualify")
            names = row["imports"]
            if not isinstance(names, list) or any(
                not isinstance(name, str) or not name for name in names
            ):
                raise ValueError("Native Lean import inventory is malformed")
            dependencies = tuple(
                dict.fromkeys(name for name in names if isinstance(name, str))
            )
            for name in dependencies:
                if (
                    re.fullmatch(
                        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", name
                    )
                    is None
                ):
                    raise ValueError(
                        "Native Lean import has an unqualified module identity"
                    )
                if (
                    name not in modules
                    and not (library / (name.replace(".", "/") + ".olean")).is_file()
                ):
                    raise ValueError("Native Lean source imports an unbound module")
            imports[module] = tuple(name for name in dependencies if name in modules)
        pending = set(modules)
        while pending:
            ready = sorted(
                module for module in pending if not set(imports[module]) & pending
            )
            if not ready:
                raise ValueError("Native Lean source import graph contains a cycle")
            for module in ready:
                relative, project_root = modules[module]
                artifact = output / (module.replace(".", "/") + ".olean")
                artifact.parent.mkdir(parents=True, exist_ok=True)
                native = run(
                    [
                        "--json",
                        "-R",
                        str(project_root),
                        "-o",
                        str(artifact),
                        str(source_path(root, relative)),
                    ],
                    project_root,
                    environment,
                )
                if native.returncode or native.stderr or not artifact.is_file():
                    raise ValueError("Native Lean source compilation did not qualify")
                if native.stdout.strip():
                    findings.extend(
                        _compiler_findings(
                            native.stdout,
                            source_path(root, relative).read_bytes(),
                            relative,
                            source_path(root, relative),
                        )
                    )
                    advisories.append(
                        relative + ": " + native.stdout.decode("utf-8").strip()
                    )
                native = run(
                    [
                        "--run",
                        str(helper),
                        "inspect",
                        str(source_path(root, relative)),
                        module,
                        str(sysroot),
                    ],
                    root,
                    environment,
                )
                if native.returncode or native.stderr:
                    raise ValueError(
                        "Native Lean complete source inspection did not qualify"
                    )
                findings.extend(
                    lean_documentation_findings(
                        native.stdout,
                        source_path(root, relative).read_bytes(),
                        relative,
                        module,
                    )
                )
                row = object_mapping(decode_quality_json(native.stdout))
                for index, value in enumerate(
                    row["messages"] if isinstance(row["messages"], list) else []
                ):
                    advisories.append(
                        relative
                        + f": native-message[{index}]: "
                        + json.dumps(value, sort_keys=True)
                    )
                pending.remove(module)
    if (
        lean_sources(root) != paths
        or hash_sources(root, paths) != sources
        or hash_sources(root, LEAN_CONFIGURATIONS) != configurations
        or hash_sources(engine_root, LEAN_ENGINE) != engine
        or hashlib.sha256(binary.read_bytes()).hexdigest() != binary_digest
        or any(
            hashlib.sha256((library / (name + ".olean")).read_bytes()).hexdigest()
            != digest
            for name, digest in standard.items()
        )
    ):
        raise ValueError(
            "Lean source, compiler or effective recipe changed during analysis"
        )
    identity = hashlib.sha256(
        json.dumps(
            {
                "profile": "lean-docs",
                "schema": SCHEMA,
                "compiler": LEAN_VERSION,
                "compiler_commit": LEAN_COMMIT,
                "binary": binary_digest,
                "standard": standard,
                "configurations": configurations,
                "engine": engine,
                "recipe": "explicit-sysroot-frontend-original-import-graph-v1",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return parse_snapshot(
        {
            "schema": SCHEMA,
            "profile": "lean-docs",
            "identity": identity,
            "sources": sources,
            "findings": findings,
            "complete": True,
            "advisories": advisories,
        }
    )
