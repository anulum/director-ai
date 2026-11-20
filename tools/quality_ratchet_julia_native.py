# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Julia quality profile
"""Capture source-bound Julia documentation through the native compiler."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import (
    Snapshot,
    parse_snapshot,
    repository_path,
    snapshot_payload,
)
from tools.quality_ratchet_inventory import hash_sources, source_path
from tools.quality_ratchet_julia_report import JULIA_REPORT_VERSION, parse_julia_report

JULIA_CONFIGURATIONS = (
    "tools/julia_tuner/Project.toml",
    "tools/julia_tuner/Manifest.toml",
)
ANALYZER_CONFIGURATIONS = ("Project.toml", "Manifest.toml")
JULIA_VERSION = "1.13.4"
JULIA_GENERATORS = ("ArgParse", "Turing", "DynamicPPL")
ENGINE_PARTS = ("Declarations.jl", "GeneratedDeclarations.jl", "inspect.jl")
ENGINE_ROOT = Path(__file__).resolve().with_name("quality_ratchet_julia")
ENGINE_PATHS = frozenset(
    f"tools/quality_ratchet_julia/{path}"
    for path in (*ENGINE_PARTS, "test/generated.jl", "test/runtests.jl")
)
JULIA_FLAGS = ("--startup-file=no", "--history-file=no", "--threads=1")


def _julia_sources(root: Path) -> tuple[str, ...]:
    """Identify all Git-visible Julia sources and refuse unqualified packages."""
    prefix = ["git", "-C", str(root)]
    top = subprocess.check_output([*prefix, "rev-parse", "--show-toplevel"], timeout=20)
    if Path(os.fsdecode(top.rstrip(b"\n"))).resolve() != root:
        raise ValueError("Native Julia inventory needs the actual Git root")
    raw = subprocess.check_output(
        [
            *prefix,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.jl",
        ],
        timeout=20,
    )
    sources = sorted(
        {repository_path(part.decode("utf-8")) for part in raw.split(b"\0") if part}
        - ENGINE_PATHS
    )
    if not sources or any(
        not path.startswith("tools/julia_tuner/") for path in sources
    ):
        raise ValueError("Native Julia source is outside the qualified package scope")
    return tuple(sources)


def _generator_identities(
    manifest: bytes, names: tuple[str, ...]
) -> dict[str, tuple[str, str, str]]:
    """Read pinned macro package version, UUID and registry tree identities."""
    lock = tomllib.loads(manifest.decode("utf-8"))
    dependencies = object_mapping(lock.get("deps"))
    identities: dict[str, tuple[str, str, str]] = {}
    for name in names:
        entries = dependencies.get(name)
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError(f"Expected one pinned Julia generator {name}")
        entry = object_mapping(entries[0])
        version, uuid, tree = (
            entry.get(key) for key in ("version", "uuid", "git-tree-sha1")
        )
        if not isinstance(version, str) or not version:
            raise ValueError(f"Unversioned native Julia generator {name}")
        if not isinstance(uuid, str) or not isinstance(tree, str) or not tree:
            raise ValueError(f"Unqualified native Julia generator source {name}")
        identities[name] = (version, uuid, tree)
    return identities


def capture_julia_documentation(
    root: Path, julia: str, *, timeout: float = 300.0
) -> Snapshot:
    """Capture all Julia authored and generated documentation findings.

    Parameters
    ----------
    root : Path
        Selected Git repository with the tracked Julia project and manifest.
    julia : str
        Native Julia executable for the qualified compiler and generator profile.
    timeout : float
        Positive finite execution budget in seconds.

    Returns
    -------
    Snapshot
        Complete native Julia documentation snapshot with original source digests.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If a tool, project, scope, diagnostic, process or capture is unqualified.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native Julia deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    root = root.resolve(strict=True)
    executable = shutil.which(julia)
    if executable is None:
        raise ValueError("Native Julia executable is unavailable")
    binary = Path(executable).resolve(strict=True)
    binary_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    paths = _julia_sources(root)
    captured = {path: source_path(root, path).read_bytes() for path in paths}
    source_hashes = {
        path: hashlib.sha256(raw).hexdigest() for path, raw in captured.items()
    }
    tuner_configs = hash_sources(root, JULIA_CONFIGURATIONS)
    tool_configs = hash_sources(ENGINE_ROOT, ANALYZER_CONFIGURATIONS)
    tuner_project = source_path(root, JULIA_CONFIGURATIONS[0]).read_bytes()
    tuner_lock = source_path(root, JULIA_CONFIGURATIONS[1]).read_bytes()
    analyzer_lock = source_path(ENGINE_ROOT, "Manifest.toml").read_bytes()
    if (
        hashlib.sha256(tuner_project).hexdigest()
        != tuner_configs[JULIA_CONFIGURATIONS[0]]
        or hashlib.sha256(tuner_lock).hexdigest()
        != tuner_configs[JULIA_CONFIGURATIONS[1]]
        or hashlib.sha256(analyzer_lock).hexdigest() != tool_configs["Manifest.toml"]
    ):
        raise ValueError("Julia project or manifest changed before native analysis")
    deps = object_mapping(tomllib.loads(tuner_project.decode("utf-8")).get("deps"))
    names = JULIA_GENERATORS if "Turing" in deps else ("ArgParse",)
    analyzer_generators = _generator_identities(analyzer_lock, JULIA_GENERATORS)
    for name, source_identity in _generator_identities(tuner_lock, names).items():
        if analyzer_generators[name] != source_identity:
            raise ValueError(
                f"Native Julia generator {name} differs from the source lock"
            )
        if name in deps and deps[name] != source_identity[1]:
            raise ValueError(
                f"Native Julia generator {name} differs from the source project"
            )
    versions = (
        JULIA_VERSION,
        *(identity[0] for identity in analyzer_generators.values()),
    )
    engine = hash_sources(ENGINE_ROOT, ENGINE_PARTS)
    environment = dict(os.environ)
    environment["JULIA_LOAD_PATH"] = os.pathsep.join(("@", "@stdlib"))
    environment["JULIA_PKG_OFFLINE"] = "true"
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("Native Julia setup exhausted its execution budget")
    result = subprocess.run(
        [
            str(binary),
            *JULIA_FLAGS,
            f"--project={ENGINE_ROOT}",
            str(ENGINE_ROOT / "inspect.jl"),
            str(root),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
        timeout=remaining,
    )
    if result.returncode:
        diagnostic = result.stderr.decode("utf-8", errors="replace").partition("\n")[0]
        raise ValueError(
            f"Native Julia process failed ({result.returncode}): {diagnostic[:220]}"
        )
    if result.stderr:
        raise ValueError("Native Julia emitted unqualified standard error output")
    findings = parse_julia_report(
        result.stdout,
        versions=versions,
        binary=binary_digest,
        project=tool_configs["Project.toml"],
        manifest=tool_configs["Manifest.toml"],
        sources=source_hashes,
        line_counts={path: len(raw.splitlines()) for path, raw in captured.items()},
    )
    if (
        _julia_sources(root) != paths
        or hash_sources(root, paths) != source_hashes
        or hash_sources(root, JULIA_CONFIGURATIONS) != tuner_configs
        or hash_sources(ENGINE_ROOT, ANALYZER_CONFIGURATIONS) != tool_configs
        or hash_sources(ENGINE_ROOT, ENGINE_PARTS) != engine
        or hashlib.sha256(binary.read_bytes()).hexdigest() != binary_digest
    ):
        raise ValueError(
            "Julia source, engine or native configuration changed during analysis"
        )
    if time.monotonic() > deadline:
        raise ValueError("Native Julia analysis exhausted its execution budget")
    identity = hashlib.sha256(
        json.dumps(
            {
                "profile": "julia-docs",
                "schema": JULIA_REPORT_VERSION,
                "versions": versions,
                "binary": binary_digest,
                "engine": engine,
                "configuration": tool_configs,
                "flags": JULIA_FLAGS,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return parse_snapshot(
        snapshot_payload(
            Snapshot("julia-docs", identity, source_hashes, findings, True)
        )
    )
