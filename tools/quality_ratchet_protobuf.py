# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — source-bound native Protobuf documentation profile
"""Measure actual protoc declarations and attached documentation without generated masks."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from tools.quality_ratchet_contracts import (
    SCHEMA,
    Snapshot,
    parse_snapshot,
    repository_path,
)
from tools.quality_ratchet_inventory import hash_sources, source_path

PROTOC_VERSION = "libprotoc 3.21.12"
PROTOBUF_VERSION = "6.33.6"
PROTOBUF_ROOTS = ("proto", "schemas")
PROTOBUF_ENGINE = (
    "tools/quality_ratchet_protobuf.py",
    "tools/quality_ratchet_protobuf_descriptors.py",
    "requirements/quality-protobuf.in",
    "requirements/quality-protobuf.txt",
)
DESCRIPTOR_OPTIONS = (
    "--proto_path=.",
    "--include_source_info",
    "--include_imports",
    "--descriptor_set_out=/dev/stdout",
)


def protobuf_sources(root: Path) -> tuple[str, ...]:
    """Inventory all maintained Git-visible schemas without generated exclusions.

    Parameters
    ----------
    root : Path
        Actual canonical Git source root.

    Returns
    -------
    tuple[str, ...]
        Complete sorted regular-source inventory, including pending additions.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If Git, scope, regular files or a nonempty inventory cannot qualify.
    """
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    if Path(top.stdout.strip()).resolve() != root:
        raise ValueError("Protobuf inventory requires the actual Git root")
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=20,
    )
    paths = set()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = repository_path(raw.decode())
        if Path(relative).suffix != ".proto":
            continue
        if Path(relative).parts[0] not in PROTOBUF_ROOTS:
            raise ValueError("Protobuf source lies outside the declared quality scope")
        candidate = root / relative
        if not candidate.exists() and not candidate.is_symlink():
            continue
        source_path(root, relative)
        paths.add(relative)
    if not paths:
        raise ValueError("Native Protobuf source inventory is empty")
    return tuple(sorted(paths))


def capture_protobuf_documentation(
    root: Path,
    protoc: str,
    *,
    timeout: float = 120.0,
) -> Snapshot:
    """Measure every schema with pinned native protoc and source-info descriptors.

    Parameters
    ----------
    root : Path
        Actual selected Git repository.
    protoc : str
        Pinned native executable; no installation is performed.
    timeout : float
        Positive finite whole-analysis deadline in seconds.

    Returns
    -------
    Snapshot
        Complete source-bound documentation snapshot. External unbound imports,
        syntax warnings and absent original source positions refuse qualification.
        This profile measures documentation presence, not prose correctness.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If dependencies, source, native analysis or original bytes cannot qualify.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native quality deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    root = root.resolve(strict=True)
    engine_root = Path(__file__).resolve().parents[1]
    engine = hash_sources(engine_root, PROTOBUF_ENGINE)
    try:
        runtime_version = version("protobuf")
    except PackageNotFoundError as error:
        raise ValueError("Native Protobuf descriptor reader is unavailable") from error
    if runtime_version != PROTOBUF_VERSION:
        raise ValueError("Native Protobuf descriptor reader differs from its pin")

    def run(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
        """Run the real compiler within the remaining whole analysis deadline."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(
                "Native Protobuf documentation exhausted its whole deadline"
            )
        return subprocess.run(
            [protoc, *arguments],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=remaining,
        )

    native_version = run(["--version"])
    if (
        native_version.returncode
        or native_version.stderr
        or native_version.stdout.decode().strip() != PROTOC_VERSION
    ):
        raise ValueError("Native protoc differs from the qualified recipe")
    from tools.quality_ratchet_protobuf_descriptors import descriptor_findings

    paths = protobuf_sources(root)
    sources = hash_sources(root, paths)
    native = run([*DESCRIPTOR_OPTIONS, *paths])
    if native.returncode or native.stderr:
        raise ValueError("Native Protobuf compiler did not completely analyse source")
    findings = descriptor_findings(native.stdout, paths)
    if (
        protobuf_sources(root) != paths
        or hash_sources(root, paths) != sources
        or hash_sources(engine_root, PROTOBUF_ENGINE) != engine
    ):
        raise ValueError("Protobuf source or engine changed during native analysis")
    identity = hashlib.sha256(
        json.dumps(
            {
                "profile": "protobuf-docs",
                "schema": SCHEMA,
                "protoc": PROTOC_VERSION,
                "protobuf": runtime_version,
                "options": DESCRIPTOR_OPTIONS,
                "roots": PROTOBUF_ROOTS,
                "engine": engine,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return parse_snapshot(
        {
            "schema": SCHEMA,
            "profile": "protobuf-docs",
            "identity": identity,
            "sources": sources,
            "findings": findings,
            "complete": True,
            "advisories": [],
        }
    )
