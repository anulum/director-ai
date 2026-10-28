# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete Git-visible quality source inventory
"""Bind native Python checks to every maintained Git-visible source byte."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from tools.quality_ratchet_contracts import repository_path

PYTHON_ROOTS = (
    "src",
    "tests",
    "packages",
    "tools",
    "scripts",
    "benchmarks",
    "examples",
    "notebooks",
    "training",
    "demo",
    "discord-bot",
    "gpu_deploy",
    "director_guard",
    "docs",
)
GENERATED_PYTHON = (
    "src/director_ai/director_pb2.py",
    "src/director_ai/director_pb2_grpc.py",
    "src/director_ai/proto/director/v1/director_pb2.py",
    "src/director_ai/proto/director/v1/director_pb2_grpc.py",
    "src/director_ai/proto/director/v1/director_pb2.pyi",
)
PYTHON_SUFFIXES = frozenset({".py", ".pyi", ".ipynb"})


def source_path(root: Path, relative: str) -> Path:
    """Resolve a regular source without following any nested symbolic link.

    Parameters
    ----------
    root : Path
        Resolved canonical repository root.
    relative : str
        Canonical repository-relative POSIX path.

    Returns
    -------
    Path
        Existing regular source within the selected root.

    Raises
    ------
    ValueError, OSError
        If the path is invalid, symbolic, absent or not a regular file.
    """
    path = _source_candidate(root, relative)
    if not path.is_file():
        raise ValueError("Quality source is not a regular file")
    return path


def _source_candidate(root: Path, relative: str) -> Path:
    """Check every original path component before any missing-source classification."""
    relative = repository_path(relative)
    path = root
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symbolic quality source is not qualified")
    return path


def python_sources(root: Path) -> tuple[str, ...]:
    """Inventory all Git-visible Python sources, including pending additions.

    Parameters
    ----------
    root : Path
        Actual Git repository root, not a parent or nested subdirectory.

    Returns
    -------
    tuple[str, ...]
        Sorted complete maintained source paths. Original generated protobuf
        files are the only Python generator exclusions. Notebooks participate
        in the native Ruff documentation profile.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If Git, scope, source identity or a nonempty inventory cannot qualify.
    """
    root = root.resolve(strict=True)
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    if Path(top.stdout.strip()).resolve() != root:
        raise ValueError("Quality inventory requires the actual Git root")
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=20,
    )
    paths = set()
    for raw in result.stdout.split(b"\x00"):
        if not raw:
            continue
        if not any(raw.endswith(suffix.encode("ascii")) for suffix in PYTHON_SUFFIXES):
            continue
        relative = repository_path(raw.decode("utf-8"))
        if relative in GENERATED_PYTHON:
            continue
        if relative != "setup.py" and Path(relative).parts[0] not in PYTHON_ROOTS:
            raise ValueError("Python source lies outside the declared quality scope")
        candidate = _source_candidate(root, relative)
        if not candidate.exists():
            continue  # Pending tracked deletion; no source byte remains.
        source_path(root, relative)
        paths.add(relative)
    if not paths:
        raise ValueError("Native quality source inventory is empty")
    return tuple(sorted(paths))


def hash_sources(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    """Hash exact regular source bytes for a previously inventoried scope.

    Parameters
    ----------
    root : Path
        Canonical selected source root.
    paths : tuple[str, ...]
        Canonical original inventory paths.

    Returns
    -------
    dict[str, str]
        Original paths bound to SHA-256 source byte identities.

    Raises
    ------
    ValueError, OSError
        If any source is invalid, symbolic, missing or unreadable.
    """
    return {
        relative: hashlib.sha256(source_path(root, relative).read_bytes()).hexdigest()
        for relative in paths
    }
