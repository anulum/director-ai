# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Git-visible source inventory and byte identity tests
"""Exercise actual Git source cohorts, filename encodings and regular-file custody."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from tools.quality_ratchet_inventory import (
    GENERATED_PYTHON,
    PYTHON_ROOTS,
    hash_sources,
    python_sources,
    source_path,
)


@pytest.fixture
def source_repository(tmp_path: Path) -> Path:
    """Create an actual Git repository with one untracked maintained source.

    Parameters
    ----------
    tmp_path : Path
        Runner-owned isolated Samsung allocation.

    Returns
    -------
    Path
        Actual source repository with an original binary byte identity.
    """
    root = tmp_path / "source-inventory"
    root.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
    )
    (root / "src").mkdir()
    (root / "src/original.py").write_bytes(b'"""Original source."""\nVALUE = "\x00"\n')
    return root


def test_inventory_keeps_every_declared_root_and_original_byte_hash(
    source_repository: Path,
) -> None:
    """Retain tracked and pending source across all declared maintained roots.

    Parameters
    ----------
    source_repository : Path
        Actual selected source repository.
    """
    paths = {"src/original.py", "setup.py", "src/declaration.pyi"}
    for directory in PYTHON_ROOTS:
        relative = directory + "/owned.py"
        target = source_repository / relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b'"""Maintained native source."""\nVALUE: int = 1\n')
        paths.add(relative)
    for relative in ("setup.py", "src/declaration.pyi"):
        (source_repository / relative).write_bytes(b"VALUE: int = 1\n")
    subprocess.run(
        ["git", "add", "src/original.py"],
        cwd=source_repository,
        check=True,
        capture_output=True,
        timeout=10,
    )
    actual = python_sources(source_repository)
    assert actual == tuple(sorted(paths))
    assert hash_sources(source_repository, actual) == {
        path: hashlib.sha256((source_repository / path).read_bytes()).hexdigest()
        for path in paths
    }


@pytest.mark.parametrize(
    "name", (b"metadata-\xff.txt", b"metadata:entry.txt", b"metadata\tentry.txt")
)
def test_unrelated_metadata_names_cannot_invalidate_the_python_cohort(
    source_repository: Path, name: bytes
) -> None:
    """Ignore genuine non-Python Git filenames before applying source-path rules.

    Parameters
    ----------
    source_repository : Path
        Actual POSIX Git repository with a maintained Python source.
    name : bytes
        Native unrelated metadata name with a non-UTF-8 byte or source-invalid text.
    """
    (source_repository / os.fsdecode(name)).write_bytes(b"unrelated metadata\n")
    assert python_sources(source_repository) == ("src/original.py",)


def test_undecodable_python_name_is_refused_instead_of_omitted(
    source_repository: Path,
) -> None:
    """Keep an actual unrepresentable maintained source from shrinking the cohort.

    Parameters
    ----------
    source_repository : Path
        Actual POSIX Git repository with an additional native byte filename.
    """
    (source_repository / os.fsdecode(b"src/native-\xff.py")).write_bytes(b"VALUE = 1\n")
    with pytest.raises(UnicodeDecodeError):
        python_sources(source_repository)


def test_inventory_respects_ignored_files_and_exact_generated_owners(
    source_repository: Path,
) -> None:
    """Exclude ignored additions and only the declared generated protobuf paths.

    Parameters
    ----------
    source_repository : Path
        Actual selected repository with owned and generator-owned source.
    """
    (source_repository / ".gitignore").write_text("ignored.py\n")
    (source_repository / "src/ignored.py").write_text("VALUE = 1\n")
    for relative in GENERATED_PYTHON:
        generated = source_repository / relative
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"generator-owned source\n")
    assert python_sources(source_repository) == ("src/original.py",)


def test_pending_tracked_deletions_leave_only_actual_remaining_sources(
    source_repository: Path,
) -> None:
    """Remove missing tracked bytes without accepting an empty native source scope.

    Parameters
    ----------
    source_repository : Path
        Actual repository before and after a staged source is removed.
    """
    subprocess.run(
        ["git", "add", "src/original.py"],
        cwd=source_repository,
        check=True,
        capture_output=True,
        timeout=10,
    )
    (source_repository / "src/original.py").unlink()
    (source_repository / "setup.py").write_text("VALUE: int = 1\n")
    assert python_sources(source_repository) == ("setup.py",)
    (source_repository / "setup.py").unlink()
    with pytest.raises(ValueError, match="inventory is empty"):
        python_sources(source_repository)


def test_python_outside_declared_roots_requires_an_explicit_scope_owner(
    source_repository: Path,
) -> None:
    """Reject an undeclared Python root rather than silently dropping new source.

    Parameters
    ----------
    source_repository : Path
        Actual maintained repository with an undeclared source path.
    """
    (source_repository / "outside.py").write_text("VALUE: int = 1\n")
    with pytest.raises(ValueError, match="outside the declared"):
        python_sources(source_repository)


def test_source_lookup_refuses_symbolic_parent_and_missing_regular_file(
    source_repository: Path, tmp_path: Path
) -> None:
    """Reject nested symbolic ownership and absent bytes through the public lookup.

    Parameters
    ----------
    source_repository : Path
        Actual selected source repository.
    tmp_path : Path
        Isolated allocation for the actual unrelated symbolic target.
    """
    external = tmp_path / "external-source"
    external.mkdir()
    (external / "outside.py").write_text("VALUE: int = 1\n")
    (source_repository / "src/alias").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic"):
        source_path(source_repository, "src/alias/outside.py")
    with pytest.raises(ValueError, match="not a regular"):
        hash_sources(source_repository, ("src/absent.py",))


def test_inventory_refuses_nonrepository_and_nested_root(
    source_repository: Path, tmp_path: Path
) -> None:
    """Require an actual whole Git repository before reporting complete source bytes.

    Parameters
    ----------
    source_repository : Path
        Actual repository whose nested source directory is not its root.
    tmp_path : Path
        Runner-owned location for an actual nonrepository directory.
    """
    with pytest.raises(ValueError, match="actual Git root"):
        python_sources(source_repository / "src")
    outside = tmp_path / "not-a-repository"
    outside.mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        python_sources(outside)


def test_tracked_source_below_broken_symbolic_parent_is_not_a_deletion(
    source_repository: Path,
) -> None:
    """Refuse reparented tracked source before classifying absent bytes as deletion.

    Parameters
    ----------
    source_repository : Path
        Actual Git repository whose tracked nested directory becomes symbolic.
    """
    parent = source_repository / "src/nested"
    parent.mkdir()
    target = parent / "original.py"
    target.write_text("VALUE: int = 1\n")
    subprocess.run(
        ["git", "add", "src/nested/original.py"],
        cwd=source_repository,
        check=True,
        capture_output=True,
        timeout=10,
    )
    target.unlink()
    parent.rmdir()
    parent.symlink_to(source_repository / "absent-target", target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic"):
        python_sources(source_repository)
