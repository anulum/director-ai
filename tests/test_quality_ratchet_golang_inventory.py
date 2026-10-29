# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Git-visible Go source custody tests
"""Exercise actual Go filename cohorts, generated owners and symbolic path refusal."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from tools.quality_ratchet_golang import GENERATED_GO, go_sources
from tools.quality_ratchet_inventory import source_candidate


@pytest.fixture
def go_source_repository(tmp_path: Path) -> Path:
    """Create a real Git source root without native tool substitutes.

    Parameters
    ----------
    tmp_path : Path
        Native runner's isolated allocation.

    Returns
    -------
    Path
        Actual repository with one pending maintained Go source.
    """
    root = tmp_path / "go-source-inventory"
    root.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
    )
    (root / "gateway").mkdir()
    (root / "gateway/original.go").write_text("package original\n")
    return root


@pytest.mark.parametrize(
    "name", (b"unrelated-\xff.txt", b"metadata:entry.txt", b"metadata\tentry.txt")
)
def test_unrelated_native_names_leave_the_complete_go_cohort_intact(
    go_source_repository: Path,
    name: bytes,
) -> None:
    """Select actual Go bytes before decoding or validating unrelated metadata names.

    Parameters
    ----------
    go_source_repository : Path
        Actual selected source repository.
    name : bytes
        Real non-Go native filename with invalid source-path text or encoding.
    """
    (go_source_repository / os.fsdecode(name)).write_bytes(b"unrelated metadata\n")
    assert go_sources(go_source_repository) == ("gateway/original.go",)


def test_unrepresentable_go_source_is_refused_without_cohort_shrinkage(
    go_source_repository: Path,
) -> None:
    """Reject an actual maintained Go name that cannot bind to the UTF-8 contract.

    Parameters
    ----------
    go_source_repository : Path
        Actual POSIX source repository.
    """
    (go_source_repository / os.fsdecode(b"gateway/source-\xff.go")).write_bytes(
        b"package original\n"
    )
    with pytest.raises(UnicodeDecodeError):
        go_sources(go_source_repository)


def test_original_tools_tests_and_generated_owners_keep_their_exact_scope(
    go_source_repository: Path,
) -> None:
    """Include maintained tooling and native tests while excluding only generated owners.

    Parameters
    ----------
    go_source_repository : Path
        Actual repository with pending and tracked Go sources.
    """
    root = go_source_repository
    (root / "tools").mkdir()
    (root / "tools/check.go").write_text("package checks\n")
    (root / "gateway/original_test.go").write_text("package original\n")
    for relative in GENERATED_GO:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("generator-owned bytes\n")
    subprocess.run(
        ["git", "add", "gateway/original.go"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert go_sources(root / "gateway/..", deadline=time.monotonic() + 10) == (
        "gateway/original.go",
        "gateway/original_test.go",
        "tools/check.go",
    )
    (root / "gateway/original.go").unlink()
    assert go_sources(root) == ("gateway/original_test.go", "tools/check.go")


def test_symbolic_parent_cannot_turn_tracked_go_source_into_an_apparent_deletion(
    go_source_repository: Path,
) -> None:
    """Reject genuine broken symbolic ownership before classifying missing source bytes.

    Parameters
    ----------
    go_source_repository : Path
        Actual repository before a tracked source parent is replaced.
    """
    root = go_source_repository
    parent = root / "gateway/nested"
    parent.mkdir()
    original = parent / "owned.go"
    original.write_text("package original\n")
    subprocess.run(
        ["git", "add", "gateway/nested/owned.go"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
    )
    original.unlink()
    parent.rmdir()
    parent.symlink_to(root / "absent", target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic"):
        go_sources(root)


def test_source_candidate_keeps_original_missing_path_identity(
    go_source_repository: Path,
) -> None:
    """Resolve missing tracked candidates without inventing bytes or following aliases.

    Parameters
    ----------
    go_source_repository : Path
        Actual selected source repository.
    """
    assert (
        source_candidate(go_source_repository, "gateway/absent.go")
        == go_source_repository / "gateway/absent.go"
    )


@pytest.mark.parametrize("deadline", (0.0, float("nan"), float("inf")))
def test_go_inventory_respects_the_shared_native_deadline(
    go_source_repository: Path, deadline: float
) -> None:
    """Refuse expired or nonfinite shared budgets before starting Git analysis.

    Parameters
    ----------
    go_source_repository : Path
        Actual selected Git source repository.
    deadline : float
        Expired or nonfinite absolute monotonic deadline.
    """
    with pytest.raises(ValueError, match="whole deadline"):
        go_sources(go_source_repository, deadline=deadline)
