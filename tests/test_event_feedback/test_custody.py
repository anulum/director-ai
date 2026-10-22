# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback protocol tests
"""Verify original custody using actual filesystem permissions and collisions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from director_ai.agentic.event_feedback.codec import FeedbackInvalidError
from director_ai.agentic.event_feedback.custody import (
    EvidenceRoot,
    durable_directory,
    private_bytes,
    retain,
)

from .conftest import registered_root


def test_original_retention_is_immutable_and_private(tmp_path: Path) -> None:
    """Publish once, reconcile identical bytes and refuse changed originals."""
    path = tmp_path / "private" / "original"
    retain(path, b"original")
    assert private_bytes(path) == b"original"
    assert path.stat().st_mode & 0o077 == 0
    retain(path, b"original")
    with pytest.raises(FeedbackInvalidError):
        retain(path, b"changed")
    assert private_bytes(path) == b"original"
    assert list(path.parent.glob(".review-*")) == []


def test_nonprivate_directory_and_file_are_refused(tmp_path: Path) -> None:
    """Reject public custody before retaining credentials or evidence."""
    parent = tmp_path / "public"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    with pytest.raises(FeedbackInvalidError):
        retain(parent / "original", b"private")
    file = tmp_path / "file"
    file.write_bytes(b"secret")
    file.chmod(0o644)
    with pytest.raises(FeedbackInvalidError):
        private_bytes(file)


def test_symlink_fifo_directory_and_missing_file_refuse(tmp_path: Path) -> None:
    """Never follow a final link or block on a nonregular credential."""
    original = tmp_path / "original"
    original.write_bytes(b"key")
    original.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(original)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, mode=0o600)
    for file in (link, fifo, tmp_path, tmp_path / "missing"):
        with pytest.raises(FeedbackInvalidError):
            private_bytes(file)
    with pytest.raises(FeedbackInvalidError):
        retain(link, b"key")


def test_one_mib_is_exact_admission_boundary(tmp_path: Path) -> None:
    """Admit the full boundary and refuse truncating an oversized original."""
    file = tmp_path / "boundary"
    retain(file, b"x" * 1048576)
    assert len(private_bytes(file)) == 1048576
    oversize = tmp_path / "oversize"
    oversize.write_bytes(b"x" * 1048577)
    oversize.chmod(0o600)
    with pytest.raises(FeedbackInvalidError):
        private_bytes(oversize)


def test_original_destination_under_regular_file_refuses(tmp_path: Path) -> None:
    """A filesystem publication failure returns a safe custody error."""
    parent = tmp_path / "not-a-directory"
    parent.write_bytes(b"original")
    with pytest.raises(FeedbackInvalidError):
        retain(parent / "child", b"private")
    assert parent.read_bytes() == b"original"


def test_illegal_original_filename_refuses_and_cleans_temporary(tmp_path: Path) -> None:
    """A real publication error refuses without leaking a temporary original."""
    parent = tmp_path / "private"
    with pytest.raises(FeedbackInvalidError):
        retain(parent / ("x" * 300), b"original")
    assert list(parent.iterdir()) == []


def test_symbolic_directory_ancestry_is_refused(tmp_path: Path) -> None:
    """An operator directory symlink cannot redirect verified custody publication."""
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(FeedbackInvalidError):
        retain(link / "child" / "original", b"private")
    assert not (real / "child" / "original").exists()


@pytest.mark.parametrize(
    "replacement", ["directory", "symlink", "removed", "permissions"]
)
def test_live_ancestry_changes_refuse_bound_publication(
    tmp_path: Path, replacement: str
) -> None:
    """A held leaf never silently switches to a replacement pathname."""
    root = registered_root(tmp_path / "registered")
    leaf = root.path / "review"
    with durable_directory(leaf, root) as custody:
        custody.retain("original", b"original")
        if replacement == "permissions":
            leaf.chmod(0o755)
        else:
            leaf.rename(root.path / "original-leaf")
            if replacement == "directory":
                leaf.mkdir(mode=0o700)
            elif replacement == "symlink":
                leaf.symlink_to(root.path / "original-leaf", target_is_directory=True)
        with pytest.raises(FeedbackInvalidError):
            custody.reserve("invocation-reserved", b"one")
    retained = leaf if replacement == "permissions" else root.path / "original-leaf"
    assert (retained / "original").read_bytes() == b"original"
    assert not (retained / "invocation-reserved").exists()


def test_leaf_replacement_is_refused_after_cold_restart(tmp_path: Path) -> None:
    """Root-level inode custody prevents a replaced leaf from clearing a reservation."""
    root = registered_root(tmp_path / "registered")
    leaf = root.path / "review"
    with durable_directory(leaf, root) as custody:
        custody.reserve("invocation-reserved", b"one")
    leaf.rename(root.path / "reserved-original")
    leaf.mkdir(mode=0o700)
    with pytest.raises(FeedbackInvalidError), durable_directory(leaf, root):
        pass
    assert (root.path / "reserved-original/invocation-reserved").read_bytes() == b"one"
    assert not (leaf / "invocation-reserved").exists()


def test_registered_root_replacement_is_refused_after_restart(tmp_path: Path) -> None:
    """The operator pin cannot silently adopt a new directory at the same path."""
    root = registered_root(tmp_path / "registered")
    root.path.rename(tmp_path / "original-root")
    root.path.mkdir(mode=0o700)
    with (
        pytest.raises(FeedbackInvalidError),
        durable_directory(root.path / "review", root),
    ):
        pass
    assert not (root.path / "review").exists()


@pytest.mark.parametrize("operation", ["exists", "retain", "reserve", "child"])
@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "bad\x00name"])
def test_pinned_operations_refuse_nonlocal_names(
    tmp_path: Path, operation: str, name: str
) -> None:
    """No public pinned operation can escape its verified directory handle."""
    with (
        durable_directory(tmp_path / "private") as custody,
        pytest.raises(FeedbackInvalidError),
    ):
        if operation == "exists":
            custody.exists(name)
        elif operation == "retain":
            custody.retain(name, b"private")
        elif operation == "reserve":
            custody.reserve(name, b"private")
        else:
            with custody.child(name):
                pass
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "path,device,inode",
    [
        ("relative", 1, 1),
        ("/", 1, 1),
        ("/absolute", -1, 1),
        ("/absolute", True, 1),
        ("/absolute", 1, 0),
        ("/absolute", 1, True),
    ],
)
def test_invalid_registered_root_identity_refuses(
    path: str, device: int, inode: int
) -> None:
    """Root identity must be absolute and structurally bound before filesystem access."""
    with pytest.raises(FeedbackInvalidError):
        EvidenceRoot(Path(path), device, inode)


def test_dot_parent_and_unregistered_descendants_refuse(tmp_path: Path) -> None:
    """A registration grants only its root and immediate review directory."""
    root = registered_root(tmp_path / "registered")
    for path in (
        tmp_path / "../outside",
        root.path / "review/nested",
        tmp_path / "other",
    ):
        with pytest.raises(FeedbackInvalidError), durable_directory(path, root):
            pass
    assert list(root.path.iterdir()) == []


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_transport_child_cannot_follow_a_non_directory(
    tmp_path: Path, kind: str
) -> None:
    """A pinned parent refuses an existing file or symbolic child before publication."""
    with durable_directory(tmp_path / "private") as custody:
        child = custody.path / "attempts"
        if kind == "file":
            child.write_bytes(b"original")
        else:
            child.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(FeedbackInvalidError), custody.child("attempts"):
            pass
    assert not (tmp_path / "request.json").exists()


def test_registered_parent_permission_change_refuses_held_child(tmp_path: Path) -> None:
    """A held review directory cannot continue after its registered root becomes public."""
    root = registered_root(tmp_path / "registered")
    with durable_directory(root.path / "review", root) as custody:
        root.path.chmod(0o755)
        with pytest.raises(FeedbackInvalidError):
            custody.reserve("invocation-reserved", b"one")
    assert not (root.path / "review/invocation-reserved").exists()
