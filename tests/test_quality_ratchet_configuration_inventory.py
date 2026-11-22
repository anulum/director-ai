# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the configuration commit shape against every profile inventory
"""Show that no path a configuration commit may change is any profile's source."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from tools.quality_ratchet_baseline import BASELINE_PATH
from tools.quality_ratchet_configuration import (
    PACKAGING_METADATA,
    PLACEHOLDER,
    PYPROJECT,
)
from tools.quality_ratchet_golang import go_sources
from tools.quality_ratchet_inventory import python_sources
from tools.quality_ratchet_julia_native import _julia_sources
from tools.quality_ratchet_lean import lean_sources
from tools.quality_ratchet_notebook import NOTEBOOK_SUFFIXES
from tools.quality_ratchet_profiles import REQUIRED_PROFILES
from tools.quality_ratchet_protobuf import protobuf_sources
from tools.quality_ratchet_rust import rust_sources
from tools.quality_ratchet_succession import TRANSITIONS_PATH
from tools.quality_ratchet_typescript import typescript_sources

ROOT = Path(__file__).resolve().parents[1]


def notebook_sources(root: Path) -> tuple[str, ...]:
    """Return the notebooks the notebook-types profile selects.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    tuple[str, ...]
        The Python inventory's notebooks.
    """
    return tuple(p for p in python_sources(root) if p.endswith(NOTEBOOK_SUFFIXES))


INVENTORIES: dict[str, Callable[[Path], tuple[str, ...]]] = {
    "python-docs": python_sources,
    "python-types": python_sources,
    "notebook-types": notebook_sources,
    "go-docs": go_sources,
    "typescript-docs": typescript_sources,
    "protobuf-docs": protobuf_sources,
    "lean-docs": lean_sources,
    "rust-docs": rust_sources,
    "julia-docs": _julia_sources,
}
"""Each registered profile's source inventory, as its adapter calls it."""

ALLOWED: dict[str, bytes] = {
    PYPROJECT: b"# appended by the configuration commit\n",
    TRANSITIONS_PATH: b"\n",
    BASELINE_PATH: b"\n",
    "MANIFEST.in": b"prune src/director_ai/training_adapters\n",
    "packages/paid_tier_manifest.json": b"\n",
    "packages/director-ai-lite/pyproject.toml": b"# appended\n",
    "packages/director-ai-new/pyproject.toml": b'[project]\nname = "new"\n',
    "src/director_ai/training_adapters/README.md": b"Excluded from the wheel.\n",
    "src/director_ai/experimental/README.md": b"Excluded from the wheel.\n",
}
"""Every kind of path the shape allows, appended to or created."""


@pytest.fixture(scope="module")
def mirror(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy every Git-visible file of this checkout into a fresh repository.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Session temporary directories.

    Returns
    -------
    Path
        The resolved root of the copy, every file tracked.
    """
    root = tmp_path_factory.mktemp("inventory-mirror")
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    for raw in listed.split(b"\0"):
        source = ROOT / os.fsdecode(raw)
        target = root / os.fsdecode(raw)
        if not raw or not (source.is_file() or source.is_symlink()):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copyfile(source, target)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    subprocess.run(["git", "add", "-A", "-f"], cwd=root, check=True, timeout=300)
    return root.resolve()


def test_every_registered_profile_has_its_inventory_here() -> None:
    """Fail when a profile is registered without extending this proof."""
    assert set(INVENTORIES) == set(REQUIRED_PROFILES)


def test_the_allowed_paths_cover_every_reviewed_kind() -> None:
    """Exercise each packaging pattern and the placeholder form."""
    for pattern in PACKAGING_METADATA:
        assert any(
            fnmatch.fnmatchcase(path, pattern) and path.count("/") == pattern.count("/")
            for path in ALLOWED
        )
    assert sum(PLACEHOLDER.fullmatch(path) is not None for path in ALLOWED) == 2


def test_no_allowed_path_enters_any_profile_inventory(mirror: Path) -> None:
    """Run all nine inventories before and after writing every allowed path."""
    before = {profile: inventory(mirror) for profile, inventory in INVENTORIES.items()}
    for name, data in ALLOWED.items():
        path = mirror / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(data)
    after = {profile: inventory(mirror) for profile, inventory in INVENTORIES.items()}
    assert after == before
    for profile, sources in after.items():
        assert sources, profile
        assert not set(ALLOWED) & set(sources), profile
