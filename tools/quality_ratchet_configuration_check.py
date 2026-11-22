# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the configuration change checks on commits and on the tip
"""Apply the configuration change rules to a commit or to the checked tip.

The per-commit walk checks every ``configuration_change`` entry a commit adds,
statically, on the parent's and the commit's Git objects: the TOML diff of
each named file, the commit's changed paths and its baseline delta. The tip
check recomputes only the TOML diff, between the trusted base and the working
tree, because the tip diff spans the whole pushed range, whose other commits
may change anything the ordinary ratchet allows. Nothing is taken from the
record; the tip's identity recompute binds the configuration bytes.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.quality_ratchet_baseline import committed_files
from tools.quality_ratchet_configuration import (
    PYPROJECT,
    baseline_delta_refusals,
    excluded_packages,
    parse_configuration,
    raw_only_refusals,
    shape_refusals,
)
from tools.quality_ratchet_contracts import Snapshot, repository_path
from tools.quality_ratchet_inventory import source_candidate
from tools.quality_ratchet_succession import Transition


def changed_paths(root: Path, parent: str, commit: str) -> dict[str, str]:
    """Return each path a commit changes against its parent, with its status.

    Parameters
    ----------
    root : Path
        Repository root.
    parent, commit : str
        The resolved first parent and commit.

    Returns
    -------
    dict[str, str]
        Repository paths mapped to A, M, D or T; renames count as an added
        and a deleted path.
    """
    diff = subprocess.run(
        ["git", "--no-replace-objects", "diff", "--name-status", "--no-renames", "-z"]
        + [parent, commit],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout.split(b"\0")
    return {
        repository_path(path.decode("utf-8")): status.decode("ascii")[0]
        for status, path in zip(diff[0::2], diff[1::2], strict=False)
        if status
    }


def _documents(
    files: Mapping[str, tuple[bytes | None, bytes | None]],
) -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    """Parse each named configuration file's bytes before and after."""
    return {
        name: (parse_configuration(before, name), parse_configuration(after, name))
        for name, (before, after) in files.items()
    }


def _raw_only(
    entries: Sequence[Transition],
    documents: Mapping[str, tuple[dict[str, object], dict[str, object]]],
) -> set[str]:
    """Return every entry's profile-qualified raw-only refusals."""
    return {
        f"{entry.profile}: {refusal}"
        for entry in entries
        for name, recorded in entry.configuration_paths.items()
        for refusal in raw_only_refusals(
            entry.profile, name, *documents[name], recorded
        )
    }


def configuration_refusals(
    entries: Sequence[Transition],
    files: Mapping[str, tuple[bytes | None, bytes | None]],
    changed: Mapping[str, str],
    previous: Mapping[str, Snapshot],
    candidate: Mapping[str, Snapshot],
) -> tuple[str, ...]:
    """Check one commit's configuration entries: diff, shape and baseline delta.

    Parameters
    ----------
    entries : Sequence[Transition]
        The ``configuration_change`` entries the step applies.
    files : Mapping[str, tuple[bytes | None, bytes | None]]
        Each named configuration file's bytes before and after.
    changed : Mapping[str, str]
        The step's changed paths and their Git status letters.
    previous, candidate : Mapping[str, Snapshot]
        The baselines' snapshots by profile, before and after.

    Returns
    -------
    tuple[str, ...]
        Sorted refusals; empty means every entry is raw-only and the commit
        has the declared shape and baseline delta.
    """
    documents = _documents(files)
    refusals = _raw_only(entries, documents)
    excluded = (
        excluded_packages(documents[PYPROJECT][1])
        if PYPROJECT in documents
        else frozenset()
    )
    refusals.update(shape_refusals(changed, frozenset(files), excluded))
    refusals.update(
        baseline_delta_refusals(
            previous,
            candidate,
            frozenset(entry.profile for entry in entries),
        )
    )
    return tuple(sorted(refusals))


def _named(entries: Sequence[Transition]) -> tuple[str, ...]:
    """Return the configuration files the entries name, sorted."""
    return tuple(sorted({name for e in entries for name in e.configuration_paths}))


def commit_configuration_refusals(
    root: Path,
    parent: str,
    commit: str,
    entries: Sequence[Transition],
    previous: Mapping[str, Snapshot],
    candidate: Mapping[str, Snapshot],
) -> tuple[str, ...]:
    """Check the configuration entries one commit adds, against its parent.

    Parameters
    ----------
    root : Path
        Repository root.
    parent, commit : str
        The resolved first parent and commit.
    entries : Sequence[Transition]
        The ``configuration_change`` entries the commit adds.
    previous, candidate : Mapping[str, Snapshot]
        The parent's and the commit's baseline snapshots by profile.

    Returns
    -------
    tuple[str, ...]
        Sorted refusals.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If Git refuses or a named file is not valid TOML.
    """
    names = _named(entries)
    before = committed_files(root, parent, names)
    after = committed_files(root, commit, names)
    files = {name: (before.get(name), after.get(name)) for name in names}
    return configuration_refusals(
        entries, files, changed_paths(root, parent, commit), previous, candidate
    )


def tip_configuration_refusals(
    root: Path, base: str, entries: Sequence[Transition]
) -> tuple[str, ...]:
    """Recompute the configuration diff of the entries the working tree applies.

    Parameters
    ----------
    root : Path
        Repository root of the checked working tree.
    base : str
        The resolved trusted base commit.
    entries : Sequence[Transition]
        The applying ``configuration_change`` entries.

    Returns
    -------
    tuple[str, ...]
        Sorted refusals; empty means each named file changed from the base to
        the working tree in recorded non-influencing pointers only.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If Git refuses, a file is unreadable or a named file is not valid TOML.
    """
    names = _named(entries)
    before = committed_files(root, base, names)
    files = {}
    for name in names:
        path = source_candidate(root, name)
        files[name] = (before.get(name), path.read_bytes() if path.is_file() else None)
    return tuple(sorted(_raw_only(entries, _documents(files))))
