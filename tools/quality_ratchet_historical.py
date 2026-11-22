# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — reviewed identity changes made before the succession mechanism
"""Honour reviewed identity changes that predate the succession mechanism.

Before the mechanism existed, a profile identity could change only through a
reviewed decision, with no transition entry. The per-commit walk would refuse
such a commit for ever, because history cannot carry an entry now. This file
lists each one: the commit, the profile, the old and new identity, and the
sha256 of the decision record that allowed it.

Three limits keep it from becoming a bypass:

- an entry is honoured only for a commit that is a strict ancestor of the
  mechanism's landing commit, the first first-parent commit carrying
  :data:`MECHANISM_PATH`; every later identity change needs a transition;
- it maps one profile's identity only, at that commit; every finding, source
  and baseline check still runs, so no finding change can hide behind it;
- the file is append-only, and each digest is checked against the decision
  records by :func:`verify_decision_digests` wherever those records exist,
  which is the lab gate. The records are internal, so CI checks the form only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_baseline import Baseline, parse_commit_id
from tools.quality_ratchet_contracts import decode_quality_json, sha256_digest
from tools.quality_ratchet_profiles import REQUIRED_PROFILES

HISTORICAL_PATH = "tools/quality_ratchet_historical_transitions.json"
HISTORICAL_SCHEMA = "quality-ratchet.historical-transitions.v1"
MECHANISM_PATH = "tools/quality_ratchet_succession.py"
_MEMBERS = {"commit", "profile", "old_identity", "new_identity", "decision_sha256"}


@dataclass(frozen=True)
class HistoricalEntry:
    """One reviewed identity change made before the mechanism.

    Attributes
    ----------
    commit : str
        The commit that changed the identity.
    profile : str
        The one profile it changed.
    old_identity, new_identity : str
        The parent's identity and the commit's identity.
    decision_sha256 : str
        The sha256 of the decision record that allowed the change.
    """

    commit: str
    profile: str
    old_identity: str
    new_identity: str
    decision_sha256: str


def parse_historical(data: bytes) -> tuple[HistoricalEntry, ...]:
    """Validate the whole historical file.

    Parameters
    ----------
    data : bytes
        Exact file bytes.

    Returns
    -------
    tuple[HistoricalEntry, ...]
        Entries in file order.

    Raises
    ------
    ValueError
        If the envelope or any entry is malformed, or two entries name the
        same commit and profile.
    """
    envelope = object_mapping(decode_quality_json(data))
    raw = envelope.get("entries")
    if (
        set(envelope) != {"schema", "entries"}
        or envelope["schema"] != HISTORICAL_SCHEMA
        or not isinstance(raw, list)
    ):
        raise ValueError("Invalid historical identity record envelope")
    entries = []
    for value in raw:
        item = object_mapping(value)
        if set(item) != _MEMBERS or item["profile"] not in REQUIRED_PROFILES:
            raise ValueError("Invalid historical identity record entry")
        entry = HistoricalEntry(
            parse_commit_id(item["commit"]),
            str(item["profile"]),
            sha256_digest(item["old_identity"]),
            sha256_digest(item["new_identity"]),
            sha256_digest(item["decision_sha256"]),
        )
        if entry.old_identity == entry.new_identity:
            raise ValueError("A historical identity record entry changes nothing")
        entries.append(entry)
    if len({(entry.commit, entry.profile) for entry in entries}) != len(entries):
        raise ValueError("Two historical identity records name one change")
    return tuple(entries)


def require_entries_append_only(base: bytes | None, candidate: bytes | None) -> None:
    """Refuse a historical file that rewrites or drops a parent's entry.

    Parameters
    ----------
    base : bytes | None
        The parent's file, or None when it has none.
    candidate : bytes | None
        The commit's file, or None when it has none.

    Raises
    ------
    ValueError
        If the commit's entries do not begin with the parent's.
    """
    if base is None:
        return
    old = object_mapping(decode_quality_json(base)).get("entries")
    new = object_mapping(decode_quality_json(candidate or b"{}")).get("entries")
    if not isinstance(old, list) or not isinstance(new, list) or new[: len(old)] != old:
        raise ValueError("Historical identity records are append-only")


def mechanism_commit(root: Path, tip: str) -> str | None:
    """Return the first first-parent commit that carries the mechanism.

    Parameters
    ----------
    root : Path
        Repository root.
    tip : str
        A resolved commit.

    Returns
    -------
    str | None
        The mechanism's landing commit, or None if the tip's history has none.
    """
    listed = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "rev-list",
            "--first-parent",
            "--reverse",
            tip,
            "--",
            MECHANISM_PATH,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.split()
    return listed[0] if listed else None


def honoured_entries(
    root: Path, entries: Sequence[HistoricalEntry], tip: str
) -> tuple[dict[tuple[str, str], HistoricalEntry], tuple[str, ...]]:
    """Split entries into those honoured and refusals for the rest.

    Parameters
    ----------
    root : Path
        Repository root.
    entries : Sequence[HistoricalEntry]
        The tip's historical entries.
    tip : str
        The resolved tip.

    Returns
    -------
    tuple[dict[tuple[str, str], HistoricalEntry], tuple[str, ...]]
        Honoured entries keyed by commit and profile, and one refusal per
        entry whose commit is not a strict ancestor of the mechanism.
    """
    landing = mechanism_commit(root, tip)
    honoured: dict[tuple[str, str], HistoricalEntry] = {}
    refusals = []
    for entry in entries:
        before = (
            landing is not None
            and entry.commit != landing
            and (
                subprocess.run(
                    ["git", "merge-base", "--is-ancestor", entry.commit, landing],
                    cwd=root,
                    capture_output=True,
                    check=False,
                    timeout=60,
                ).returncode
                == 0
            )
        )
        if before:
            honoured[(entry.commit, entry.profile)] = entry
        else:
            refusals.append(
                f"Historical identity record for {entry.commit[:12]} {entry.profile} "
                "does not precede the succession mechanism"
            )
    return honoured, tuple(refusals)


def apply_historical(
    commit: str,
    previous: Baseline,
    candidate: Baseline,
    honoured: Mapping[tuple[str, str], HistoricalEntry],
) -> tuple[Baseline, tuple[str, ...]]:
    """Map honoured identity changes back, so every other check still runs.

    Parameters
    ----------
    commit : str
        The commit being checked.
    previous, candidate : Baseline
        Its parent's baseline and its own.
    honoured : Mapping[tuple[str, str], HistoricalEntry]
        The honoured entries.

    Returns
    -------
    tuple[Baseline, tuple[str, ...]]
        The candidate with each honoured profile's identity set back to the
        parent's, and refusals for entries that do not match this step.
    """
    profiles = dict(candidate.profiles)
    refusals = []
    for (named, profile), entry in honoured.items():
        if (
            named != commit
            or profile not in profiles
            or profile not in previous.profiles
        ):
            continue
        if (previous.profiles[profile].identity, profiles[profile].identity) != (
            entry.old_identity,
            entry.new_identity,
        ):
            refusals.append(
                f"Historical identity record for {profile} does not match this commit"
            )
            continue
        profiles[profile] = dataclasses.replace(
            profiles[profile], identity=entry.old_identity
        )
    return dataclasses.replace(candidate, profiles=profiles), tuple(refusals)


def verify_decision_digests(
    entries: Sequence[HistoricalEntry], records_root: Path
) -> tuple[str, ...]:
    """Refuse an entry whose decision digest names no record file.

    Parameters
    ----------
    entries : Sequence[HistoricalEntry]
        The historical entries.
    records_root : Path
        A directory tree holding the decision records.

    Returns
    -------
    tuple[str, ...]
        One refusal per entry whose digest matches no regular file there.
    """
    digests = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in records_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    return tuple(
        f"Historical identity record for {entry.commit[:12]} {entry.profile} "
        "names no decision record"
        for entry in entries
        if entry.decision_sha256 not in digests
    )
