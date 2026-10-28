# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — immutable Git authority for historical quality findings
"""Read original quality allowances without working-tree or Git replacement aliases."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_comparison import compare_snapshots
from tools.quality_ratchet_contracts import (
    Snapshot,
    decode_quality_json,
    parse_snapshot,
    repository_path,
    snapshot_payload,
)

BASELINE_PATH = "tools/quality_ratchet_baseline.json"
BASELINE_SCHEMA = "quality-ratchet.baseline.v1"


@dataclass(frozen=True)
class Baseline:
    """Retain the immutable adoption commit and complete native profiles.

    Attributes
    ----------
    origin_commit : str
        Original full immutable Git commit identifier.
    profiles : dict[str, Snapshot]
        Complete native snapshots keyed by their exact profile names.
    """

    origin_commit: str
    profiles: dict[str, Snapshot]


def parse_commit_id(value: object) -> str:
    """Validate a full immutable SHA-1 or SHA-256 Git commit identifier.

    Parameters
    ----------
    value : object
        Decoded baseline member or stripped native Git revision output.

    Returns
    -------
    str
        Original lowercase hexadecimal object identifier.

    Raises
    ------
    ValueError
        If the value is not a full identifier in either Git object format.
    """
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", value) is None
    ):
        raise ValueError("Invalid immutable quality origin commit")
    return value


def resolve_commit(root: Path, reference: str) -> str:
    """Resolve a Git reference once, rejecting options and noncommit objects.

    Parameters
    ----------
    root : Path
        Selected Git repository root.
    reference : str
        Explicit caller-selected base reference, before immutable resolution.

    Returns
    -------
    str
        Full immutable Git commit identifier.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If the reference cannot resolve to an actual commit.
    """
    result = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "rev-parse",
            "--verify",
            "--end-of-options",
            reference + "^{commit}",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return parse_commit_id(result.stdout.strip())


def parse_baseline(data: bytes) -> Baseline:
    """Validate original baseline JSON without duplicate or unknown members.

    Parameters
    ----------
    data : bytes
        Exact original baseline JSON bytes.

    Returns
    -------
    Baseline
        Validated complete individual allowances, not trusted by parsing alone.

    Raises
    ------
    ValueError
        If an envelope, origin, profile or complete snapshot is invalid.
    """
    value = object_mapping(decode_quality_json(data))
    if set(value) != {"schema", "origin_commit", "profiles"}:
        raise ValueError("Invalid original quality baseline members")
    if value["schema"] != BASELINE_SCHEMA:
        raise ValueError("Invalid original quality baseline schema")
    origin = parse_commit_id(value["origin_commit"])
    profiles = {}
    for name, raw in object_mapping(value["profiles"]).items():
        snapshot = parse_snapshot(raw)
        if name != snapshot.profile or not snapshot.complete:
            raise ValueError("Invalid or incomplete native baseline profile")
        profiles[name] = snapshot
    if not profiles:
        raise ValueError("Original quality baseline has no native profiles")
    return Baseline(origin, profiles)


def baseline_payload(baseline: Baseline) -> dict[str, object]:
    """Encode every individual source-bound allowance with its adoption origin.

    Parameters
    ----------
    baseline : Baseline
        Validated original baseline.

    Returns
    -------
    dict[str, object]
        Complete JSON-compatible baseline with deterministic profile ordering.
    """
    return {
        "schema": BASELINE_SCHEMA,
        "origin_commit": baseline.origin_commit,
        "profiles": {
            name: snapshot_payload(snapshot)
            for name, snapshot in sorted(baseline.profiles.items())
        },
    }


def committed_files(
    root: Path, commit: str, paths: tuple[str, ...]
) -> dict[str, bytes]:
    """Read exact regular file bytes from a resolved immutable commit.

    Parameters
    ----------
    root : Path
        Selected repository root.
    commit : str
        Previously resolved full immutable commit identifier.
    paths : tuple[str, ...]
        Exact canonical relative paths; missing paths stay absent.

    Returns
    -------
    dict[str, bytes]
        Original Git blob bytes, excluding symbolic and nonregular entries.
        Commit, tree and blob replacement refs do not affect the lookup.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If object identity, native framing or a requested file mode is invalid.
    """
    if resolve_commit(root, commit) != commit:
        raise ValueError("Quality source lookup requires an immutable commit")
    requested = {repository_path(path).encode("utf-8") for path in paths}
    result = subprocess.run(
        ["git", "--no-replace-objects", "ls-tree", "-rz", "--full-tree", commit],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=20,
    )
    entries = {}
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        if raw_path not in requested:
            continue
        path = raw_path.decode("utf-8")
        mode, kind, oid = metadata.split(b" ")
        if mode not in (b"100644", b"100755") or kind != b"blob":
            raise ValueError("Original quality source is not a regular Git blob")
        entries[path] = oid
    if not entries:
        return {}
    ordered = sorted(entries)
    result = subprocess.run(
        ["git", "--no-replace-objects", "cat-file", "--batch"],
        cwd=root,
        input=b"".join(entries[path] + b"\n" for path in ordered),
        capture_output=True,
        check=True,
        timeout=20,
    )
    return parse_git_blob_batch(result.stdout, entries)


def parse_git_blob_batch(data: bytes, objects: dict[str, bytes]) -> dict[str, bytes]:
    """Decode exact Git cat-file batch frames for the requested original blobs.

    Parameters
    ----------
    data : bytes
        Complete native batch output, including each header and blob separator.
    objects : dict[str, bytes]
        Selected paths mapped to original object identifiers, requested in
        sorted path order. Paths and identifiers come from the original Git tree.

    Returns
    -------
    dict[str, bytes]
        Exact binary blob bytes indexed by their selected paths.

    Raises
    ------
    ValueError
        If a frame has the wrong identity, kind, size or separator, is truncated,
        or leaves unconsumed bytes after the last requested object.
    """
    cursor, files = 0, {}
    for path, expected in sorted(objects.items()):
        end = data.index(b"\n", cursor)
        oid, kind, raw_size = data[cursor:end].split(b" ")
        size = int(raw_size)
        if oid != expected or kind != b"blob" or size < 0:
            raise ValueError("Invalid original quality Git object response")
        start, cursor = end + 1, end + size + 2
        if cursor > len(data) or data[cursor - 1 : cursor] != b"\n":
            raise ValueError("Incomplete original quality Git blob response")
        files[path] = data[start : cursor - 1]
    if cursor != len(data):
        raise ValueError("Unexpected trailing original quality Git bytes")
    return files


def read_committed_baseline(root: Path, base: str) -> Baseline | None:
    """Read the original base baseline, distinguishing absence from Git refusal.

    Parameters
    ----------
    root : Path
        Selected repository root.
    base : str
        Resolved full immutable base commit identifier.

    Returns
    -------
    Baseline | None
        Original parsed baseline, or None only for an actually absent Git path.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If Git or the original baseline is invalid or unavailable.
    """
    files = committed_files(root, base, (BASELINE_PATH,))
    return parse_baseline(files[BASELINE_PATH]) if BASELINE_PATH in files else None


def compare_baselines(original: Baseline, candidate: Baseline) -> tuple[str, ...]:
    """Refuse origin/profile changes and additions to individual debt allowances.

    Parameters
    ----------
    original : Baseline
        Original baseline read from the immutable trusted base commit.
    candidate : Baseline
        Working-tree candidate baseline, which cannot supply its own authority.

    Returns
    -------
    tuple[str, ...]
        Exact shrink-only refusals; removed findings cannot fund added findings.
    """
    if original.origin_commit != candidate.origin_commit:
        return ("Original quality adoption commit changed",)
    if set(original.profiles) != set(candidate.profiles):
        return ("Original native quality profile set changed",)
    return tuple(
        refusal
        for name in sorted(original.profiles)
        for refusal in compare_snapshots(
            original.profiles[name], candidate.profiles[name]
        )
    )


def verify_adoption_sources(root: Path, base: str, candidate: Baseline) -> None:
    """Permit initial legacy findings only on byte-identical immutable sources.

    Parameters
    ----------
    root : Path
        Selected repository root.
    base : str
        Resolved exact original base commit for this initial adoption.
    candidate : Baseline
        Initial native baseline, separately compared with the actual current scan.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If the exact origin or any source behind an initial allowance cannot qualify.
    """
    origin = resolve_commit(root, candidate.origin_commit)
    if origin != base:
        raise ValueError(
            "Initial quality adoption requires the exact original base commit"
        )
    paths = tuple(
        sorted({f.path for s in candidate.profiles.values() for f in s.findings})
    )
    files = committed_files(root, origin, paths)
    for snapshot in candidate.profiles.values():
        for finding in snapshot.findings:
            data = files.get(finding.path)
            if (
                data is None
                or hashlib.sha256(data).hexdigest() != snapshot.sources[finding.path]
            ):
                raise ValueError(
                    "Initial quality allowance has new or modified source bytes"
                )
