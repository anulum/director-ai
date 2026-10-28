# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — original Git baseline and adoption authority tests
"""Exercise original Git objects and refuse malformed historical allowances."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.quality_ratchet_baseline import (
    BASELINE_PATH,
    Baseline,
    baseline_payload,
    committed_files,
    compare_baselines,
    parse_baseline,
    parse_git_blob_batch,
    read_committed_baseline,
    resolve_commit,
    verify_adoption_sources,
)
from tools.quality_ratchet_contracts import Finding, Snapshot, snapshot_payload


def git(root: Path, *arguments: str) -> str:
    """Run actual Git with isolated test authorship and bounded execution.

    Parameters
    ----------
    root : Path
        Isolated actual Git repository.
    arguments : str
        Native Git command and its original arguments.

    Returns
    -------
    str
        Native standard output, preserving failures as process exceptions.
    """
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Quality Test",
            "-c",
            "user.email=quality@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    ).stdout


@pytest.fixture(params=("sha1", "sha256"))
def git_repository(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Create actual immutable binary, executable and symbolic Git entries.

    Parameters
    ----------
    tmp_path : Path
        Test runner's isolated Samsung temporary directory.
    request : pytest.FixtureRequest
        Native Git object format for the complete original-object contract.

    Returns
    -------
    Path
        Committed repository with real distinct entry modes and byte framing.
    """
    root = tmp_path / "original-objects"
    root.mkdir()
    git(root, "init", "--initial-branch=main", "--object-format=" + str(request.param))
    (root / "src").mkdir()
    (root / "src/original.py").write_bytes(b"original\x00bytes\nwith\ttab\n")
    executable = root / "src/executable.py"
    executable.write_bytes(b"executable bytes\n")
    executable.chmod(0o755)
    (root / "src/symbolic.py").symlink_to("original.py")
    git(root, "add", ".")
    git(root, "commit", "-m", "Record original object identities")
    return root


def original_baseline(root: Path) -> Baseline:
    """Construct a source-bound carrier for public original-object API checks.

    Parameters
    ----------
    root : Path
        Actual committed source repository.

    Returns
    -------
    Baseline
        Explicit schema/provenance test carrier, without native scan authority.
    """
    relative = "src/original.py"
    return Baseline(
        resolve_commit(root, "HEAD"),
        {
            "python-docs": Snapshot(
                "python-docs",
                "1" * 64,
                {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()},
                (Finding(relative, "D100", 1, 1, "Missing documentation"),),
                True,
            ),
        },
    )


def test_committed_files_preserve_original_binary_bytes_and_missing_paths(
    git_repository: Path,
) -> None:
    """Read original binary and executable objects after working-tree replacement.

    Parameters
    ----------
    git_repository : Path
        Real repository with distinct immutable object modes.
    """
    origin = resolve_commit(git_repository, "HEAD")
    expected = {
        path: (git_repository / path).read_bytes()
        for path in ("src/original.py", "src/executable.py")
    }
    (git_repository / "src/original.py").write_bytes(b"candidate replacement\n")
    (git_repository / "src/executable.py").unlink()
    actual = committed_files(
        git_repository,
        origin,
        ("src/executable.py", "src/absent.py", "src/original.py"),
    )
    assert actual == expected
    assert committed_files(git_repository, origin, ("src/absent.py",)) == {}


def test_selected_sources_ignore_unrelated_non_utf8_git_names(
    git_repository: Path,
) -> None:
    """Read selected Unicode paths beside an unrelated native Git byte filename.

    Parameters
    ----------
    git_repository : Path
        Real POSIX repository using the selected Git object format.
    """
    unrelated = git_repository / os.fsdecode(b"unrelated-\xff.py")
    unrelated.write_bytes(b"unrelated Git path bytes\n")
    selected = "src/\u017elt\u00fd.py"
    expected = b"selected original bytes\x00\n"
    (git_repository / selected).write_bytes(expected)
    git(git_repository, "add", ".")
    git(git_repository, "commit", "-m", "Record distinct filename encodings")
    origin = resolve_commit(git_repository, "HEAD")
    assert committed_files(git_repository, origin, (selected,)) == {selected: expected}
    assert committed_files(git_repository, origin, ("src/absent.py",)) == {}


@pytest.mark.parametrize("replacement_kind", ("commit", "tree", "blob"))
def test_committed_files_ignore_replacement_refs_for_every_object_layer(
    git_repository: Path, replacement_kind: str
) -> None:
    """Read original objects despite real commit, tree or blob replacement refs.

    Parameters
    ----------
    git_repository : Path
        Actual source repository in the selected native object format.
    replacement_kind : str
        Git object layer whose replacement would change the original source bytes.
    """
    original = original_baseline(git_repository)
    origin = original.origin_commit
    relative = "src/original.py"
    expected = (git_repository / relative).read_bytes()
    (git_repository / relative).write_bytes(b"replacement committed bytes\n")
    git(git_repository, "add", relative)
    git(git_repository, "commit", "-m", "Record alternate source objects")
    replacement = resolve_commit(git_repository, "HEAD")
    suffix = {"commit": "", "tree": "^{tree}", "blob": ":" + relative}[replacement_kind]
    original_object = git(git_repository, "rev-parse", origin + suffix).strip()
    replacement_object = git(git_repository, "rev-parse", replacement + suffix).strip()
    git(git_repository, "replace", original_object, replacement_object)
    assert git(git_repository, "replace", "-l").strip() == original_object
    assert git(git_repository, "show", origin + ":" + relative).encode() != expected
    assert committed_files(git_repository, origin, (relative,)) == {relative: expected}
    verify_adoption_sources(git_repository, origin, original)


def test_adoption_rejects_a_different_original_base_commit(
    git_repository: Path,
) -> None:
    """Refuse adoption authority from an earlier commit with unchanged source.

    Parameters
    ----------
    git_repository : Path
        Actual original repository before a distinct metadata-only commit.
    """
    original = original_baseline(git_repository)
    git(git_repository, "commit", "--allow-empty", "-m", "Advance selected base")
    current = resolve_commit(git_repository, "HEAD")
    assert current != original.origin_commit
    with pytest.raises(ValueError, match="exact original base commit"):
        verify_adoption_sources(git_repository, current, original)


@pytest.mark.parametrize("kind", ("symbolic", "gitlink"))
def test_committed_files_reject_nonregular_original_objects(
    git_repository: Path, kind: str
) -> None:
    """Refuse actual symbolic-link and submodule commit objects as source bytes.

    Parameters
    ----------
    git_repository : Path
        Real repository containing original regular and symbolic entries.
    kind : str
        Actual nonregular Git tree entry to exercise.
    """
    relative = "src/symbolic.py"
    if kind == "gitlink":
        relative = "src/submodule"
        origin = resolve_commit(git_repository, "HEAD")
        git(
            git_repository,
            "update-index",
            "--add",
            "--cacheinfo",
            "160000",
            origin,
            relative,
        )
        git(git_repository, "commit", "-m", "Record actual gitlink mode")
    with pytest.raises(ValueError, match="not a regular Git blob"):
        committed_files(
            git_repository, resolve_commit(git_repository, "HEAD"), (relative,)
        )


def test_original_lookup_requires_an_immutable_commit_and_refuses_blob_references(
    git_repository: Path,
) -> None:
    """Reject mutable source lookups, noncommit objects and option-like references.

    Parameters
    ----------
    git_repository : Path
        Actual committed source repository.
    """
    with pytest.raises(ValueError, match="immutable commit"):
        committed_files(git_repository, "HEAD", ("src/original.py",))
    for reference in ("HEAD:src/original.py", "--help", "absent-branch"):
        with pytest.raises(subprocess.CalledProcessError):
            resolve_commit(git_repository, reference)


def test_committed_baseline_uses_original_bytes_and_refuses_committed_corruption(
    git_repository: Path,
) -> None:
    """Distinguish genuine absence from edited candidates and invalid original JSON.

    Parameters
    ----------
    git_repository : Path
        Actual repository before and after baseline commitment.
    """
    baseline = original_baseline(git_repository)
    assert read_committed_baseline(git_repository, baseline.origin_commit) is None
    path = git_repository / BASELINE_PATH
    path.parent.mkdir()
    path.write_text(json.dumps(baseline_payload(baseline)))
    git(git_repository, "add", BASELINE_PATH)
    git(git_repository, "commit", "-m", "Record original allowance carrier")
    base = resolve_commit(git_repository, "HEAD")
    path.write_bytes(b'{"schema":"forged"}')
    assert read_committed_baseline(git_repository, base) == baseline
    git(git_repository, "add", BASELINE_PATH)
    git(git_repository, "commit", "-m", "Record invalid original carrier")
    with pytest.raises(ValueError, match="members"):
        read_committed_baseline(git_repository, resolve_commit(git_repository, "HEAD"))


@pytest.mark.parametrize(
    "damage",
    (
        "members",
        "schema",
        "origin-type",
        "origin-shape",
        "profile-name",
        "incomplete",
        "empty",
    ),
)
def test_baseline_schema_refuses_ambiguous_origin_and_profile_authority(
    git_repository: Path, damage: str
) -> None:
    """Reject distinct unqualified adoption identities and native profile envelopes.

    Parameters
    ----------
    git_repository : Path
        Actual original bytes used to bind the explicit parser input.
    damage : str
        Malformed schema or authority condition, without native success claims.
    """
    baseline = original_baseline(git_repository)
    payload = baseline_payload(baseline)
    if damage == "members":
        payload["unexpected"] = True
    elif damage == "schema":
        payload["schema"] = "quality-ratchet.baseline.v2"
    elif damage == "origin-type":
        payload["origin_commit"] = True
    elif damage == "origin-shape":
        payload["origin_commit"] = "HEAD"
    elif damage == "profile-name":
        payload["profiles"] = {
            "python-types": snapshot_payload(baseline.profiles["python-docs"])
        }
    elif damage == "incomplete":
        original = baseline.profiles["python-docs"]
        payload["profiles"] = baseline_payload(
            Baseline(
                baseline.origin_commit,
                {
                    "python-docs": Snapshot(
                        original.profile,
                        original.identity,
                        dict(original.sources),
                        original.findings,
                        False,
                    )
                },
            )
        )["profiles"]
    else:
        payload["profiles"] = {}
    with pytest.raises(ValueError):
        parse_baseline(json.dumps(payload).encode())


def test_baseline_shrink_comparison_keeps_original_origin_and_profile_set(
    git_repository: Path,
) -> None:
    """Reject replaced authority and accept only removal of individual allowances.

    Parameters
    ----------
    git_repository : Path
        Actual original source used by the explicit comparison carrier.
    """
    original = original_baseline(git_repository)
    snapshot = original.profiles["python-docs"]
    assert compare_baselines(original, Baseline("2" * 40, original.profiles))
    assert compare_baselines(original, Baseline(original.origin_commit, {}))
    removed = Snapshot(
        snapshot.profile, snapshot.identity, dict(snapshot.sources), (), True
    )
    assert (
        compare_baselines(
            original, Baseline(original.origin_commit, {snapshot.profile: removed})
        )
        == ()
    )
    assert parse_baseline(json.dumps(baseline_payload(original)).encode()) == original


def test_adoption_rejects_a_finding_without_original_committed_source_bytes(
    git_repository: Path,
) -> None:
    """Refuse absent original source and retain exact byte-bound original evidence.

    Parameters
    ----------
    git_repository : Path
        Actual repository containing only the original committed source.
    """
    original = original_baseline(git_repository)
    verify_adoption_sources(git_repository, original.origin_commit, original)
    absent = "src/absent.py"
    candidate = Baseline(
        original.origin_commit,
        {
            "python-docs": Snapshot(
                "python-docs",
                "1" * 64,
                {absent: "2" * 64},
                (Finding(absent, "D100", 1, 1, "Missing documentation"),),
                True,
            )
        },
    )
    with pytest.raises(ValueError, match="new or modified source bytes"):
        verify_adoption_sources(git_repository, original.origin_commit, candidate)


def capture_git_blob_batch(
    root: Path, paths: tuple[str, ...]
) -> tuple[bytes, dict[str, bytes]]:
    """Capture actual batch output and original object identities from Git.

    Parameters
    ----------
    root : Path
        Real committed binary and executable source repository.
    paths : tuple[str, ...]
        Original regular paths to request in sorted batch order.

    Returns
    -------
    tuple[bytes, dict[str, bytes]]
        Unmodified native output and its exact original requested identifiers.
    """
    objects = {
        path: git(root, "--no-replace-objects", "rev-parse", "HEAD:" + path)
        .strip()
        .encode("ascii")
        for path in sorted(paths)
    }
    result = subprocess.run(
        ["git", "--no-replace-objects", "cat-file", "--batch"],
        cwd=root,
        input=b"".join(oid + b"\n" for oid in objects.values()),
        capture_output=True,
        check=True,
        timeout=20,
    )
    return result.stdout, objects


@pytest.mark.parametrize("paths", ((), ("src/original.py", "src/executable.py")))
def test_native_blob_batch_parser_round_trips_captured_git_frames(
    git_repository: Path, paths: tuple[str, ...]
) -> None:
    """Decode actual empty and binary multi-object batches without interception.

    Parameters
    ----------
    git_repository : Path
        Actual source repository in the selected native object format.
    paths : tuple[str, ...]
        Empty or deliberately unsorted binary and executable requests.
    """
    data, objects = capture_git_blob_batch(git_repository, paths)
    expected = {path: (git_repository / path).read_bytes() for path in paths}
    assert parse_git_blob_batch(data, objects) == expected
    assert (
        committed_files(git_repository, resolve_commit(git_repository, "HEAD"), paths)
        == expected
    )


@pytest.mark.parametrize(
    "fault",
    (
        "object-id",
        "kind",
        "negative-size",
        "invalid-size",
        "short",
        "separator",
        "trailing",
        "header-members",
        "missing-header-newline",
    ),
)
def test_native_blob_batch_parser_refuses_malformed_captured_frames(
    git_repository: Path, fault: str
) -> None:
    """Refuse explicit damaged controls derived from actual native binary frames.

    Parameters
    ----------
    git_repository : Path
        Real source repository in the selected Git object format.
    fault : str
        Identity, header, length, separator or trailing-byte violation.
    """
    paths = ("src/original.py", "src/executable.py")
    data, objects = capture_git_blob_batch(git_repository, paths)
    assert parse_git_blob_batch(data, objects) == {
        path: (git_repository / path).read_bytes() for path in paths
    }
    header, rest = data.split(b"\n", 1)
    oid, kind, raw_size = header.split(b" ")
    separator = len(header) + 1 + int(raw_size)
    damaged = {
        "object-id": b"0" * len(oid) + b" " + kind + b" " + raw_size + b"\n" + rest,
        "kind": oid + b" tree " + raw_size + b"\n" + rest,
        "negative-size": oid + b" blob -1\n" + rest,
        "invalid-size": oid + b" blob invalid\n" + rest,
        "short": data[: separator - 1],
        "separator": data[:separator] + b"x" + data[separator + 1 :],
        "trailing": data + b"unexpected",
        "header-members": header + b" extra\n" + rest,
        "missing-header-newline": header,
    }[fault]
    with pytest.raises(ValueError):
        parse_git_blob_batch(damaged, objects)
