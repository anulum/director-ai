# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — trusted quality policy and baseline gate
"""Enforce the complete native profile policy against immutable Git authority."""

from __future__ import annotations

import hashlib
import subprocess
import tomllib
from pathlib import Path

from tools.quality_ratchet_baseline import (
    Baseline,
    committed_files,
    compare_baselines,
    parse_commit_id,
    read_committed_baseline,
    resolve_commit,
    verify_adoption_sources,
)
from tools.quality_ratchet_comparison import compare_snapshots
from tools.quality_ratchet_contracts import Snapshot, snapshot_payload
from tools.quality_ratchet_inventory import hash_sources, source_path
from tools.quality_ratchet_julia_native import ANALYZER_CONFIGURATIONS, ENGINE_ROOT
from tools.quality_ratchet_profiles import PROFILE_CONFIGURATIONS, REQUIRED_PROFILES
from tools.quality_ratchet_succession import Transition

POLICY_PATH = "tools/quality_ratchet_policy.toml"
POLICY_SCHEMA = "quality-ratchet.policy.v1"


def policy_origin(data: bytes) -> str:
    """Validate the exact policy and return its immutable historical commit.

    Parameters
    ----------
    data : bytes
        Policy read from a regular source file or its trusted Git blob.

    Returns
    -------
    str
        The full immutable commit identifier admitted by this policy.

    Raises
    ------
    ValueError
        If the carrier or required profile sequence is incomplete or changed.
    """
    policy = tomllib.loads(data.decode("utf-8"))
    if (
        set(policy) != {"schema", "origin_commit", "profiles"}
        or policy["schema"] != POLICY_SCHEMA
        or policy["profiles"] != list(REQUIRED_PROFILES)
    ):
        raise ValueError("Invalid or incomplete required quality policy")
    return parse_commit_id(policy["origin_commit"])


def required_baseline_scope(baseline: Baseline, origin: str) -> None:
    """Require every complete profile and the immutable origin in a carrier.

    Parameters
    ----------
    baseline : Baseline
        Parsed baseline carrier whose authority has been established separately.
    origin : str
        Full historical commit declared by the required policy.

    Raises
    ------
    ValueError
        If the origin, required profile set or completeness differs.
    """
    if baseline.origin_commit != origin or set(baseline.profiles) != set(
        REQUIRED_PROFILES
    ):
        raise ValueError("Required baseline authority or profile set changed")
    if any(
        name != snapshot.profile or not snapshot.complete
        for name, snapshot in baseline.profiles.items()
    ):
        raise ValueError("Required baseline contains an incomplete profile")


def required_authority(
    root: Path, base: str, *, recording: bool
) -> tuple[str, Baseline | None]:
    """Read policy and previous baseline from the same trusted Git commit.

    Parameters
    ----------
    root : Path
        Worktree that contains the candidate policy.
    base : str
        Full trusted Git base selected before running the native processes.
    recording : bool
        Permit the initial record at the policy's original clean checkout.

    Returns
    -------
    tuple of str and Baseline or None
        Immutable origin and committed baseline when already adopted.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If established authority has drifted or first recording is unqualified.
    """
    raw = source_path(root, POLICY_PATH).read_bytes()
    origin = policy_origin(raw)
    previous = read_committed_baseline(root, base)
    committed = committed_files(root, base, (POLICY_PATH,)).get(POLICY_PATH)
    if previous is not None:
        if committed != raw:
            raise ValueError("Required quality policy differs from the trusted base")
        required_baseline_scope(previous, origin)
        return origin, previous
    if not recording or base != origin or resolve_commit(root, "HEAD") != origin:
        raise ValueError("Required quality baseline has no trusted Git authority")
    if committed is not None and committed != raw:
        raise ValueError("Initial required quality policy differs from its origin")
    status = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=20,
    )
    if status.stdout not in (b"", f"?? {POLICY_PATH}\0".encode()):
        raise ValueError(
            "Initial required recording needs the clean historical checkout"
        )
    return origin, None


def qualify_required_record(root: Path, base: str, actual: Baseline) -> None:
    """Bind a complete first record to the current exact historical checkout.

    Parameters
    ----------
    root : Path
        Historical Git checkout that produced the native findings.
    base : str
        Full historical base already qualified by the required policy.
    actual : Baseline
        Complete carrier drawn from the actual native collectors.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If tracked inputs moved or any historical source differs from its Git blob.
    """
    origin, previous = required_authority(root, base, recording=True)
    if previous is not None:
        raise ValueError("Required baseline was adopted during initial recording")
    required_baseline_scope(actual, origin)
    sources = tuple(
        sorted(
            {path for profile in actual.profiles.values() for path in profile.sources}
        )
    )
    originals = committed_files(root, origin, sources)
    for profile in actual.profiles.values():
        for path, expected in profile.sources.items():
            if (
                path not in originals
                or hashlib.sha256(originals[path]).hexdigest() != expected
            ):
                raise ValueError(
                    "Required historical capture does not match original source"
                )
    verify_adoption_sources(root, base, actual)


def required_input_state(root: Path) -> tuple[bytes, dict[str, str]]:
    """Capture the visible Git path inventory and native configuration digests.

    Parameters
    ----------
    root : Path
        Actual repository containing the complete set of native profiles.

    Returns
    -------
    tuple of bytes and dict
        Native NUL-framed inventory and exact configuration source hashes.
    """
    inventory = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=20,
    )
    paths = tuple(
        sorted({path for group in PROFILE_CONFIGURATIONS.values() for path in group})
    )
    source_configs = hash_sources(root, paths)
    analyzer_configs = hash_sources(ENGINE_ROOT, ANALYZER_CONFIGURATIONS)
    source_configs.update(
        {
            f"tools/quality_ratchet_julia/{path}": digest
            for path, digest in analyzer_configs.items()
        }
    )
    return inventory.stdout, source_configs


def verify_required_captures(
    root: Path, actual: Baseline, before: tuple[bytes, dict[str, str]]
) -> None:
    """Ensure the combined capture describes one consistent set of source bytes.

    Parameters
    ----------
    root : Path
        Repository analysed by every native profile.
    actual : Baseline
        The complete snapshots produced by the current native processes.
    before : tuple of bytes and dict
        Git inventory and profile configuration digests before native analysis.

    Raises
    ------
    ValueError
        If a source, inventory or configuration changed during collection.
    """
    if required_input_state(root) != before:
        raise ValueError(
            "Required source inventory or configuration changed during analysis"
        )
    paths = tuple(
        sorted(
            {path for snapshot in actual.profiles.values() for path in snapshot.sources}
        )
    )
    observed = hash_sources(root, paths)
    for snapshot in actual.profiles.values():
        if any(observed[path] != digest for path, digest in snapshot.sources.items()):
            raise ValueError("Required source bytes changed between native profiles")


def compare_required_baselines(
    previous: Baseline,
    candidate: Baseline,
    actual: Baseline,
    origin: str,
    transitions: tuple[Transition, ...] = (),
) -> tuple[str, ...]:
    """Compare every actual profile with committed and proposed allowances.

    Parameters
    ----------
    previous : Baseline
        Independently trusted baseline from the selected base commit.
    candidate : Baseline
        Proposed shrink-only historical allowances for this worktree.
    actual : Baseline
        Complete snapshots from the real native processes in this invocation.
    origin : str
        Immutable historical origin from the trusted policy.
    transitions : tuple[Transition, ...]
        Validated declared tool successions; each applies only to exactly the
        identity change it declares.

    Returns
    -------
    tuple of str
        Profile-qualified refusals and violations in deterministic order.
    """
    for baseline in (previous, candidate, actual):
        required_baseline_scope(baseline, origin)
    violations = set(compare_baselines(previous, candidate, transitions))
    for profile in REQUIRED_PROFILES:
        for trusted in (previous, candidate):
            violations.update(
                f"{profile}: {violation}"
                for violation in compare_snapshots(
                    trusted.profiles[profile], actual.profiles[profile], transitions
                )
            )
    return tuple(sorted(violations))


def qualify_adoption(
    root: Path, base: str, baseline: Baseline, actual: Snapshot
) -> None:
    """Verify initial original source/configuration authority and actual findings.

    Parameters
    ----------
    root : Path
        Actual selected source repository.
    base : str
        Immutable original base commit.
    baseline : Baseline
        Candidate adoption carrier, before qualification.
    actual : Snapshot
        Complete current analysis from the pinned native quality process.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If candidate authority, native findings or original configurations disagree.
    """
    if set(baseline.profiles) != {actual.profile}:
        raise ValueError("Initial baseline differs from the actual native profile set")
    if snapshot_payload(baseline.profiles[actual.profile]) != snapshot_payload(actual):
        raise ValueError("Initial baseline is not the complete actual native snapshot")
    verify_adoption_sources(root, base, baseline)
    configuration = PROFILE_CONFIGURATIONS[actual.profile]
    originals = committed_files(root, baseline.origin_commit, configuration)
    for relative in configuration:
        if originals.get(relative) != source_path(root, relative).read_bytes():
            raise ValueError(
                "Initial native quality configuration differs from its origin"
            )


def check_native_baseline(
    root: Path,
    base: str,
    candidate: Baseline,
    actual: Snapshot,
    transitions: tuple[Transition, ...] = (),
) -> tuple[str, ...]:
    """Check shrink-only candidate allowances and actual native source diagnostics.

    Parameters
    ----------
    root : Path
        Actual selected source repository.
    base : str
        Resolved immutable original base commit.
    candidate : Baseline
        Working-tree carrier, never substituted for original baseline authority.
    actual : Snapshot
        Complete current native quality scan.
    transitions : tuple[Transition, ...]
        Validated declared tool successions for exact identity changes.

    Returns
    -------
    tuple[str, ...]
        Exact refusals; zero only qualifies this actual native profile.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If initial adoption, original Git state or a profile is unqualified.
    """
    original = read_committed_baseline(root, base)
    if original is None:
        qualify_adoption(root, base, candidate, actual)
        return ()
    if set(original.profiles) != {actual.profile}:
        raise ValueError("Original baseline differs from the actual native profile set")
    refusals = list(compare_baselines(original, candidate, transitions))
    refusals.extend(
        compare_snapshots(original.profiles[actual.profile], actual, transitions)
    )
    if set(candidate.profiles) != {actual.profile}:
        refusals.append("Candidate native profile set changed")
    elif snapshot_payload(candidate.profiles[actual.profile]) != snapshot_payload(
        actual
    ):
        refusals.append(
            "Candidate baseline must retain the exact current native snapshot"
        )
    return tuple(sorted(set(refusals)))
