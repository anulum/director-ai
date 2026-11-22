# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — seeded known-positive canaries for native tool successions
"""Run a profile's seeded canaries so a tool change cannot silently stop checking.

Equal findings on unchanged sources prove nothing for a rule with no findings
there. Each profile that may change its tool identity therefore keeps a
canary fixture under :data:`CANARY_ROOT`, one seeded known positive per rule
it enforces, outside every profile's source roots. A transition is applied
only after the profile's runner reproduces the recorded canary findings with
the engine binary the new identity names.

A profile receives its runner and fixture at its first transition. Until
then, :func:`run_profile_canaries` refuses it by name, so no transition of
that profile can apply.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tools.quality_ratchet_succession import FindingKey

CANARY_ROOT = "tests/fixtures/quality_ratchet_canaries"


@dataclass(frozen=True)
class CanaryRun:
    """The result of one canary run.

    Attributes
    ----------
    enabled_rules : tuple[str, ...]
        The rules the profile enforces, derived from its configuration.
    fixture_digest : str
        Digest of the fixture the run read.
    platform : str
        The running process's platform, for example ``linux-x64``.
    engine_digest : str
        Digest of the engine binary the run executed.
    findings : tuple[FindingKey, ...]
        The canary findings.
    """

    enabled_rules: tuple[str, ...]
    fixture_digest: str
    platform: str
    engine_digest: str
    findings: tuple[FindingKey, ...]


CanaryRunner = Callable[[Path, argparse.Namespace], CanaryRun]
CANARY_RUNNERS: dict[str, CanaryRunner] = {}
"""Canary runners by profile; a profile joins at its first transition."""


def fixture_digest(root: Path, profile: str) -> str:
    """Digest a profile's canary fixture as a sorted set of named files.

    Parameters
    ----------
    root : Path
        Repository root.
    profile : str
        The native profile.

    Returns
    -------
    str
        SHA-256 over each file's relative name and byte digest, in name order.

    Raises
    ------
    ValueError
        If the fixture is missing, empty, or holds anything but regular files.
    """
    directory = root / CANARY_ROOT / profile
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Profile {profile} has no canary fixture")
    entries = sorted(directory.rglob("*"))
    files = [entry for entry in entries if entry.is_file() and not entry.is_symlink()]
    if not files or len(files) != len([e for e in entries if not e.is_dir()]):
        raise ValueError(f"Profile {profile} canary fixture is not regular files")
    digest = hashlib.sha256()
    for path in files:
        name = path.relative_to(directory).as_posix()
        content = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{name}\0{content}\n".encode())
    return digest.hexdigest()


def run_profile_canaries(
    root: Path, args: argparse.Namespace, profile: str
) -> CanaryRun:
    """Run one profile's canaries through its registered runner.

    Parameters
    ----------
    root : Path
        Repository root.
    args : argparse.Namespace
        The ratchet's native tool arguments.
    profile : str
        The native profile.

    Returns
    -------
    CanaryRun
        The run's result.

    Raises
    ------
    ValueError
        If the profile has no canary runner.
    """
    runner = CANARY_RUNNERS.get(profile)
    if runner is None:
        raise ValueError(f"Profile {profile} has no canary runner")
    return runner(root, args)
