# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — loading and canary verification of tool successions
"""Load the transition file against its trusted base and verify the canaries.

The candidate's transition file must begin with the trusted base commit's
entries unchanged; entries are only ever appended. Every transition the
current capture applies, meaning the base snapshot's identity and the
recomputed identity match it, must then reproduce its recorded canary run
with the engine the new identity names. A configuration change has no canary:
the TOML diff of its configuration files is recomputed against the trusted
base instead
(:func:`tools.quality_ratchet_configuration_check.tip_configuration_refusals`);
the per-commit walk checks the shape of the commit that added it.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_baseline import committed_files, resolve_commit
from tools.quality_ratchet_canaries import run_profile_canaries
from tools.quality_ratchet_configuration_check import tip_configuration_refusals
from tools.quality_ratchet_contracts import Snapshot, decode_quality_json
from tools.quality_ratchet_inventory import source_candidate
from tools.quality_ratchet_succession import (
    CONFIGURATION_REASON,
    TRANSITIONS_PATH,
    Transition,
)
from tools.quality_ratchet_succession_check import transition_for, verify_canary_run
from tools.quality_ratchet_succession_parse import parse_transitions


def require_append_only(base: bytes | None, candidate: bytes | None) -> None:
    """Refuse a candidate file that rewrites or removes a trusted base entry.

    Parameters
    ----------
    base : bytes | None
        The trusted base commit's file, or None when it has none.
    candidate : bytes | None
        The working tree's file, or None when it has none.

    Raises
    ------
    ValueError
        If the candidate drops the file, or its entries do not begin with the
        base's entries, member for member.
    """
    if base is None:
        return
    if candidate is None:
        raise ValueError("The trusted native quality transition file was removed")
    old = object_mapping(decode_quality_json(base)).get("transitions")
    new = object_mapping(decode_quality_json(candidate)).get("transitions")
    if not isinstance(old, list) or not isinstance(new, list) or new[: len(old)] != old:
        raise ValueError("Native quality transitions are append-only")


def load_transitions(root: Path, base: str) -> tuple[Transition, ...]:
    """Read and validate the candidate's transition file against the base.

    Parameters
    ----------
    root : Path
        Candidate repository root.
    base : str
        Resolved trusted base commit.

    Returns
    -------
    tuple[Transition, ...]
        The validated entries, or an empty tuple when neither commit has a file.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If Git, the append-only rule or an entry refuses.
    """
    trusted = committed_files(root, base, (TRANSITIONS_PATH,)).get(TRANSITIONS_PATH)
    path = source_candidate(root, TRANSITIONS_PATH)
    if path.exists() and not path.is_file():
        raise ValueError("The native quality transition file is not a regular file")
    candidate = path.read_bytes() if path.exists() else None
    require_append_only(trusted, candidate)
    return () if candidate is None else parse_transitions(candidate)


def canary_refusals(
    root: Path,
    args: argparse.Namespace,
    trusted: Mapping[str, Snapshot],
    actual: Mapping[str, Snapshot],
    transitions: tuple[Transition, ...],
) -> tuple[str, ...]:
    """Run the canaries of every transition the current capture applies.

    Configuration changes run no canary; the TOML diff of their files is
    recomputed against the trusted base ``args.base_ref`` instead.

    Parameters
    ----------
    root : Path
        Candidate repository root.
    args : argparse.Namespace
        The ratchet's native tool arguments.
    trusted : Mapping[str, Snapshot]
        The trusted base snapshots by profile.
    actual : Mapping[str, Snapshot]
        The snapshots captured now by profile.
    transitions : tuple[Transition, ...]
        The validated candidate transitions.

    Returns
    -------
    tuple[str, ...]
        Profile-qualified refusals, sorted; empty when no transition applies.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If Git refuses or a configuration file is missing or not valid TOML.
    """
    refusals: set[str] = set()
    configuration: list[Transition] = []
    for profile, snapshot in sorted(actual.items()):
        original = trusted.get(profile)
        if original is None or original.identity == snapshot.identity:
            continue
        transition = transition_for(
            transitions, profile, original.identity, snapshot.identity
        )
        if transition is None:
            continue
        if transition.reason == CONFIGURATION_REASON:
            configuration.append(transition)
            continue
        try:
            run = run_profile_canaries(root, args, profile)
        except ValueError as error:
            refusals.add(f"{profile}: {error}")
            continue
        refusals.update(
            f"{profile}: {refusal}"
            for refusal in verify_canary_run(
                transition,
                enabled_rules=run.enabled_rules,
                fixture_digest=run.fixture_digest,
                platform=run.platform,
                engine_digest=run.engine_digest,
                findings=run.findings,
            )
        )
    if configuration:
        refusals.update(
            tip_configuration_refusals(
                root, resolve_commit(root, args.base_ref), configuration
            )
        )
    return tuple(sorted(refusals))
