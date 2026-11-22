# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — what the pushed range's tip still owes its transitions
"""Range-level checks that need the tip: added entries, canaries and the workflow.

A transition entry added anywhere in a pushed range runs its canary at the tip
when the tip still applies it (:func:`range_canary_refusals`); an entry the tip
no longer applies is named by :func:`superseded_entries`, because its canary
ran only in the lab gate. :func:`audit_ratchet_workflow` refuses a quality
workflow that would compare a pushed commit with itself.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.quality_ratchet_baseline import committed_files
from tools.quality_ratchet_canaries import run_profile_canaries
from tools.quality_ratchet_contracts import Snapshot
from tools.quality_ratchet_succession import (
    CONFIGURATION_REASON,
    TRANSITIONS_PATH,
    Transition,
)
from tools.quality_ratchet_succession_check import verify_canary_run
from tools.quality_ratchet_succession_parse import parse_transitions


def entries_since(
    root: Path, start: str, transitions: Sequence[Transition]
) -> tuple[Transition, ...]:
    """Return the entries of a transitions file that a start commit lacks.

    Parameters
    ----------
    root : Path
        Repository root.
    start : str
        A resolved commit.
    transitions : Sequence[Transition]
        A validated, append-only transitions file, in file order.

    Returns
    -------
    tuple[Transition, ...]
        The entries after those the start commit already had.
    """
    before = committed_files(root, start, (TRANSITIONS_PATH,)).get(TRANSITIONS_PATH)
    known = 0 if before is None else len(parse_transitions(before))
    return tuple(transitions[known:])


def added_entries(root: Path, start: str, tip: str) -> tuple[Transition, ...]:
    """Return the transition entries the range adds, in file order.

    Parameters
    ----------
    root : Path
        Repository root.
    start : str
        The walk's start commit.
    tip : str
        The pushed commit.

    Returns
    -------
    tuple[Transition, ...]
        The tip's entries after those the start already had.
    """
    after = committed_files(root, tip, (TRANSITIONS_PATH,)).get(TRANSITIONS_PATH)
    return () if after is None else entries_since(root, start, parse_transitions(after))


def superseded_entries(
    entries: Sequence[Transition], tip_identities: Mapping[str, str]
) -> tuple[str, ...]:
    """Name each added entry whose new identity is not the tip's.

    Such an entry's canary ran only in the lab gate, because the tip's
    engine is no longer the one it names.

    Parameters
    ----------
    entries : Sequence[Transition]
        The entries the range adds.
    tip_identities : Mapping[str, str]
        The tip baseline's identity of each profile.

    Returns
    -------
    tuple[str, ...]
        One line per superseded entry.
    """
    return tuple(
        f"{entry.profile}: {entry.old_identity[:12]} -> {entry.new_identity[:12]} "
        "is superseded at the tip; its canary ran only in the lab gate"
        for entry in entries
        if tip_identities.get(entry.profile) != entry.new_identity
    )


def range_canary_refusals(
    root: Path,
    args: argparse.Namespace,
    entries: Sequence[Transition],
    actual: Mapping[str, Snapshot],
) -> tuple[str, ...]:
    """Run at the tip the canaries of every added entry it still applies.

    A configuration change has no canary; the per-commit walk checked it
    statically at the commit that added it.

    Parameters
    ----------
    root : Path
        The tip's worktree.
    args : argparse.Namespace
        The ratchet's native tool arguments.
    entries : Sequence[Transition]
        The entries the range adds.
    actual : Mapping[str, Snapshot]
        The snapshots captured now at the tip.

    Returns
    -------
    tuple[str, ...]
        Profile-qualified refusals, sorted.
    """
    refusals: set[str] = set()
    for entry in entries:
        snapshot = actual.get(entry.profile)
        if (
            snapshot is None
            or snapshot.identity != entry.new_identity
            or entry.reason == CONFIGURATION_REASON
        ):
            continue
        try:
            run = run_profile_canaries(root, args, entry.profile)
        except ValueError as error:
            refusals.add(f"{entry.profile}: {error}")
            continue
        refusals.update(
            f"{entry.profile}: {refusal}"
            for refusal in verify_canary_run(
                entry,
                enabled_rules=run.enabled_rules,
                fixture_digest=run.fixture_digest,
                platform=run.platform,
                engine_digest=run.engine_digest,
                findings=run.findings,
            )
        )
    return tuple(sorted(refusals))


def audit_ratchet_workflow(text: str) -> tuple[str, ...]:
    """Refuse a quality workflow whose push check compares a commit with itself.

    Parameters
    ----------
    text : str
        The quality workflow's source.

    Returns
    -------
    tuple[str, ...]
        Refusals; empty means the push is walked and the tip is checked
        against its first parent.
    """
    refusals = []
    if "python -m tools.quality_ratchet_walk" not in text.replace(
        '"$QUALITY_PYTHON"', "python"
    ):
        refusals.append("The workflow does not walk the pushed commits")
    if "github.event.before" not in text:
        refusals.append("The walk does not start at the push's previous commit")
    if "--pr-base" not in text or re.search(
        r"BEFORE:\s*\$\{\{\s*github\.event\.pull_request\.base\.sha", text
    ):
        refusals.append("A pull request is not walked from its merge base")
    if re.search(r"TRUSTED_BASE:\s*\$\{\{[^}]*\|\|\s*github\.sha\s*\}\}", text):
        refusals.append("On a push the trusted base is the pushed commit itself")
    if '"$GITHUB_SHA^1"' not in text:
        refusals.append("The tip is not checked against its first parent")
    return tuple(refusals)
