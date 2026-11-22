# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the comparison a declared tool succession permits
"""Compare two snapshots across a declared identity change.

A transition applies only when its old identity is the trusted base
snapshot's and its new identity is the one the candidate's adapter computed
now; neither is ever taken from the record alone. Findings are compared by
their message-free key (path, rule, line, column, cell):

- on a byte-identical source, each key keeps its multiplicity, since one
  position may carry several findings; only a declared drop (TS-2) may remove
  a key, and then every occurrence of it;
- a key whose sorted messages alone changed is listed in the record;
- a rule whose findings all disappear from the unchanged sources is refused,
  even when every drop is declared;
- a new or modified source has no findings, as without a transition.

The canary run of :func:`verify_canary_run` shows that every enabled rule
still fires on seeded known positives (TS-3).
"""

from __future__ import annotations

from collections.abc import Iterable

from tools.quality_ratchet_contracts import Finding, Snapshot
from tools.quality_ratchet_succession import FindingKey, Transition


def finding_key(finding: Finding) -> FindingKey:
    """Return a finding's message-free comparison key.

    Parameters
    ----------
    finding : Finding
        A native finding.

    Returns
    -------
    FindingKey
        Its path, rule, line, column and cell.
    """
    return FindingKey(
        finding.path, finding.rule, finding.line, finding.column, finding.cell
    )


def transition_for(
    transitions: Iterable[Transition], profile: str, old: str, new: str
) -> Transition | None:
    """Return the declared transition between two identities of a profile.

    Parameters
    ----------
    transitions : Iterable[Transition]
        Every validated transition of the candidate file.
    profile : str
        The native profile.
    old : str
        The trusted base snapshot's identity.
    new : str
        The identity the candidate's adapter computed now.

    Returns
    -------
    Transition | None
        The matching record, or None when no record declares this change.
    """
    for transition in transitions:
        if (transition.profile, transition.old_identity, transition.new_identity) == (
            profile,
            old,
            new,
        ):
            return transition
    return None


def _messages(snapshot: Snapshot, paths: set[str]) -> dict[FindingKey, tuple[str, ...]]:
    """Map each key on the given paths to the sorted messages of its findings."""
    grouped: dict[FindingKey, list[str]] = {}
    for finding in snapshot.findings:
        if finding.path in paths:
            grouped.setdefault(finding_key(finding), []).append(finding.message)
    return {key: tuple(sorted(messages)) for key, messages in grouped.items()}


def compare_transition(
    original: Snapshot, current: Snapshot, transition: Transition
) -> tuple[str, ...]:
    """Refuse every difference a declared identity change does not permit.

    Parameters
    ----------
    original : Snapshot
        The trusted base snapshot.
    current : Snapshot
        The snapshot computed now under the new identity.
    transition : Transition
        The record declaring the change.

    Returns
    -------
    tuple[str, ...]
        Sorted refusals; empty means the change preserves the findings.
    """
    if (original.profile, original.identity, current.identity) != (
        transition.profile,
        transition.old_identity,
        transition.new_identity,
    ):
        return ("The transition does not declare this identity change",)
    unchanged = {
        path
        for path, digest in original.sources.items()
        if current.sources.get(path) == digest
    }
    before = _messages(original, unchanged)
    after = _messages(current, unchanged)
    declared = {drop.key for drop in transition.drops}
    refusals: set[str] = set()
    for key in set(before) | set(after):
        old, new = len(before.get(key, ())), len(after.get(key, ()))
        if new > old:
            refusals.add(
                "Unchanged source gained a native finding under the new identity: "
                f"{key.path}"
            )
        elif new < old and not (key in declared and new == 0):
            refusals.add(
                f"Unchanged source lost an undeclared native finding: {key.path}"
            )
    refusals.update(
        f"A declared dropped finding is still reported: {key.path}"
        for key in declared
        if key in after
    )
    refusals.update(
        f"A declared dropped finding was never reported: {key.path}"
        for key in declared
        if key not in before
    )
    lost_rules = {key.rule for key in before if key not in after}
    for rule in lost_rules:
        if not any(key.rule == rule for key in after):
            refusals.add(f"Rule {rule} lost every finding on the unchanged sources")
    reworded = {
        key
        for key in set(before) & set(after)
        if len(before[key]) == len(after[key]) and before[key] != after[key]
    }
    if reworded != set(transition.message_only):
        refusals.add("Message-only finding changes differ from the declared list")
    refusals.update(
        f"New or modified source has native findings: {finding.path}"
        for finding in current.findings
        if finding.path not in unchanged
    )
    return tuple(sorted(refusals))


def verify_canary_run(
    transition: Transition,
    *,
    enabled_rules: tuple[str, ...],
    fixture_digest: str,
    platform: str,
    engine_digest: str,
    findings: Iterable[FindingKey],
) -> tuple[str, ...]:
    """Check a canary run made now against the record's canary evidence.

    Parameters
    ----------
    transition : Transition
        The applied transition.
    enabled_rules : tuple[str, ...]
        The rules the profile enforces, derived from its configuration.
    fixture_digest : str
        Digest of the canary fixture as it exists now.
    platform : str
        The platform of the running process.
    engine_digest : str
        Digest of the engine binary that ran.
    findings : Iterable[FindingKey]
        The canary findings of this run.

    Returns
    -------
    tuple[str, ...]
        Sorted refusals; empty means the run reproduces the recorded canaries.
    """
    canary = transition.canary
    if canary is None:
        return ("The transition has no canary run",)
    refusals = []
    if tuple(sorted(enabled_rules)) != canary.enabled_rules:
        refusals.append("The recorded enabled rules differ from the profile's rules")
    if fixture_digest != canary.fixture_digest:
        refusals.append("The canary fixture differs from the recorded fixture")
    if platform != canary.platform:
        refusals.append("The canary ran on a platform other than the recorded one")
    if engine_digest != canary.new_engine_digest:
        refusals.append("The canary ran an engine other than the recorded one")
    if tuple(sorted(findings)) != canary.new_findings:
        refusals.append("The canary findings differ from the recorded findings")
    return tuple(sorted(refusals))
