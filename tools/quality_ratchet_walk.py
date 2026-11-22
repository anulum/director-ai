# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the per-commit static walk of a pushed range
"""Check every pushed commit's ratchet files against its own first parent.

A push is a fast-forward of several commits. Comparing only the tip with
itself, or only with the previous tip, would let an earlier commit raise
the baseline or adopt an identity change that a later commit hides. This
module walks ``git rev-list --reverse --first-parent start..tip`` and
checks each commit against its first parent using Git objects alone, with
no native tool, so a push that spans a tool change is still checked:

- the policy is unchanged, and both baselines carry its scope;
- the baseline only shrinks, and an identity changes only through a
  declared transition (:func:`compare_baselines`);
- the transitions file is append-only, and every entry parses;
- a ratchet file the parent has is never removed, emptied or corrupted.

A commit where neither it nor its parent has any ratchet file comes before
the ratchet's adoption and is skipped by name. The commit that adds them is
the adoption, checked for the policy's scope. The native capture itself is
checked once, at the tip, by the ratchet's ``check`` against the tip's
first parent; :func:`range_canary_refusals` runs there the canaries of
every entry added in the range whose new identity is the tip's.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.quality_ratchet_baseline import (
    BASELINE_PATH,
    committed_files,
    compare_baselines,
    parse_baseline,
    resolve_commit,
)
from tools.quality_ratchet_configuration_check import commit_configuration_refusals
from tools.quality_ratchet_gate import (
    POLICY_PATH,
    policy_origin,
    required_baseline_scope,
)
from tools.quality_ratchet_historical import (
    HISTORICAL_PATH,
    HistoricalEntry,
    apply_historical,
    honoured_entries,
    parse_historical,
    require_entries_append_only,
    verify_decision_digests,
)
from tools.quality_ratchet_succession import (
    CONFIGURATION_REASON,
    TRANSITIONS_PATH,
    Transition,
)
from tools.quality_ratchet_succession_load import require_append_only
from tools.quality_ratchet_succession_parse import parse_transitions
from tools.quality_ratchet_walk_tip import added_entries, superseded_entries

RATCHET_FILES = (BASELINE_PATH, POLICY_PATH, TRANSITIONS_PATH, HISTORICAL_PATH)
MAX_COMMITS = 100
_ZERO = re.compile(r"0{40}(0{24})?")


@dataclass(frozen=True)
class WalkStep:
    """The static check of one pushed commit.

    Attributes
    ----------
    commit : str
        The pushed commit.
    parent : str | None
        Its first parent, or None for a root commit.
    status : str
        ``skipped-before-adoption``, ``adoption`` or ``checked``.
    refusals : tuple[str, ...]
        Sorted refusals; empty means the step qualifies.
    """

    commit: str
    parent: str | None
    status: str
    refusals: tuple[str, ...]


def _git(
    root: Path, *arguments: str, check: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run Git without replacement objects; ``check`` raises on failure."""
    return subprocess.run(
        ["git", "--no-replace-objects", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=check,
        timeout=60,
    )


def push_range(
    root: Path,
    before: str,
    tip: str,
    default_ref: str,
    *,
    max_commits: int = MAX_COMMITS,
) -> tuple[str, tuple[str, ...]]:
    """Return the walk's start and its first-parent commits, oldest first.

    Parameters
    ----------
    root : Path
        Repository root.
    before : str
        The pushed ref's previous commit; all zeros for a new ref.
    tip : str
        The pushed commit.
    default_ref : str
        The default branch, whose merge base starts a new ref's walk.
    max_commits : int
        The bound on the walk's length.

    Returns
    -------
    tuple[str, tuple[str, ...]]
        The start commit and the commits after it, up to the tip.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If a commit does not resolve, ``before`` is not an ancestor of the
        tip, or the walk exceeds the bound.
    """
    tip = resolve_commit(root, tip)
    if _ZERO.fullmatch(before):
        base = _git(root, "merge-base", tip, resolve_commit(root, default_ref))
        if base.returncode != 0:
            raise ValueError("A new ref has no merge base with the default branch")
        start = base.stdout.strip()
    else:
        start = resolve_commit(root, before)
        if _git(root, "merge-base", "--is-ancestor", start, tip).returncode != 0:
            raise ValueError(
                "The push rewrites history: its previous commit is not an ancestor"
            )
    return start, _listed(root, start, tip, max_commits, f"--before {start}")


def pr_range(
    root: Path, base: str, tip: str, *, max_commits: int = MAX_COMMITS
) -> tuple[str, tuple[str, ...]]:
    """Return a pull request's walk: from its merge base with the base branch.

    The base branch's head may be ahead of the commit the branch was cut
    from, so the walk starts at their merge base, never at the base's head.

    Parameters
    ----------
    root : Path
        Repository root.
    base : str
        The pull request's base commit, the base branch's head.
    tip : str
        The pull request's head commit.
    max_commits : int
        The bound on the walk's length.

    Returns
    -------
    tuple[str, tuple[str, ...]]
        The merge base and the commits after it, up to the head.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If a commit does not resolve, there is no merge base, or the walk
        exceeds the bound.
    """
    tip = resolve_commit(root, tip)
    base = resolve_commit(root, base)
    merged = _git(root, "merge-base", base, tip)
    if merged.returncode != 0:
        raise ValueError("The pull request has no merge base with its base")
    start = merged.stdout.strip()
    return start, _listed(root, start, tip, max_commits, f"--pr-base {base}")


def _listed(
    root: Path, start: str, tip: str, max_commits: int, origin: str
) -> tuple[str, ...]:
    """List ``start..tip`` first-parent, oldest first, within the bound."""
    listed = _git(
        root, "rev-list", "--reverse", "--first-parent", f"{start}..{tip}", check=True
    )
    commits = tuple(listed.stdout.split())
    if len(commits) > max_commits:
        raise ValueError(
            f"The walk has {len(commits)} commits, more than {max_commits}; run "
            f"python -m tools.quality_ratchet_walk {origin} --to {tip} "
            f"--max-commits {len(commits)}"
        )
    return commits


def first_parent(root: Path, commit: str) -> str | None:
    """Return a commit's first parent, or None for a root commit.

    Parameters
    ----------
    root : Path
        Repository root.
    commit : str
        A resolved commit.

    Returns
    -------
    str | None
        The first parent.

    Raises
    ------
    subprocess.CalledProcessError
        If Git cannot list the parents; an unknown parent is never read as
        a root commit, which would turn a full check into an adoption.
    """
    listed = _git(root, "rev-list", "--parents", "-n", "1", commit, check=True)
    parents = listed.stdout.split()
    return parents[1] if len(parents) > 1 else None


def check_step(
    root: Path,
    parent: str | None,
    commit: str,
    honoured: Mapping[tuple[str, str], HistoricalEntry] | None = None,
) -> WalkStep:
    """Check one commit's ratchet files against its first parent.

    Parameters
    ----------
    root : Path
        Repository root.
    parent : str | None
        The first parent, or None for a root commit.
    commit : str
        The commit.
    honoured : Mapping[tuple[str, str], HistoricalEntry] | None
        Reviewed pre-mechanism identity changes, by commit and profile.

    Returns
    -------
    WalkStep
        The step's status and refusals.
    """
    old = {} if parent is None else committed_files(root, parent, RATCHET_FILES)
    new = committed_files(root, commit, RATCHET_FILES)
    if not old and not new:
        return WalkStep(commit, parent, "skipped-before-adoption", ())
    refusals = {
        f"A ratchet file was removed or emptied: {path}"
        for path, data in old.items()
        if not new.get(path, b"").strip()
    }
    if refusals:
        return WalkStep(commit, parent, "checked", tuple(sorted(refusals)))
    adoption = BASELINE_PATH not in old and POLICY_PATH not in old
    try:
        origin = policy_origin(new[POLICY_PATH])
        candidate = parse_baseline(new[BASELINE_PATH])
        required_baseline_scope(candidate, origin)
        transitions: tuple[Transition, ...] = ()
        inherited = 0
        if TRANSITIONS_PATH in new:
            require_append_only(old.get(TRANSITIONS_PATH), new[TRANSITIONS_PATH])
            transitions = parse_transitions(new[TRANSITIONS_PATH])
            if TRANSITIONS_PATH in old:
                inherited = len(parse_transitions(old[TRANSITIONS_PATH]))
        if HISTORICAL_PATH in new:
            require_entries_append_only(old.get(HISTORICAL_PATH), new[HISTORICAL_PATH])
            parse_historical(new[HISTORICAL_PATH])
        if not adoption:
            if old.get(POLICY_PATH) != new[POLICY_PATH]:
                raise ValueError("The required quality policy changed")
            previous = parse_baseline(old[BASELINE_PATH])
            required_baseline_scope(previous, origin)
            mapped, mismatched = apply_historical(
                commit, previous, candidate, honoured or {}
            )
            refusals.update(mismatched)
            refusals.update(compare_baselines(previous, mapped, transitions))
            configuration = [
                entry
                for entry in transitions[inherited:]
                if entry.reason == CONFIGURATION_REASON
            ]
            if configuration and parent is not None:
                refusals.update(
                    commit_configuration_refusals(
                        root,
                        parent,
                        commit,
                        configuration,
                        previous.profiles,
                        candidate.profiles,
                    )
                )
    except (KeyError, ValueError) as error:
        name = f"missing {error}" if isinstance(error, KeyError) else str(error)
        refusals.add(f"A ratchet file does not qualify: {name}")
    return WalkStep(
        commit, parent, "adoption" if adoption else "checked", tuple(sorted(refusals))
    )


@dataclass(frozen=True)
class WalkReport:
    """The result of walking a pushed range.

    Attributes
    ----------
    start : str
        The commit the walk starts after.
    steps : tuple[WalkStep, ...]
        One step per pushed commit, oldest first.
    superseded : tuple[str, ...]
        Added entries the tip no longer applies.
    refusals : tuple[str, ...]
        Every step's refusals, prefixed by its commit, sorted.
    """

    start: str
    steps: tuple[WalkStep, ...]
    superseded: tuple[str, ...]
    refusals: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        """Return the report as JSON-ready data, the walk's CI artefact.

        Returns
        -------
        dict[str, object]
            The start, every step, the superseded entries and the refusals.
        """
        return {
            "start": self.start,
            "steps": [
                {
                    "commit": step.commit,
                    "parent": step.parent,
                    "status": step.status,
                    "refusals": list(step.refusals),
                }
                for step in self.steps
            ],
            "superseded": list(self.superseded),
            "refusals": list(self.refusals),
        }


def walk(
    root: Path,
    before: str | None,
    tip: str,
    default_ref: str,
    *,
    max_commits: int = MAX_COMMITS,
    decisions_root: Path | None = None,
    pr_base: str | None = None,
) -> WalkReport:
    """Walk a pushed range and report every step.

    Parameters
    ----------
    root : Path
        Repository root.
    before : str | None
        The pushed ref's previous commit, all zeros for a new ref; None for
        a pull request.
    tip : str
        The pushed commit, or the pull request's head.
    default_ref : str
        The default branch for a new ref.
    max_commits : int
        The bound on the walk's length.
    pr_base : str | None
        A pull request's base commit; the walk then starts at its merge
        base with ``tip``. Exactly one of ``before`` and ``pr_base`` is given.
    decisions_root : Path | None
        Where the decision records live, to check each historical digest;
        None where they are not available, as in CI.

    Returns
    -------
    WalkReport
        Every step, the superseded entries and the refusals.
    """
    if before is not None and pr_base is None:
        start, commits = push_range(
            root, before, tip, default_ref, max_commits=max_commits
        )
    elif pr_base is not None and before is None:
        start, commits = pr_range(root, pr_base, tip, max_commits=max_commits)
    else:
        raise ValueError("A walk needs exactly one of a previous commit or a PR base")
    resolved = resolve_commit(root, tip)
    tip_files = committed_files(root, resolved, (BASELINE_PATH, HISTORICAL_PATH))
    entries = (
        parse_historical(tip_files[HISTORICAL_PATH])
        if HISTORICAL_PATH in tip_files
        else ()
    )
    honoured, historical_refusals = honoured_entries(root, entries, resolved)
    if decisions_root is not None:
        historical_refusals += verify_decision_digests(entries, decisions_root)
    steps = tuple(
        check_step(root, first_parent(root, commit), commit, honoured)
        for commit in commits
    )
    identities = (
        {
            name: snapshot.identity
            for name, snapshot in parse_baseline(
                tip_files[BASELINE_PATH]
            ).profiles.items()
        }
        if BASELINE_PATH in tip_files
        else {}
    )
    return WalkReport(
        start,
        steps,
        superseded_entries(added_entries(root, start, resolved), identities),
        tuple(
            sorted(
                [
                    f"{step.commit[:12]}: {refusal}"
                    for step in steps
                    for refusal in step.refusals
                ]
                + [f"historical: {refusal}" for refusal in historical_refusals]
            )
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Walk a pushed range from the command line.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments, or the process's own.

    Returns
    -------
    int
        Zero when every step qualifies, one for a refusal, two when the walk
        itself cannot run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--before", help="The pushed ref's previous commit")
    origin.add_argument("--pr-base", help="A pull request's base commit")
    parser.add_argument("--to", required=True)
    parser.add_argument("--default-ref", default="origin/main")
    parser.add_argument("--max-commits", type=int, default=MAX_COMMITS)
    parser.add_argument(
        "--decisions-root",
        type=Path,
        help="Check each historical record's digest against these decision records",
    )
    args = parser.parse_args(argv)
    try:
        report = walk(
            args.root.resolve(strict=True),
            args.before,
            args.to,
            args.default_ref,
            max_commits=args.max_commits,
            decisions_root=args.decisions_root,
            pr_base=args.pr_base,
        )
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Native quality walk refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report.payload(), indent=2))
    return 1 if report.refusals else 0


if __name__ == "__main__":
    raise SystemExit(main())
