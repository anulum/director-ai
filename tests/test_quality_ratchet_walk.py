# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the per-commit static walk of a pushed range
"""Walk real Git histories built from the committed baseline and policy."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import tools.quality_ratchet_walk as walk_module
from tests.quality_ratchet_succession_support import (
    NEW_IDENTITY,
    OLD_IDENTITY,
    entry,
    snapshot,
)
from tools.quality_ratchet_baseline import BASELINE_PATH
from tools.quality_ratchet_canaries import CANARY_RUNNERS, CanaryRun
from tools.quality_ratchet_gate import POLICY_PATH
from tools.quality_ratchet_succession import TRANSITIONS_PATH, TRANSITIONS_SCHEMA
from tools.quality_ratchet_succession_parse import parse_transition
from tools.quality_ratchet_walk import (
    MAX_COMMITS,
    check_step,
    first_parent,
    main,
    pr_range,
    push_range,
    walk,
)
from tools.quality_ratchet_walk_tip import (
    added_entries,
    audit_ratchet_workflow,
    range_canary_refusals,
    superseded_entries,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci-quality.yml"
ZERO = "0" * 40


def small_baseline() -> dict[str, Any]:
    """Return the committed baseline with at most three findings per profile.

    Returns
    -------
    dict[str, Any]
        A valid baseline carrier.
    """
    data: dict[str, Any] = json.loads(
        (ROOT / BASELINE_PATH).read_text(encoding="utf-8")
    )
    for profile in data["profiles"].values():
        profile["findings"] = profile["findings"][:3]
    return data


def raised(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a baseline that allows one more go-docs finding.

    Parameters
    ----------
    data : dict[str, Any]
        A baseline carrier.

    Returns
    -------
    dict[str, Any]
        The raised carrier.
    """
    value = copy.deepcopy(data)
    findings = value["profiles"]["go-docs"]["findings"]
    extra = dict(findings[0])
    extra["line"] = extra["line"] + 1_000_000
    findings.append(extra)
    return value


def git(root: Path, *arguments: str) -> str:
    """Run Git in a test repository and return its output.

    Parameters
    ----------
    root : Path
        Test repository.
    *arguments : str
        Git arguments.

    Returns
    -------
    str
        Standard output, stripped.
    """
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def commit(root: Path, files: dict[str, object | bytes | None], message: str) -> str:
    """Write, remove or keep files and commit the result.

    Parameters
    ----------
    root : Path
        Test repository.
    files : dict[str, object | bytes | None]
        Path to JSON data, raw bytes, or None to remove it.
    message : str
        Commit message.

    Returns
    -------
    str
        The new commit.
    """
    for name, value in files.items():
        path = root / name
        if value is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    (root / "notes.txt").write_text(message + "\n", encoding="utf-8")
    git(root, "add", "-A")
    git(
        root,
        "-c",
        "user.name=Walk fixture",
        "-c",
        "user.email=walk@example.test",
        "commit",
        "-q",
        "-m",
        message,
    )
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create a repository whose first commit predates the ratchet.

    Parameters
    ----------
    tmp_path : Path
        Isolated directory.

    Returns
    -------
    Path
        The repository root.
    """
    git(tmp_path, "init", "-q", "-b", "main")
    commit(tmp_path, {}, "before the ratchet")
    return tmp_path


POLICY = (ROOT / POLICY_PATH).read_bytes()


def adopt(root: Path) -> str:
    """Commit the policy and a valid baseline.

    Parameters
    ----------
    root : Path
        Test repository.

    Returns
    -------
    str
        The adoption commit.
    """
    return commit(root, {POLICY_PATH: POLICY, BASELINE_PATH: small_baseline()}, "adopt")


def test_the_four_cases(repo: Path) -> None:
    """Skip before adoption, check the adoption, check, and refuse a removal."""
    start = git(repo, "rev-parse", "HEAD")
    skipped = commit(repo, {}, "still before the ratchet")
    adoption = adopt(repo)
    clean = commit(repo, {}, "an ordinary change")
    removed = commit(repo, {BASELINE_PATH: None}, "remove the baseline")
    report = walk(repo, start, removed, "main")
    statuses = [(step.commit, step.status) for step in report.steps]
    assert statuses == [
        (skipped, "skipped-before-adoption"),
        (adoption, "adoption"),
        (clean, "checked"),
        (removed, "checked"),
    ]
    assert report.refusals == (
        f"{removed[:12]}: A ratchet file was removed or emptied: {BASELINE_PATH}",
    )


def test_a_raise_in_a_commit_before_the_tip_is_refused(repo: Path) -> None:
    """Refuse a raised baseline even when a later commit is the pushed tip."""
    start = adopt(repo)
    raise_commit = commit(repo, {BASELINE_PATH: raised(small_baseline())}, "raise")
    tip = commit(repo, {}, "the tip")
    report = walk(repo, start, tip, "main")
    refused = [step.commit for step in report.steps if step.refusals]
    assert refused == [raise_commit]
    assert all(line.startswith(raise_commit[:12]) for line in report.refusals)


def test_delete_then_raise_is_refused_at_the_delete_and_after(repo: Path) -> None:
    """A removed baseline cannot make the next commit look like an adoption."""
    start = adopt(repo)
    deleted = commit(repo, {BASELINE_PATH: None}, "delete")
    restored = commit(repo, {BASELINE_PATH: raised(small_baseline())}, "raise")
    report = walk(repo, start, restored, "main")
    steps = {step.commit: step for step in report.steps}
    assert steps[deleted].refusals == (
        f"A ratchet file was removed or emptied: {BASELINE_PATH}",
    )
    assert steps[restored].status == "checked"
    assert steps[restored].refusals == (
        "A ratchet file does not qualify: missing 'tools/quality_ratchet_baseline.json'",
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({BASELINE_PATH: b"   \n"}, "removed or emptied"),
        ({BASELINE_PATH: b"{"}, "does not qualify"),
        ({POLICY_PATH: POLICY + b"# changed\n"}, "policy changed"),
    ],
)
def test_an_emptied_corrupted_or_changed_file_is_refused(
    repo: Path, change: dict[str, object | bytes | None], message: str
) -> None:
    """Refuse an empty or unparseable baseline and any policy edit."""
    start = adopt(repo)
    bad = commit(repo, change, "fault")
    step = check_step(repo, start, bad)
    assert len(step.refusals) == 1 and message in step.refusals[0]


def test_an_adoption_outside_the_policy_scope_is_refused(repo: Path) -> None:
    """Check an adoption against its policy's origin, never skip it."""
    data = small_baseline()
    data["origin_commit"] = "1" * 40
    adoption = commit(repo, {POLICY_PATH: POLICY, BASELINE_PATH: data}, "adopt")
    step = check_step(repo, first_parent(repo, adoption), adoption)
    assert step.status == "adoption"
    assert step.refusals == (
        "A ratchet file does not qualify: Required baseline authority or profile set changed",
    )


def transitions_file(*entries: dict[str, object]) -> dict[str, object]:
    """Return a transitions envelope.

    Parameters
    ----------
    *entries : dict[str, object]
        Its entries.

    Returns
    -------
    dict[str, object]
        The envelope.
    """
    return {"schema": TRANSITIONS_SCHEMA, "transitions": list(entries)}


def test_transitions_are_append_only_and_never_removed(repo: Path) -> None:
    """Accept an appended file; refuse a rewritten or removed one."""
    start = adopt(repo)
    added = commit(repo, {TRANSITIONS_PATH: transitions_file(entry())}, "add entry")
    assert check_step(repo, start, added).refusals == ()
    changed = entry()
    changed["reason"] = "engine_fix"
    rewritten = commit(repo, {TRANSITIONS_PATH: transitions_file(changed)}, "rewrite")
    assert check_step(repo, added, rewritten).refusals == (
        "A ratchet file does not qualify: Native quality transitions are append-only",
    )
    removed = commit(repo, {TRANSITIONS_PATH: None}, "remove")
    assert check_step(repo, rewritten, removed).refusals == (
        f"A ratchet file was removed or emptied: {TRANSITIONS_PATH}",
    )


def test_the_first_parent_is_used_for_a_merge(repo: Path) -> None:
    """Compare a merge with its first parent, not with the merged branch."""
    adopt(repo)
    larger = commit(repo, {BASELINE_PATH: raised(small_baseline())}, "larger, main")
    git(repo, "checkout", "-q", "-b", "side", "HEAD~1")
    commit(repo, {}, "side branch")
    git(repo, "checkout", "-q", "main")
    git(
        repo,
        "-c",
        "user.name=Walk fixture",
        "-c",
        "user.email=walk@example.test",
        "merge",
        "-q",
        "--no-ff",
        "-s",
        "ours",
        "side",
        "-m",
        "merge",
    )
    merge = git(repo, "rev-parse", "HEAD")
    assert first_parent(repo, merge) == larger
    assert check_step(repo, first_parent(repo, merge), merge).refusals == ()


def test_the_range_starts_and_bounds_as_ruled(repo: Path) -> None:
    """Start a new ref at the merge base, refuse a rewrite and a long push."""
    base = git(repo, "rev-parse", "HEAD")
    first = adopt(repo)
    second = commit(repo, {}, "second")
    assert push_range(repo, base, second, "main") == (base, (first, second))
    assert push_range(repo, first, second, "main", max_commits=1) == (first, (second,))
    git(repo, "branch", "default", base)
    assert push_range(repo, ZERO, second, "default")[0] == base
    with pytest.raises(ValueError, match="more than 1; run python -m"):
        push_range(repo, base, second, "main", max_commits=1)
    git(repo, "checkout", "-q", "--orphan", "other")
    unrelated = commit(repo, {}, "unrelated")
    with pytest.raises(ValueError, match="rewrites history"):
        push_range(repo, second, unrelated, "main")
    with pytest.raises(ValueError, match="no merge base"):
        push_range(repo, ZERO, unrelated, "main")
    assert first_parent(repo, unrelated) is None
    assert MAX_COMMITS == 100


def test_a_root_commit_with_ratchet_files_is_an_adoption(tmp_path: Path) -> None:
    """Treat a root commit's files as an adoption with no parent."""
    git(tmp_path, "init", "-q", "-b", "main")
    root = adopt(tmp_path)
    assert check_step(tmp_path, None, root).status == "adoption"


def test_added_and_superseded_entries_are_named(repo: Path) -> None:
    """Report the range's entries, and those the tip no longer applies."""
    start = adopt(repo)
    assert added_entries(repo, start, start) == ()
    tip = commit(repo, {TRANSITIONS_PATH: transitions_file(entry())}, "entry")
    entries = added_entries(repo, start, tip)
    assert [item.new_identity for item in entries] == [NEW_IDENTITY]
    assert added_entries(repo, tip, tip) == ()
    assert superseded_entries(entries, {"typescript-docs": NEW_IDENTITY}) == ()
    assert superseded_entries(entries, {"typescript-docs": "9" * 64}) == (
        f"typescript-docs: {OLD_IDENTITY[:12]} -> {NEW_IDENTITY[:12]} is superseded "
        "at the tip; its canary ran only in the lab gate",
    )
    report = walk(repo, start, tip, "main")
    assert report.superseded == (
        f"typescript-docs: {OLD_IDENTITY[:12]} -> {NEW_IDENTITY[:12]} is superseded "
        "at the tip; its canary ran only in the lab gate",
    )


def test_a_tip_without_a_baseline_reports_no_identities(repo: Path) -> None:
    """Walk a range that never adopts the ratchet."""
    start = git(repo, "rev-parse", "HEAD")
    tip = commit(repo, {}, "plain")
    report = walk(repo, start, tip, "main")
    assert report.superseded == () and report.refusals == ()


def test_range_canaries_run_only_for_entries_the_tip_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run each applied entry's canary; skip the others; refuse a bad run."""
    transition = parse_transition(entry())
    args = argparse.Namespace()
    applied = {"typescript-docs": snapshot(NEW_IDENTITY)}
    assert range_canary_refusals(Path("."), args, (transition,), {}) == ()
    other = {"typescript-docs": snapshot("9" * 64)}
    assert range_canary_refusals(Path("."), args, (transition,), other) == ()
    monkeypatch.delitem(CANARY_RUNNERS, "typescript-docs", raising=False)
    assert range_canary_refusals(Path("."), args, (transition,), applied) == (
        "typescript-docs: Profile typescript-docs has no canary runner",
    )
    canary = transition.canary
    good = CanaryRun(
        canary.enabled_rules,
        canary.fixture_digest,
        canary.platform,
        canary.new_engine_digest,
        canary.new_findings,
    )
    bad = CanaryRun(
        good.enabled_rules, good.fixture_digest, good.platform, good.engine_digest, ()
    )
    runs = iter((good, bad))
    monkeypatch.setitem(CANARY_RUNNERS, "typescript-docs", lambda root, a: next(runs))
    assert range_canary_refusals(Path("."), args, (transition,), applied) == ()
    assert range_canary_refusals(Path("."), args, (transition,), applied) == (
        "typescript-docs: The canary findings differ from the recorded findings",
    )


def test_the_workflow_walks_the_push_and_the_fault_is_caught() -> None:
    """The real workflow passes; trusting the pushed commit itself is refused."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert audit_ratchet_workflow(text) == ()
    fault = text.replace('base=$(git rev-parse "$GITHUB_SHA^1")', "base=$GITHUB_SHA")
    fault = fault.replace(
        "TRUSTED_BASE: ${{ github.event.pull_request.base.sha }}",
        "TRUSTED_BASE: ${{ github.event.pull_request.base.sha || github.sha }}",
    )
    assert audit_ratchet_workflow(fault) == (
        "On a push the trusted base is the pushed commit itself",
        "The tip is not checked against its first parent",
    )
    assert audit_ratchet_workflow("") == (
        "The workflow does not walk the pushed commits",
        "The walk does not start at the push's previous commit",
        "A pull request is not walked from its merge base",
        "The tip is not checked against its first parent",
    )
    from_base = text.replace(
        "BEFORE: ${{ github.event.before }}",
        "BEFORE: ${{ github.event.pull_request.base.sha || github.event.before }}",
    )
    assert audit_ratchet_workflow(from_base) == (
        "A pull request is not walked from its merge base",
    )


def test_the_command_line_reports_and_exits_by_outcome(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit zero when clean, one for a refusal, two when the walk cannot run."""
    start = adopt(repo)
    clean = commit(repo, {}, "clean")
    arguments = ["--root", str(repo), "--before", start, "--default-ref", "main"]
    assert main([*arguments, "--to", clean]) == 0
    assert json.loads(capsys.readouterr().out)["refusals"] == []
    bad = commit(repo, {BASELINE_PATH: raised(small_baseline())}, "raise")
    assert main([*arguments, "--to", bad]) == 1
    capsys.readouterr()
    assert main([*arguments, "--to", bad, "--max-commits", "1"]) == 2
    assert "more than 1" in capsys.readouterr().err


def test_the_module_runs_as_a_program(repo: Path) -> None:
    """Run the walk the way the workflow does."""
    start = adopt(repo)
    tip = commit(repo, {BASELINE_PATH: raised(small_baseline())}, "raise")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.quality_ratchet_walk",
            "--root",
            str(repo),
            "--before",
            start,
            "--to",
            tip,
            "--default-ref",
            "main",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["refusals"]


def test_a_failed_parent_lookup_is_refused_never_an_adoption(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Git failure listing a parent stops the walk instead of skipping checks."""
    start = adopt(repo)
    tip = commit(repo, {BASELINE_PATH: raised(small_baseline())}, "raise")
    real = walk_module._git

    def failing(
        root: Path, *arguments: str, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        """Fail only the parent listing, the way Git would."""
        if "--parents" not in arguments:
            return real(root, *arguments, check=check)
        if check:
            raise subprocess.CalledProcessError(128, ["git", *arguments])
        return subprocess.CompletedProcess(["git", *arguments], 128, "", "fatal")

    monkeypatch.setattr(walk_module, "_git", failing)
    with pytest.raises(subprocess.CalledProcessError):
        walk(repo, start, tip, "main")
    arguments = ["--root", str(repo), "--before", start, "--to", tip]
    assert main([*arguments, "--default-ref", "main"]) == 2
    assert "Native quality walk refused" in capsys.readouterr().err


def pull_request(repo: Path, *, raise_first: bool) -> tuple[str, str, str, str]:
    """Build a pull request cut before the base branch's current head.

    Parameters
    ----------
    repo : Path
        Test repository.
    raise_first : bool
        Whether the pull request's first commit raises the baseline.

    Returns
    -------
    tuple[str, str, str, str]
        The branch point, the base's head, the first and the head commit.
    """
    point = adopt(repo)
    base = commit(repo, {}, "the base branch moves on")
    git(repo, "checkout", "-q", "-b", "feature", point)
    change = raised(small_baseline()) if raise_first else small_baseline()
    first = commit(repo, {BASELINE_PATH: change}, "first pull request commit")
    head = commit(repo, {}, "pull request head")
    return point, base, first, head


def test_a_pull_request_is_walked_from_its_merge_base(repo: Path) -> None:
    """A branch cut behind the base's head is judged commit by commit."""
    point, base, first, head = pull_request(repo, raise_first=False)
    with pytest.raises(ValueError, match="rewrites history"):
        push_range(repo, base, head, "main")
    report = walk(repo, None, head, "main", pr_base=base)
    assert report.start == point
    assert [step.commit for step in report.steps] == [first, head]
    assert report.refusals == ()
    arguments = ["--root", str(repo), "--pr-base", base, "--to", head]
    assert main(arguments) == 0


def test_a_pull_request_raise_before_its_head_fails(repo: Path) -> None:
    """A raise in a pull request's non-head commit is refused."""
    _, base, first, head = pull_request(repo, raise_first=True)
    report = walk(repo, None, head, "main", pr_base=base)
    assert [step.commit for step in report.steps if step.refusals] == [first]


def test_a_pull_request_range_is_bounded_and_needs_a_merge_base(repo: Path) -> None:
    """Name the pull-request command above the bound; refuse unrelated history."""
    _, base, _, head = pull_request(repo, raise_first=False)
    with pytest.raises(ValueError, match="--pr-base"):
        pr_range(repo, base, head, max_commits=1)
    git(repo, "checkout", "-q", "--orphan", "unrelated")
    unrelated = commit(repo, {}, "unrelated")
    with pytest.raises(ValueError, match="no merge base with its base"):
        pr_range(repo, base, unrelated)


def test_a_walk_needs_exactly_one_origin(repo: Path) -> None:
    """Refuse a walk with both origins or with neither."""
    start = adopt(repo)
    with pytest.raises(ValueError, match="exactly one"):
        walk(repo, None, start, "main")
    with pytest.raises(ValueError, match="exactly one"):
        walk(repo, start, start, "main", pr_base=start)
