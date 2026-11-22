# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tool succession loading, canary and gate wiring tests
"""Verify the append-only file, the canary gate and the baseline comparisons."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from tests.quality_ratchet_succession_support import (
    ENGINE,
    NEW_IDENTITY,
    OLD_IDENTITY,
    RULES,
    entry,
    file_bytes,
    finding,
    snapshot,
)
from tools.quality_ratchet_baseline import Baseline, compare_baselines
from tools.quality_ratchet_canaries import (
    CANARY_ROOT,
    CANARY_RUNNERS,
    CanaryRun,
    fixture_digest,
    run_profile_canaries,
)
from tools.quality_ratchet_succession import TRANSITIONS_PATH, FindingKey
from tools.quality_ratchet_succession_load import (
    canary_refusals,
    load_transitions,
    require_append_only,
)
from tools.quality_ratchet_succession_parse import parse_transition

ARGS = argparse.Namespace()


def commit(root: Path) -> str:
    """Commit every file of a test repository and return the commit.

    Parameters
    ----------
    root : Path
        Test repository.

    Returns
    -------
    str
        The new commit's identifier.
    """
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Quality Test",
            "-c",
            "user.email=quality@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "state",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return head.stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Return an initialised empty Git repository.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary directory.

    Returns
    -------
    Path
        The repository root.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def write_transitions(root: Path, data: bytes) -> None:
    """Write the transition file into a repository.

    Parameters
    ----------
    root : Path
        Repository root.
    data : bytes
        File bytes.
    """
    path = root / TRANSITIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_absent_file_on_both_sides_loads_nothing(repository: Path) -> None:
    """Load no transitions when neither commit has the file."""
    assert load_transitions(repository, commit(repository)) == ()


def test_appended_entries_load_and_rewritten_entries_refuse(repository: Path) -> None:
    """Load an appended file; refuse a removed or rewritten trusted entry."""
    write_transitions(repository, file_bytes())
    base = commit(repository)
    write_transitions(repository, file_bytes(entry()))
    assert len(load_transitions(repository, base)) == 1
    with_entry = commit(repository)
    write_transitions(repository, file_bytes())
    with pytest.raises(ValueError, match="append-only"):
        load_transitions(repository, with_entry)
    (repository / TRANSITIONS_PATH).unlink()
    with pytest.raises(ValueError, match="was removed"):
        load_transitions(repository, with_entry)


def test_transition_path_must_be_a_regular_file(repository: Path) -> None:
    """Refuse a directory in place of the transition file."""
    (repository / TRANSITIONS_PATH).mkdir(parents=True)
    with pytest.raises(ValueError, match="not a regular file"):
        load_transitions(repository, commit(repository))


def test_append_only_refuses_non_list_transitions() -> None:
    """Refuse an envelope whose transitions member is not a list."""
    with pytest.raises(ValueError, match="append-only"):
        require_append_only(b'{"transitions": {}}', b'{"transitions": []}')
    require_append_only(None, None)


def make_fixture(root: Path) -> Path:
    """Create a two-file canary fixture for typescript-docs.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    Path
        The fixture directory.
    """
    directory = root / CANARY_ROOT / "typescript-docs"
    (directory / "nested").mkdir(parents=True)
    (directory / "canary.ts").write_text("export const a = 1;\n", encoding="utf-8")
    (directory / "nested" / "b.ts").write_text("// @ts-ignore\n", encoding="utf-8")
    return directory


def test_fixture_digest_is_stable_and_names_its_files(tmp_path: Path) -> None:
    """Digest a fixture by file name and content; any change alters it."""
    directory = make_fixture(tmp_path)
    first = fixture_digest(tmp_path, "typescript-docs")
    assert first == fixture_digest(tmp_path, "typescript-docs")
    (directory / "canary.ts").write_text("export const b = 1;\n", encoding="utf-8")
    assert fixture_digest(tmp_path, "typescript-docs") != first


def test_fixture_must_exist_and_hold_only_regular_files(tmp_path: Path) -> None:
    """Refuse a missing, linked, empty or non-regular fixture."""
    with pytest.raises(ValueError, match="has no canary fixture"):
        fixture_digest(tmp_path, "typescript-docs")
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / CANARY_ROOT).mkdir(parents=True)
    (tmp_path / CANARY_ROOT / "typescript-docs").symlink_to(target)
    with pytest.raises(ValueError, match="has no canary fixture"):
        fixture_digest(tmp_path, "typescript-docs")
    (tmp_path / CANARY_ROOT / "typescript-docs").unlink()
    empty = tmp_path / CANARY_ROOT / "typescript-docs"
    empty.mkdir()
    with pytest.raises(ValueError, match="not regular files"):
        fixture_digest(tmp_path, "typescript-docs")
    (empty / "real.ts").write_text("x\n", encoding="utf-8")
    (empty / "link.ts").symlink_to(empty / "real.ts")
    with pytest.raises(ValueError, match="not regular files"):
        fixture_digest(tmp_path, "typescript-docs")


def run_for(findings: tuple[FindingKey, ...]) -> CanaryRun:
    """Return a canary run that matches the support record except its findings.

    Parameters
    ----------
    findings : tuple[FindingKey, ...]
        The run's findings.

    Returns
    -------
    CanaryRun
        The run.
    """
    return CanaryRun(RULES, "4" * 64, "linux-x64", ENGINE, findings)


def test_profile_without_a_runner_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a profile that has no canary runner; run a registered one."""
    with pytest.raises(ValueError, match="go-docs has no canary runner"):
        run_profile_canaries(Path("."), ARGS, "go-docs")
    expected = run_for(())
    monkeypatch.setitem(CANARY_RUNNERS, "go-docs", lambda root, args: expected)
    assert run_profile_canaries(Path("."), ARGS, "go-docs") is expected


def test_canary_gate_runs_only_applied_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip unchanged or undeclared identities; refuse a missing or wrong run."""
    transition = parse_transition(entry())
    before = {"typescript-docs": snapshot(OLD_IDENTITY)}
    after = {"typescript-docs": snapshot(NEW_IDENTITY)}
    assert canary_refusals(Path("."), ARGS, before, before, (transition,)) == ()
    assert canary_refusals(Path("."), ARGS, {}, after, (transition,)) == ()
    other = {"typescript-docs": snapshot("9" * 64)}
    assert canary_refusals(Path("."), ARGS, before, other, (transition,)) == ()
    monkeypatch.delitem(CANARY_RUNNERS, "typescript-docs")
    assert canary_refusals(Path("."), ARGS, before, after, (transition,)) == (
        "typescript-docs: Profile typescript-docs has no canary runner",
    )
    runs = iter((run_for(transition.canary.new_findings), run_for(())))
    monkeypatch.setitem(
        CANARY_RUNNERS, "typescript-docs", lambda root, args: next(runs)
    )
    assert canary_refusals(Path("."), ARGS, before, after, (transition,)) == ()
    assert canary_refusals(Path("."), ARGS, before, after, (transition,)) == (
        "typescript-docs: The canary findings differ from the recorded findings",
    )


def test_baseline_comparison_applies_a_declared_transition() -> None:
    """Pass a baseline whose profile changed under its record; refuse without it."""
    transition = parse_transition(entry())
    original = Baseline(
        "0" * 40, {"typescript-docs": snapshot(OLD_IDENTITY, finding())}
    )
    candidate = Baseline(
        "0" * 40, {"typescript-docs": snapshot(NEW_IDENTITY, finding())}
    )
    assert compare_baselines(original, candidate, (transition,)) == ()
    assert compare_baselines(original, candidate) == (
        "Native quality tool, configuration or scope identity changed",
    )
