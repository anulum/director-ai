# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the configuration change checks on real Git histories
"""Run the per-commit walk and the tip check over configuration commits."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import tools.quality_ratchet_configuration as configuration_module
import tools.quality_ratchet_succession_load as load_module
import tools.quality_ratchet_walk_tip as tip_module
from tests.quality_ratchet_configuration_support import (
    AFTER,
    BEFORE,
    LATER,
    PROFILES,
    configuration_entry,
    pyproject,
    sha,
    with_field,
)
from tests.quality_ratchet_succession_support import file_bytes
from tools.quality_ratchet_baseline import BASELINE_PATH
from tools.quality_ratchet_comparison import compare_snapshots
from tools.quality_ratchet_configuration import EXCLUDE_POINTER, PYPROJECT
from tools.quality_ratchet_configuration_check import (
    changed_paths,
    tip_configuration_refusals,
)
from tools.quality_ratchet_contracts import Snapshot
from tools.quality_ratchet_gate import POLICY_PATH
from tools.quality_ratchet_native import documentation_identity
from tools.quality_ratchet_succession import (
    TRANSITIONS_PATH,
    TRANSITIONS_SCHEMA,
    Transition,
)
from tools.quality_ratchet_succession_check import verify_canary_run
from tools.quality_ratchet_succession_load import canary_refusals
from tools.quality_ratchet_succession_parse import (
    parse_transition,
    parse_transitions,
)
from tools.quality_ratchet_walk import check_step, walk
from tools.quality_ratchet_walk_tip import range_canary_refusals

ROOT = Path(__file__).resolve().parents[1]
POLICY = (ROOT / POLICY_PATH).read_bytes()
RUFF_AFTER = pyproject(("director_ai.experimental*",), ruff=120)
PACKAGING = {
    "MANIFEST.in": b"prune src/director_ai/training_adapters\n",
    "packages/paid_tier_manifest.json": b'{"excluded": ["training_adapters"]}\n',
    "packages/director-ai-lite/pyproject.toml": b'[project]\nname = "lite"\n',
    "src/director_ai/training_adapters/README.md": b"Excluded from the wheel.\n",
}


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
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit(root: Path, files: dict[str, object | bytes | None], message: str) -> str:
    """Write, remove or keep exactly the given files and commit the result.

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
    write(root, files)
    git(root, "add", "-A")
    git(
        root,
        "-c",
        "user.name=Configuration fixture",
        "-c",
        "user.email=configuration@example.test",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        message,
    )
    return git(root, "rev-parse", "HEAD")


def write(root: Path, files: dict[str, object | bytes | None]) -> None:
    """Write or remove files in the working tree without committing.

    Parameters
    ----------
    root : Path
        Test repository.
    files : dict[str, object | bytes | None]
        Path to JSON data, raw bytes, or None to remove it.
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


def baseline(entries: list[dict[str, object]], side: str) -> dict[str, Any]:
    """Return the committed baseline with the entries' identities on one side.

    Parameters
    ----------
    entries : list[dict[str, object]]
        Configuration entries.
    side : str
        ``old`` or ``new``.

    Returns
    -------
    dict[str, Any]
        A valid baseline carrier with at most three findings per profile.
    """
    data: dict[str, Any] = json.loads((ROOT / BASELINE_PATH).read_text("utf-8"))
    for profile in data["profiles"].values():
        profile["findings"] = profile["findings"][:3]
    for item in entries:
        data["profiles"][item["profile"]]["identity"] = item[f"{side}_identity"]
    return data


def envelope(*entries: dict[str, object]) -> dict[str, object]:
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


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create a repository that adopted the ratchet over the base project.

    Parameters
    ----------
    tmp_path : Path
        Isolated directory.

    Returns
    -------
    Path
        The repository root; HEAD is the adoption commit.
    """
    git(tmp_path, "init", "-q", "-b", "main")
    commit(tmp_path, {"README.md": b"before the ratchet\n"}, "before")
    old = [configuration_entry(p) for p in PROFILES]
    commit(
        tmp_path,
        {
            POLICY_PATH: POLICY,
            BASELINE_PATH: baseline(old, "old"),
            PYPROJECT: BEFORE,
            "src/director_ai/core.py": b'"""Core."""\n',
        },
        "adopt",
    )
    return tmp_path


def configure(
    root: Path,
    after: bytes = AFTER,
    *,
    entries: list[dict[str, object]] | None = None,
    extra: dict[str, object | bytes | None] | None = None,
    profiles: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Commit a configuration change with its entries and identity lines.

    Parameters
    ----------
    root : Path
        Test repository.
    after : bytes
        The new ``pyproject.toml``.
    entries : list[dict[str, object]] | None
        The entries; by default one per Python profile for this change.
    extra : dict[str, object | bytes | None] | None
        More paths the commit changes.
    profiles : dict[str, Any] | None
        A baseline to commit instead of the entries' new identity lines.

    Returns
    -------
    tuple[str, str]
        The parent and the configuration commit.
    """
    parent = git(root, "rev-parse", "HEAD")
    if entries is None:
        entries = [configuration_entry(p, after=after) for p in PROFILES]
    files: dict[str, object | bytes | None] = {
        PYPROJECT: after,
        TRANSITIONS_PATH: envelope(*entries),
        BASELINE_PATH: profiles if profiles is not None else baseline(entries, "new"),
    }
    files.update(extra or {})
    return parent, commit(root, files, "configure")


def test_the_walk_accepts_a_pure_configuration_commit(repo: Path) -> None:
    """Accept the file, the entries, the identity lines and the packaging paths."""
    parent, change = configure(repo, extra=dict(PACKAGING))
    assert check_step(repo, parent, change).refusals == ()


def test_the_walk_refuses_a_source_change_in_the_configuration_commit(
    repo: Path,
) -> None:
    """Refuse a source edit and a deletion carried by the configuration commit."""
    parent, change = configure(
        repo,
        extra={"src/director_ai/scorer.py": b'"""Scorer."""\n', "README.md": None},
    )
    assert check_step(repo, parent, change).refusals == (
        "A configuration transition carries no source change: README.md",
        "A configuration transition carries no source change: "
        "src/director_ai/scorer.py",
    )


def test_the_walk_refuses_an_effective_change_and_a_wrong_record(repo: Path) -> None:
    """Refuse a Ruff change recorded as nothing, for every profile."""
    entries = [configuration_entry(p, after=RUFF_AFTER, pointers=()) for p in PROFILES]
    parent, change = configure(repo, RUFF_AFTER, entries=entries)
    effective = "Effective configuration changed; needs canaries: pyproject.toml"
    recorded = "Recorded configuration paths differ from the configuration diff"
    assert check_step(repo, parent, change).refusals == tuple(
        sorted(
            [
                "python-docs: Influencing configuration changed: pyproject.toml "
                "/tool/ruff",
                *(f"{p}: {effective} /tool/ruff/line-length" for p in PROFILES),
                *(f"{p}: {recorded}: pyproject.toml" for p in PROFILES),
            ]
        )
    )


def test_the_walk_refuses_a_baseline_change_beyond_identity_lines(
    repo: Path,
) -> None:
    """Refuse a fixed finding of another profile in the configuration commit."""
    entries = [configuration_entry(p) for p in PROFILES]
    data = baseline(entries, "new")
    data["profiles"]["go-docs"]["findings"] = data["profiles"]["go-docs"]["findings"][
        :1
    ]
    parent, change = configure(repo, entries=entries, profiles=data)
    assert check_step(repo, parent, change).refusals == (
        "A configuration transition changes more than identity: go-docs",
    )


def test_the_walk_refuses_an_identity_line_without_its_entry(repo: Path) -> None:
    """Refuse a profile whose identity line moved with no entry of its own."""
    entries = [configuration_entry(p) for p in PROFILES]
    parent, change = configure(
        repo, entries=entries[:1], profiles=baseline(entries, "new")
    )
    refusals = check_step(repo, parent, change).refusals
    assert "A configuration transition changes another identity: python-types" in (
        refusals
    )
    assert "A configuration transition changes another identity: notebook-types" in (
        refusals
    )
    assert "Native quality tool, configuration or scope identity changed" in refusals


def test_the_walk_checks_only_the_commit_that_adds_the_entry(repo: Path) -> None:
    """Leave a later ordinary commit to the ordinary ratchet."""
    start = git(repo, "rev-parse", "HEAD")
    _, change = configure(repo)
    later = commit(repo, {"src/director_ai/scorer.py": b'"""Scorer."""\n'}, "source")
    assert check_step(repo, change, later).refusals == ()
    report = walk(repo, start, later, "main")
    assert report.refusals == ()
    assert [step.status for step in report.steps] == ["checked", "checked"]


def test_a_second_configuration_commit_is_checked_on_its_own(repo: Path) -> None:
    """Check a later entry against its own parent, not the first one's."""
    first = [configuration_entry(p) for p in PROFILES]
    configure(repo, entries=first)
    second = [
        configuration_entry(p, before=AFTER, after=LATER, rigor=None) for p in PROFILES
    ]
    parent, change = configure(repo, LATER, entries=first + second)
    assert check_step(repo, parent, change).refusals == ()
    git(repo, "checkout", "-q", "-b", "mixed", parent)
    extra: dict[str, object | bytes | None] = {
        "src/director_ai/scorer.py": b'"""Scorer."""\n'
    }
    _, mixed = configure(repo, LATER, entries=first + second, extra=extra)
    assert check_step(repo, parent, mixed).refusals == (
        "A configuration transition carries no source change: "
        "src/director_ai/scorer.py",
    )


def test_a_configuration_file_that_is_missing_or_not_toml_is_refused(
    repo: Path,
) -> None:
    """Refuse an entry whose named file is absent at the parent or not TOML."""
    broken = b"[project\n"
    entries = [configuration_entry(p, after=broken, pointers=()) for p in PROFILES]
    parent, change = configure(repo, broken, entries=entries)
    assert check_step(repo, parent, change).refusals == (
        "A ratchet file does not qualify: A configuration file is not valid TOML: "
        "pyproject.toml",
    )


def test_a_configuration_file_absent_at_the_parent_is_refused(tmp_path: Path) -> None:
    """Refuse an entry that names a file the parent does not have."""
    git(tmp_path, "init", "-q", "-b", "main")
    old = [configuration_entry(p) for p in PROFILES]
    commit(
        tmp_path,
        {POLICY_PATH: POLICY, BASELINE_PATH: baseline(old, "old")},
        "adopt without the file",
    )
    parent, change = configure(tmp_path)
    assert check_step(tmp_path, parent, change).refusals == (
        "A ratchet file does not qualify: A configuration transition names a "
        "missing file: pyproject.toml",
    )


def test_changed_paths_lists_each_status_and_splits_renames(repo: Path) -> None:
    """Report added, modified, deleted and renamed paths by their own status."""
    parent = git(repo, "rev-parse", "HEAD")
    git(repo, "mv", "src/director_ai/core.py", "src/director_ai/moved.py")
    change = commit(
        repo,
        {PYPROJECT: AFTER, "README.md": None, "new file.txt": b"x\n"},
        "paths",
    )
    assert changed_paths(repo, parent, change) == {
        "README.md": "D",
        "new file.txt": "A",
        PYPROJECT: "M",
        "src/director_ai/core.py": "D",
        "src/director_ai/moved.py": "A",
    }


def transitions(*entries: dict[str, object]) -> tuple[Transition, ...]:
    """Parse configuration entries.

    Parameters
    ----------
    *entries : dict[str, object]
        Decoded entries.

    Returns
    -------
    tuple[Transition, ...]
        The validated records.
    """
    return tuple(parse_transition(item) for item in entries)


def test_the_tip_recomputes_the_diff_from_the_base_to_the_working_tree(
    repo: Path,
) -> None:
    """Pass a raw-only working tree; refuse an effective one."""
    base = git(repo, "rev-parse", "HEAD")
    applied = transitions(*(configuration_entry(p) for p in PROFILES))
    write(repo, {PYPROJECT: AFTER})
    assert tip_configuration_refusals(repo, base, applied) == ()
    write(repo, {PYPROJECT: pyproject(("director_ai.training_adapters*",), ruff=99)})
    effective = "Effective configuration changed; needs canaries: pyproject.toml"
    recorded = "Recorded configuration paths differ from the configuration diff"
    assert tip_configuration_refusals(repo, base, applied) == tuple(
        sorted(
            [
                "python-docs: Influencing configuration changed: pyproject.toml "
                "/tool/ruff",
                *(f"{p}: {effective} /tool/ruff/line-length" for p in PROFILES),
                *(f"{p}: {recorded}: pyproject.toml" for p in PROFILES),
            ]
        )
    )


def test_the_tip_accepts_a_range_with_other_commits(repo: Path) -> None:
    """Judge only the configuration bytes at the tip, not the range's sources."""
    base = git(repo, "rev-parse", "HEAD")
    configure(repo, extra=dict(PACKAGING))
    commit(repo, {"src/director_ai/scorer.py": b'"""Scorer."""\n'}, "source")
    write(repo, {"src/director_ai/untracked.py": b'"""Untracked."""\n'})
    applied = transitions(*(configuration_entry(p) for p in PROFILES))
    assert tip_configuration_refusals(repo, base, applied) == ()


def test_the_tip_refuses_a_missing_or_invalid_working_file(repo: Path) -> None:
    """Raise for a deleted or unparsable configuration file at the tip."""
    base = git(repo, "rev-parse", "HEAD")
    applied = transitions(configuration_entry())
    write(repo, {PYPROJECT: None})
    with pytest.raises(ValueError, match="names a missing file: pyproject.toml"):
        tip_configuration_refusals(repo, base, applied)
    write(repo, {PYPROJECT: b"[project\n"})
    with pytest.raises(ValueError, match="is not valid TOML: pyproject.toml"):
        tip_configuration_refusals(repo, base, applied)


def snapshot(profile: str, identity: str) -> Snapshot:
    """Return a complete snapshot without findings.

    Parameters
    ----------
    profile : str
        The profile.
    identity : str
        Its identity.

    Returns
    -------
    Snapshot
        The snapshot.
    """
    return Snapshot(profile, identity, {"src/director_ai/core.py": "a" * 64}, (), True)


def no_canary(root: Path, args: argparse.Namespace, profile: str) -> object:
    """Fail the test if any canary runner is called.

    Parameters
    ----------
    root : Path
        Ignored.
    args : argparse.Namespace
        Ignored.
    profile : str
        The profile whose canary would run.

    Raises
    ------
    AssertionError
        Always.
    """
    raise AssertionError(f"a canary ran for {profile}")


def test_the_canary_step_routes_configuration_entries_to_the_tip_diff(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run no canary for a configuration entry; recompute its diff instead."""
    monkeypatch.setattr(load_module, "run_profile_canaries", no_canary)
    base = git(repo, "rev-parse", "HEAD")
    items = [configuration_entry(p) for p in PROFILES]
    applied = transitions(*items)
    trusted = {
        str(i["profile"]): snapshot(str(i["profile"]), str(i["old_identity"]))
        for i in items
    }
    actual = {
        str(i["profile"]): snapshot(str(i["profile"]), str(i["new_identity"]))
        for i in items
    }
    args = argparse.Namespace(base_ref=base)
    write(repo, {PYPROJECT: AFTER})
    assert canary_refusals(repo, args, trusted, actual, applied) == ()
    write(repo, {PYPROJECT: RUFF_AFTER})
    assert (
        "python-docs: Influencing configuration changed: pyproject.toml /tool/ruff"
        in (canary_refusals(repo, args, trusted, actual, applied))
    )
    assert canary_refusals(repo, args, trusted, trusted, applied) == ()
    elsewhere = {p: snapshot(p, "9" * 64) for p in PROFILES}
    assert canary_refusals(repo, args, trusted, elsewhere, applied) == ()


def test_the_range_canaries_skip_configuration_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run no canary at the tip for an applied configuration entry."""
    monkeypatch.setattr(tip_module, "run_profile_canaries", no_canary)
    item = configuration_entry()
    actual = {"python-docs": snapshot("python-docs", str(item["new_identity"]))}
    args = argparse.Namespace()
    assert range_canary_refusals(Path("."), args, transitions(item), actual) == ()


def test_a_configuration_entry_has_no_canary_run_to_verify() -> None:
    """Refuse to verify a canary run against an entry that has none."""
    (transition,) = transitions(configuration_entry())
    assert verify_canary_run(
        transition,
        enabled_rules=(),
        fixture_digest="",
        platform="",
        engine_digest="",
        findings=(),
    ) == ("The transition has no canary run",)


def test_a_packaging_change_that_moves_an_identity_field_is_refused_at_the_tip(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a capture whose identity also binds a packaging file's digest.

    The entry declares only the ``pyproject.toml`` digest. Were a packaging
    file to enter the identity, the recomputed identity would differ from the
    entry's new identity: no entry would apply and the tip refuses the change,
    while an entry that declared the field is refused when parsed.
    """
    monkeypatch.setattr(load_module, "run_profile_canaries", no_canary)
    item = configuration_entry()
    field = "/configurations/packages~1director-ai-lite~1pyproject.toml"
    moved = with_field(
        item, field, sha(PACKAGING["packages/director-ai-lite/pyproject.toml"])
    )
    original = snapshot("python-docs", str(item["old_identity"]))
    current = snapshot("python-docs", str(moved["new_identity"]))
    applied = transitions(item)
    assert compare_snapshots(original, current, applied) == (
        "Native quality tool, configuration or scope identity changed",
    )
    args = argparse.Namespace(base_ref=git(repo, "rev-parse", "HEAD"))
    assert (
        canary_refusals(
            repo, args, {"python-docs": original}, {"python-docs": current}, applied
        )
        == ()
    )
    with pytest.raises(ValueError, match="outside its configuration files"):
        parse_transition(moved)
    honest = snapshot("python-docs", str(item["new_identity"]))
    assert compare_snapshots(original, honest, applied) == ()


def test_the_documentation_identity_binds_no_packaging_file(tmp_path: Path) -> None:
    """Show the real python-docs identity ignores packaging and the placeholder."""
    write(tmp_path, {PYPROJECT: BEFORE, "requirements/ci-lint.txt": b"ruff\n"})
    root = tmp_path.resolve()
    identity = documentation_identity(root)
    write(tmp_path, dict(PACKAGING))
    assert documentation_identity(root) == identity
    write(tmp_path, {PYPROJECT: AFTER})
    assert documentation_identity(root) != identity


def test_a_range_with_two_configuration_commits_is_judged_as_one_diff(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check each commit in the walk, and the range's whole diff at the tip.

    No entry chains: one entry must lead from the base identity to the tip's,
    so two configuration commits of one profile apply no entry at the tip,
    and an entry spanning both leaves an identity another entry leaves. The
    tip diff, the union of both commits' pointers, must equal the record of
    the one entry that applies.
    """
    monkeypatch.setattr(load_module, "run_profile_canaries", no_canary)
    namespaces = "/tool/setuptools/packages/find/namespaces"
    for profile in PROFILES:
        monkeypatch.setitem(
            configuration_module.NON_INFLUENCING,
            profile,
            {PYPROJECT: (EXCLUDE_POINTER, namespaces)},
        )
    both = AFTER.replace(b'where = ["src"]', b'where = ["src"]\nnamespaces = false')
    base = git(repo, "rev-parse", "HEAD")
    first = [configuration_entry(p) for p in PROFILES]
    configure(repo, entries=first)
    second = [
        configuration_entry(p, before=AFTER, after=both, pointers=(namespaces,))
        for p in PROFILES
    ]
    _, tip = configure(repo, both, entries=first + second)
    assert walk(repo, base, tip, "main").refusals == ()
    applied = transitions(*first, *second)
    original = snapshot("python-docs", str(first[0]["old_identity"]))
    current = snapshot("python-docs", str(second[0]["new_identity"]))
    assert compare_snapshots(original, current, applied) == (
        "Native quality tool, configuration or scope identity changed",
    )
    args = argparse.Namespace(base_ref=base)
    trusted, actual = {"python-docs": original}, {"python-docs": current}
    assert canary_refusals(repo, args, trusted, actual, applied) == ()
    spanning = configuration_entry(after=both, pointers=(EXCLUDE_POINTER, namespaces))
    with pytest.raises(ValueError, match="Two transitions leave the same profile"):
        parse_transitions(file_bytes(*first, *second, spanning))
    assert tip_configuration_refusals(repo, base, transitions(spanning)) == ()
    last_only = configuration_entry(after=both, pointers=(namespaces,))
    assert tip_configuration_refusals(repo, base, transitions(last_only)) == (
        "python-docs: Recorded configuration paths differ from the configuration "
        "diff: pyproject.toml",
    )
