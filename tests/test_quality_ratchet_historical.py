# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — reviewed identity changes made before the succession mechanism
"""Walk real histories with a reviewed pre-mechanism identity change."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_quality_ratchet_walk import (
    adopt,
    commit,
    git,
    raised,
    small_baseline,
)
from tests.test_quality_ratchet_walk import repo as repo
from tools.quality_ratchet_baseline import BASELINE_PATH
from tools.quality_ratchet_historical import (
    HISTORICAL_PATH,
    HISTORICAL_SCHEMA,
    MECHANISM_PATH,
    mechanism_commit,
    parse_historical,
    require_entries_append_only,
)
from tools.quality_ratchet_walk import check_step, main, walk

DECISION = b"the reviewed decision record\n"
DECISION_SHA = hashlib.sha256(DECISION).hexdigest()
NEW = "a" * 64


def changed(
    data: dict[str, Any], *profiles: str, identity: str = NEW
) -> dict[str, Any]:
    """Return a baseline whose named profiles carry a new identity.

    Parameters
    ----------
    data : dict[str, Any]
        A baseline carrier.
    *profiles : str
        The profiles to change.
    identity : str
        The new identity.

    Returns
    -------
    dict[str, Any]
        The changed carrier.
    """
    value = copy.deepcopy(data)
    for name in profiles:
        value["profiles"][name]["identity"] = identity
    return value


def record(*entries: dict[str, str]) -> dict[str, object]:
    """Return a historical file.

    Parameters
    ----------
    *entries : dict[str, str]
        Its entries.

    Returns
    -------
    dict[str, object]
        The envelope.
    """
    return {"schema": HISTORICAL_SCHEMA, "entries": list(entries)}


def item(commit_id: str, old: str, *, profile: str = "go-docs") -> dict[str, str]:
    """Return one historical entry.

    Parameters
    ----------
    commit_id : str
        The changing commit.
    old : str
        The parent's identity.
    profile : str
        The profile.

    Returns
    -------
    dict[str, str]
        The entry.
    """
    return {
        "commit": commit_id,
        "profile": profile,
        "old_identity": old,
        "new_identity": NEW,
        "decision_sha256": DECISION_SHA,
    }


def history(repo: Path, baseline: dict[str, Any]) -> tuple[str, str, str]:
    """Adopt, change an identity, then land the mechanism.

    Parameters
    ----------
    repo : Path
        Test repository.
    baseline : dict[str, Any]
        The changing commit's baseline.

    Returns
    -------
    tuple[str, str, str]
        The walk's start, the changing commit and the mechanism's commit.
    """
    start = git(repo, "rev-parse", "HEAD")
    adopt(repo)
    change = commit(repo, {BASELINE_PATH: baseline}, "refresh an identity")
    landing = commit(repo, {MECHANISM_PATH: b"mechanism\n"}, "land the mechanism")
    return start, change, landing


OLD = small_baseline()["profiles"]["go-docs"]["identity"]


def test_a_reviewed_change_passes_and_without_its_record_is_refused(
    repo: Path,
) -> None:
    """Honour the record; without it, the identity change is refused."""
    start, change, landing = history(repo, changed(small_baseline(), "go-docs"))
    bare = walk(repo, start, landing, "main")
    assert bare.refusals == (
        f"{change[:12]}: Native quality tool, configuration or scope identity changed",
    )
    tip = commit(repo, {HISTORICAL_PATH: record(item(change, OLD))}, "record it")
    assert mechanism_commit(repo, tip) == landing
    assert walk(repo, start, tip, "main").refusals == ()


def test_a_record_for_a_commit_after_the_mechanism_is_refused(repo: Path) -> None:
    """An entry naming the mechanism's commit or a later one is never honoured."""
    start, _, landing = history(repo, small_baseline())
    later = commit(
        repo, {BASELINE_PATH: changed(small_baseline(), "go-docs")}, "a later change"
    )
    tip = commit(
        repo,
        {
            HISTORICAL_PATH: record(
                item(later, OLD), item(landing, OLD, profile="lean-docs")
            )
        },
        "record it",
    )
    refusals = walk(repo, start, tip, "main").refusals
    assert (
        f"historical: Historical identity record for {later[:12]} go-docs does not precede the succession mechanism"
        in refusals
    )
    assert (
        f"historical: Historical identity record for {landing[:12]} lean-docs does not precede the succession mechanism"
        in refusals
    )
    assert (
        f"{later[:12]}: Native quality tool, configuration or scope identity changed"
        in refusals
    )


def test_a_record_with_another_old_identity_is_refused(repo: Path) -> None:
    """Refuse an entry whose old identity is not the parent's."""
    start, change, _ = history(repo, changed(small_baseline(), "go-docs"))
    tip = commit(repo, {HISTORICAL_PATH: record(item(change, "9" * 64))}, "record")
    refusals = walk(repo, start, tip, "main").refusals
    assert (
        f"{change[:12]}: Historical identity record for go-docs does not match this commit"
        in refusals
    )
    assert (
        f"{change[:12]}: Native quality tool, configuration or scope identity changed"
        in refusals
    )


def test_a_record_cannot_hide_a_finding_change(repo: Path) -> None:
    """Every finding check still runs at the honoured commit."""
    start, change, _ = history(repo, changed(raised(small_baseline()), "go-docs"))
    tip = commit(repo, {HISTORICAL_PATH: record(item(change, OLD))}, "record")
    refusals = walk(repo, start, tip, "main").refusals
    assert len(refusals) == 1 and refusals[0].startswith(change[:12])
    assert "gained a native finding" in refusals[0]


def test_a_record_maps_only_its_own_profile(repo: Path) -> None:
    """Another profile's identity change at the same commit is still refused."""
    start, change, _ = history(repo, changed(small_baseline(), "go-docs", "lean-docs"))
    tip = commit(repo, {HISTORICAL_PATH: record(item(change, OLD))}, "record")
    assert walk(repo, start, tip, "main").refusals == (
        f"{change[:12]}: Native quality tool, configuration or scope identity changed",
    )


def test_the_decision_digest_must_name_a_record_file(
    repo: Path, tmp_path: Path
) -> None:
    """Refuse a digest that names no record; accept one that does."""
    start, change, _ = history(repo, changed(small_baseline(), "go-docs"))
    tip = commit(repo, {HISTORICAL_PATH: record(item(change, OLD))}, "record")
    records = tmp_path / "records"
    (records / "nested").mkdir(parents=True)
    (records / "nested" / "other.md").write_bytes(b"another record\n")
    assert walk(repo, start, tip, "main", decisions_root=records).refusals == (
        f"historical: Historical identity record for {change[:12]} go-docs names no decision record",
    )
    (records / "nested" / "decision.md").write_bytes(DECISION)
    assert walk(repo, start, tip, "main", decisions_root=records).refusals == ()


def test_no_mechanism_in_history_honours_nothing(repo: Path) -> None:
    """Without a mechanism commit, every entry is refused."""
    start = git(repo, "rev-parse", "HEAD")
    adopt(repo)
    change = commit(repo, {BASELINE_PATH: changed(small_baseline(), "go-docs")}, "c")
    tip = commit(repo, {HISTORICAL_PATH: record(item(change, OLD))}, "record")
    assert mechanism_commit(repo, tip) is None
    assert (
        f"historical: Historical identity record for {change[:12]} go-docs does not precede the succession mechanism"
        in walk(repo, start, tip, "main").refusals
    )


def test_the_record_is_append_only_and_never_removed(repo: Path) -> None:
    """Refuse a rewritten or a removed historical file at its step."""
    _, change, _ = history(repo, changed(small_baseline(), "go-docs"))
    first = commit(repo, {HISTORICAL_PATH: record(item(change, OLD))}, "record")
    appended = commit(
        repo,
        {
            HISTORICAL_PATH: record(
                item(change, OLD), item(change, OLD, profile="rust-docs")
            )
        },
        "append",
    )
    assert check_step(repo, first, appended).refusals == ()
    rewritten = commit(
        repo, {HISTORICAL_PATH: record(item(change, OLD, profile="lean-docs"))}, "rw"
    )
    assert check_step(repo, appended, rewritten).refusals == (
        "A ratchet file does not qualify: Historical identity records are append-only",
    )
    removed = commit(repo, {HISTORICAL_PATH: None}, "remove")
    assert check_step(repo, rewritten, removed).refusals == (
        f"A ratchet file was removed or emptied: {HISTORICAL_PATH}",
    )
    require_entries_append_only(None, None)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"schema": "other", "entries": []}, "envelope"),
        ({"schema": HISTORICAL_SCHEMA, "entries": {}}, "envelope"),
        ({"schema": HISTORICAL_SCHEMA, "entries": [], "extra": 1}, "envelope"),
        (record({**item("1" * 40, OLD), "extra": "x"}), "entry"),
        (record(item("1" * 40, OLD, profile="unknown")), "entry"),
        (record({**item("1" * 40, OLD), "new_identity": OLD}), "changes nothing"),
        (record(item("1" * 40, OLD), item("1" * 40, OLD)), "name one change"),
    ],
)
def test_a_malformed_record_is_refused(value: dict[str, object], message: str) -> None:
    """Refuse every malformed envelope or entry by name."""
    with pytest.raises(ValueError, match=message):
        parse_historical(json.dumps(value).encode())


def test_a_malformed_record_refuses_the_walk_and_its_step(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed tip file stops the walk; at a step it does not qualify."""
    start = adopt(repo)
    bad = commit(repo, {HISTORICAL_PATH: {"schema": "other", "entries": []}}, "bad")
    assert check_step(repo, start, bad).refusals == (
        "A ratchet file does not qualify: Invalid historical identity record envelope",
    )
    arguments = [
        "--root",
        str(repo),
        "--before",
        start,
        "--to",
        bad,
        "--default-ref",
        "main",
    ]
    assert main(arguments) == 2
    assert "Invalid historical identity record envelope" in capsys.readouterr().err


def test_the_command_line_checks_decision_digests(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pass the decision records on the command line."""
    start, change, _ = history(repo, changed(small_baseline(), "go-docs"))
    tip = commit(repo, {HISTORICAL_PATH: record(item(change, OLD))}, "record")
    records = tmp_path / "records"
    records.mkdir()
    (records / "decision.md").write_bytes(DECISION)
    arguments = [
        "--root",
        str(repo),
        "--before",
        start,
        "--to",
        tip,
        "--default-ref",
        "main",
    ]
    assert main([*arguments, "--decisions-root", str(records)]) == 0
    assert json.loads(capsys.readouterr().out)["refusals"] == []
