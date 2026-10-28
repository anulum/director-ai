# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — diagnostic identity and no-net-trading contract tests
"""Verify lossless diagnostic identity and reject independently meaningful debt swaps."""

from __future__ import annotations

import json

import pytest

from tools.quality_ratchet_comparison import compare_snapshots
from tools.quality_ratchet_contracts import (
    SCHEMA,
    Finding,
    Snapshot,
    decode_quality_json,
    parse_snapshot,
    repository_path,
    snapshot_payload,
)


def original_snapshot() -> Snapshot:
    """Construct parser input with two distinct original notebook diagnostics.

    Returns
    -------
    Snapshot
        Explicit unit-test carrier, without native runtime qualification authority.
    """
    return Snapshot(
        "python-docs",
        "1" * 64,
        {"notebooks/legacy.ipynb": "2" * 64},
        (
            Finding("notebooks/legacy.ipynb", "D103", 1, 1, "Missing documentation", 1),
            Finding("notebooks/legacy.ipynb", "D103", 1, 1, "Missing documentation", 2),
        ),
        True,
    )


def test_original_snapshot_round_trip_preserves_notebook_identity() -> None:
    """Retain individual equal-position findings from different cells through JSON."""
    original = original_snapshot()
    encoded = json.dumps(snapshot_payload(original)).encode()
    current = parse_snapshot(decode_quality_json(encoded))
    assert current == original
    assert compare_snapshots(original, current) == ()


@pytest.mark.parametrize(
    "path",
    (
        "",
        ".",
        "../src/a.py",
        "src/../a.py",
        "/src/a.py",
        "src//a.py",
        "src/./a.py",
        "src\\a.py",
        "src:a.py",
        "src/a\nb.py",
        "src/a\x00b.py",
        "src/a\x7fb.py",
    ),
)
def test_repository_path_rejects_aliases_and_git_protocol_delimiters(path: str) -> None:
    """Refuse ambiguous source identity and injection into original-object lookup.

    Parameters
    ----------
    path : str
        Noncanonical or protocol-breaking candidate source path.
    """
    with pytest.raises(ValueError):
        repository_path(path)


def test_json_duplicate_members_cannot_replace_the_source_inventory() -> None:
    """Reject duplicate JSON before overwritten values can erase original debt."""
    with pytest.raises(ValueError, match="Repeated"):
        decode_quality_json(b'{"sources": {"src/a.py": "old"}, "sources": {}}')


@pytest.mark.parametrize(
    "mutation",
    (
        "empty",
        "outside",
        "duplicate",
        "boolean",
        "unknown",
        "profile",
        "profile-text",
        "completion",
        "hash",
        "findings-type",
        "finding-members",
        "finding-text",
        "cell",
        "mapping",
        "sources-mapping",
        "advisories-type",
        "advisory-text",
        "advisory-duplicate",
        "column",
    ),
)
def test_snapshot_invalid_or_incomplete_identity_refuses(mutation: str) -> None:
    """Reject empty inventories, unbound findings and ambiguous positions/members.

    Parameters
    ----------
    mutation : str
        Distinct malformed carrier mutation, independent of native compiler output.
    """
    payload = snapshot_payload(original_snapshot())
    findings: list[dict[str, object]] = [
        {
            "path": "notebooks/legacy.ipynb",
            "rule": "D103",
            "line": 1,
            "column": 1,
            "message": "Missing documentation",
            "cell": 1,
        },
    ]
    payload["findings"] = findings
    if mutation == "empty":
        payload["sources"] = {}
    elif mutation == "outside":
        findings[0]["path"] = "notebooks/other.ipynb"
    elif mutation == "duplicate":
        findings.append(dict(findings[0]))
    elif mutation == "boolean":
        findings[0]["line"] = True
    elif mutation == "unknown":
        payload["ignored"] = True
    elif mutation == "profile":
        payload["profile"] = "Python-Docs"
    elif mutation == "profile-text":
        payload["profile"] = "\x00"
    elif mutation == "completion":
        payload["complete"] = 1
    elif mutation == "hash":
        payload["identity"] = "A" * 64
    elif mutation == "findings-type":
        payload["findings"] = {}
    elif mutation == "finding-members":
        del findings[0]["cell"]
    elif mutation == "finding-text":
        findings[0]["message"] = "   "
    elif mutation == "cell":
        findings[0]["cell"] = False
    elif mutation == "mapping":
        payload["findings"] = [None]
    elif mutation == "advisories-type":
        payload["advisories"] = {}
    elif mutation == "advisory-text":
        payload["advisories"] = ["   "]
    elif mutation == "advisory-duplicate":
        payload["advisories"] = ["native information", "native information"]
    elif mutation == "column":
        findings[0]["column"] = False
    else:
        payload["sources"] = []
    with pytest.raises(ValueError):
        parse_snapshot(payload)


def test_deleting_one_finding_cannot_fund_a_different_finding() -> None:
    """Reject a same-count debt swap on byte-identical source."""
    original = original_snapshot()
    current = Snapshot(
        original.profile,
        original.identity,
        dict(original.sources),
        (Finding("notebooks/legacy.ipynb", "D103", 5, 1, "Different finding", 1),),
        True,
    )
    assert len(current.findings) < len(original.findings)
    assert compare_snapshots(original, current) == (
        "Unchanged source gained a native finding: notebooks/legacy.ipynb",
    )


def test_clean_changed_source_and_individual_debt_reduction_pass() -> None:
    """Permit exact historical removals and fully clean changed source bytes."""
    original = original_snapshot()
    retained = Snapshot(
        original.profile,
        original.identity,
        dict(original.sources),
        original.findings[:1],
        True,
    )
    assert compare_snapshots(original, retained) == ()
    clean = Snapshot(
        original.profile,
        original.identity,
        {"notebooks/legacy.ipynb": "3" * 64, "src/new.py": "4" * 64},
        (),
        True,
    )
    assert compare_snapshots(original, clean) == ()


@pytest.mark.parametrize("change", ("incomplete", "profile", "identity", "source"))
def test_incomplete_or_changed_profile_source_cannot_keep_debt(change: str) -> None:
    """Refuse missing native completion, profile drift or residual changed-file debt.

    Parameters
    ----------
    change : str
        Authority or source difference applied to an otherwise original carrier.
    """
    original = original_snapshot()
    current = Snapshot(
        "python-types" if change == "profile" else original.profile,
        "3" * 64 if change == "identity" else original.identity,
        {"notebooks/legacy.ipynb": "3" * 64}
        if change == "source"
        else dict(original.sources),
        original.findings,
        change != "incomplete",
    )
    assert compare_snapshots(original, current)


def test_parser_rejects_a_different_schema_instead_of_silently_migrating() -> None:
    """Require explicitly reconciled schema migration before adopting changed carriers."""
    payload = snapshot_payload(original_snapshot())
    assert payload["schema"] == SCHEMA
    payload["schema"] = "quality-ratchet.snapshot.v1"
    with pytest.raises(ValueError, match="schema"):
        parse_snapshot(payload)


def test_unknown_column_and_native_advisories_round_trip_without_fabrication() -> None:
    """Keep unavailable and known columns distinct and preserve native information."""
    original = Snapshot(
        "python-types",
        "1" * 64,
        {"src/legacy.py": "2" * 64},
        (
            Finding("src/legacy.py", "unused-ignore", 1, None, "Unused ignore"),
            Finding("src/legacy.py", "unused-ignore", 1, 1, "Unused ignore"),
        ),
        True,
        ("mypy-unused-override:unused.*",),
    )
    encoded = json.dumps(snapshot_payload(original)).encode()
    assert parse_snapshot(decode_quality_json(encoded)) == original
    current = Snapshot(
        original.profile,
        original.identity,
        original.sources,
        original.findings[1:],
        True,
        original.advisories,
    )
    assert compare_snapshots(current, original) == (
        "Unchanged source gained a native finding: src/legacy.py",
    )
