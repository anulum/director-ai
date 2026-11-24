# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — declared tool succession record and comparison tests
"""Verify that a declared tool succession is validated in full and permits no loss."""

from __future__ import annotations

from typing import Any

import pytest

from tests.quality_ratchet_succession_support import (
    CANARY_KEYS,
    ENGINE,
    NEW_IDENTITY,
    NEW_PREIMAGE,
    OLD_IDENTITY,
    OLD_PREIMAGE,
    RULES,
    canary_of,
    entry,
    file_bytes,
    finding,
    mapping,
    members,
    snapshot,
)
from tools.quality_ratchet_comparison import compare_snapshots
from tools.quality_ratchet_contracts import Finding
from tools.quality_ratchet_profiles import REQUIRED_PROFILES
from tools.quality_ratchet_succession import (
    ABSENT,
    PREIMAGE_PROFILES,
    FieldChange,
    FindingKey,
    canonical_leaf,
    pointer_leaf,
    pointer_token,
    preimage_changes,
    preimage_identity,
)
from tools.quality_ratchet_succession_check import (
    compare_transition,
    finding_key,
    transition_for,
    verify_canary_run,
)
from tools.quality_ratchet_succession_parse import (
    parse_finding_key,
    parse_transition,
    parse_transitions,
)


def test_valid_entry_parses_with_every_member() -> None:
    """Parse a complete record, keeping the identities and the canary evidence."""
    transition = parse_transition(entry())
    assert (transition.old_identity, transition.new_identity) == (
        OLD_IDENTITY,
        NEW_IDENTITY,
    )
    assert canary_of(transition).enabled_rules == RULES
    assert canary_of(transition).new_engine_digest == ENGINE
    assert canary_of(transition).platform == "linux-x64"
    assert len(transition.changed_fields) == 4
    assert parse_transitions(file_bytes(entry())) == (transition,)


def test_preimage_identity_is_the_sorted_json_digest() -> None:
    """Hash the preimage independently of member order."""
    reordered = dict(reversed(list(OLD_PREIMAGE.items())))
    assert preimage_identity(reordered) == OLD_IDENTITY


def test_preimage_changes_cover_the_union_of_keys() -> None:
    """Report added, removed and changed leaves by their full dotted path."""
    changes = preimage_changes({"a": {"b": 1}, "c": [1]}, {"a": {"d": 2}, "c": [2]})
    assert changes == (
        FieldChange("/a/b", 1, ABSENT),
        FieldChange("/a/d", ABSENT, 2),
        FieldChange("/c", [1], [2]),
    )
    assert preimage_changes({"a": {}}, {"a": {}}) == ()


@pytest.mark.parametrize(
    ("name", "pointer"),
    (
        (
            "tools/quality_ratchet_typescript.py",
            "/tools~1quality_ratchet_typescript.py",
        ),
        ("x~y", "/x~0y"),
        ("~1", "/~01"),
        ("", "/"),
    ),
)
def test_member_names_are_escaped_as_pointer_tokens(name: str, pointer: str) -> None:
    """Escape any member name unambiguously and read it back by its pointer."""
    (change,) = preimage_changes({name: 1}, {name: 2})
    assert change.path == pointer
    assert pointer_leaf({name: 2}, pointer) == 2
    assert pointer_token(name) == pointer[1:]


def test_pointer_leaf_reports_absent_members() -> None:
    """Return None for a missing member or a path through a non-object."""
    assert pointer_leaf({"a": 1}, "/b") is None
    assert pointer_leaf({"a": 1}, "/a/b") is None
    assert pointer_leaf({"a": {"b": 3}}, "/a/b") == 3


def test_non_text_member_name_is_refused() -> None:
    """Refuse a member name that JSON could not have produced."""
    with pytest.raises(ValueError, match="member name"):
        non_text: dict[Any, object] = {3: 1}
        preimage_changes(non_text, {"a": 1})


def test_real_preimage_shape_with_file_path_keys_parses() -> None:
    """Accept a record whose changed leaf is an engine digest keyed by file path."""
    transition = parse_transition(entry())
    paths = {change.path for change in transition.changed_fields}
    assert "/engine/tools~1quality_ratchet_typescript.py" in paths
    assert "/engine/packages~1vercel-ai~1package-lock.json" not in paths


def test_reconstructed_old_preimage_with_another_hash_is_refused() -> None:
    """Refuse an old preimage that does not hash to the base identity it claims."""
    record = entry()
    record["old_preimage"] = {**OLD_PREIMAGE, "roots": ["packages"]}
    with pytest.raises(ValueError, match="old_preimage does not hash"):
        parse_transition(record)


def test_new_preimage_must_hash_to_the_new_identity() -> None:
    """Refuse a new preimage that does not hash to the declared new identity."""
    record = entry()
    record["new_identity"] = "9" * 64
    with pytest.raises(ValueError, match="new_preimage does not hash"):
        parse_transition(record)


def test_empty_preimage_is_refused() -> None:
    """Refuse an empty preimage even before hashing it."""
    record = entry()
    record["old_preimage"] = {}
    with pytest.raises(ValueError, match="old_preimage is empty"):
        parse_transition(record)


def test_undeclared_added_key_is_refused() -> None:
    """Refuse a record that omits a leaf the new preimage adds."""
    record = entry()
    record["changed_fields"] = [
        item
        for item in members(record, "changed_fields")
        if item["path"] != "/artefacts/engine/linux-x64/tarball"
    ]
    with pytest.raises(ValueError, match="differ from the preimage diff"):
        parse_transition(record)


def test_declared_fields_are_compared_in_path_order() -> None:
    """Accept the declared leaves in any order."""
    record = entry()
    record["changed_fields"] = list(reversed(members(record, "changed_fields")))
    assert len(parse_transition(record).changed_fields) == 4


@pytest.mark.parametrize(
    ("member", "value", "message"),
    (
        ("changed_fields", [], "nonempty list"),
        ("changed_fields", {}, "nonempty list"),
        ("changed_fields", [{"path": 1, "old": 1, "new": 2}], "changed field"),
        ("changed_fields", [{"path": "a"}], "changed field"),
        ("profile", "unknown-docs", "does not hash an identity preimage"),
        ("profile", 1, "does not hash an identity preimage"),
        ("reason", "convenience", "Unknown transition reason"),
        ("reason", None, "Unknown transition reason"),
        ("message_only", {}, "must be a list"),
        ("drops", {}, "must be a list"),
    ),
)
def test_malformed_members_are_refused(
    member: str, value: object, message: str
) -> None:
    """Refuse each malformed member by name."""
    record = entry()
    record[member] = value
    with pytest.raises(ValueError, match=message):
        parse_transition(record)


def test_unknown_or_missing_entry_member_is_refused() -> None:
    """Refuse an entry whose member set is not exact."""
    record = entry()
    record["extra"] = 1
    with pytest.raises(ValueError, match="entry members"):
        parse_transition(record)


def test_a_transition_must_change_the_identity() -> None:
    """Refuse a record whose two identities are equal."""
    record = entry()
    record["new_identity"] = OLD_IDENTITY
    with pytest.raises(ValueError, match="must change the identity"):
        parse_transition(record)


def test_changed_artefact_needs_its_verification_record() -> None:
    """Refuse a changed artefact leaf with no signature-verification record."""
    record = entry()
    record["verification_records"] = {"/artefacts/engine/linux-x64/binary": "1" * 64}
    with pytest.raises(ValueError, match="verification record"):
        parse_transition(record)


def test_verification_record_for_an_unchanged_field_is_refused() -> None:
    """Refuse a verification record for a leaf that did not change."""
    record = entry()
    records = mapping(record, "verification_records")
    records["/native/node"] = "1" * 64
    record["verification_records"] = records
    with pytest.raises(ValueError, match="verification record"):
        parse_transition(record)


@pytest.mark.parametrize(
    "approvals",
    (("6" * 64, "6" * 64, "8" * 64), ("6" * 64, "7" * 64, "7" * 64)),
)
def test_packet_and_two_approvals_must_be_distinct(approvals: tuple[str, ...]) -> None:
    """Refuse an entry with a single approval or an approval equal to its packet."""
    record = entry()
    names = ("packet_digest", "authority_approval_digest", "rigor_approval_digest")
    record.update(dict(zip(names, approvals, strict=True)))
    with pytest.raises(ValueError, match="distinct records"):
        parse_transition(record)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"extra": 1}, "canary members"),
        ({"platform": ""}, "canary platform"),
        ({"platform": "linux/x64"}, "canary platform"),
        ({"platform": "linux~x64"}, "canary platform"),
        ({"platform": 3}, "canary platform"),
        ({"new_engine_digest": "5" * 64}, "new canary run's engine"),
        ({"enabled_rules": []}, "unique enabled rules"),
        ({"enabled_rules": [RULES[0], RULES[0]]}, "unique enabled rules"),
        ({"enabled_rules": [""]}, "unique enabled rules"),
        ({"enabled_rules": "rules"}, "unique enabled rules"),
        ({"old_findings": CANARY_KEYS[:1]}, "differ between the identities"),
        (
            {"old_findings": CANARY_KEYS[:1], "new_findings": CANARY_KEYS[:1]},
            "exactly the enabled rules",
        ),
    ),
)
def test_canary_evidence_is_refused_when_incomplete(
    change: dict[str, object], message: str
) -> None:
    """Refuse canary evidence that cannot show every enabled rule still fires."""
    record = entry()
    canary = mapping(record, "canary")
    canary.update(change)
    record["canary"] = canary
    with pytest.raises(ValueError, match=message):
        parse_transition(record)


def test_old_engine_must_match_a_preimage_that_records_it() -> None:
    """Refuse an old canary engine digest that its own preimage contradicts."""
    record = entry()
    record["old_preimage"] = NEW_PREIMAGE
    record["old_identity"] = NEW_IDENTITY
    newer = {**NEW_PREIMAGE, "native": {"node": "26.13.2", "typescript": "6.0.3"}}
    record["new_preimage"] = newer
    record["new_identity"] = preimage_identity(newer)
    record["changed_fields"] = [
        {"path": "/native/node", "old": "26.13.1", "new": "26.13.2"}
    ]
    record["verification_records"] = {}
    with pytest.raises(ValueError, match="old canary run's engine"):
        parse_transition(record)
    canary = mapping(record, "canary")
    canary["old_engine_digest"] = ENGINE
    record["canary"] = canary
    assert canary_of(parse_transition(record)).old_engine_digest == ENGINE


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ({"path": "a.ts", "rule": "r", "line": 1, "column": None}, "key members"),
        ({"path": "a.ts", "rule": "", "line": 1, "column": None, "cell": None}, "rule"),
        (
            {"path": "a.ts", "rule": "r", "line": 0, "column": None, "cell": None},
            "position",
        ),
        (
            {"path": "a.ts", "rule": "r", "line": True, "column": None, "cell": None},
            "position",
        ),
        (
            {"path": "a.ts", "rule": "r", "line": None, "column": None, "cell": None},
            "position",
        ),
        (
            {"path": "a.ts", "rule": "r", "line": 1, "column": 0, "cell": None},
            "position",
        ),
    ),
)
def test_finding_keys_are_validated(value: dict[str, object], message: str) -> None:
    """Refuse a malformed message-free finding key."""
    with pytest.raises(ValueError, match=message):
        parse_finding_key(value)


def test_repeated_keys_are_refused() -> None:
    """Refuse a key list that repeats a finding."""
    record = entry()
    record["message_only"] = [CANARY_KEYS[0], CANARY_KEYS[0]]
    with pytest.raises(ValueError, match="repeats a finding"):
        parse_transition(record)


def test_drops_need_a_changelog_and_a_canaried_rule() -> None:
    """Parse a declared drop and refuse drops that lack their evidence."""
    record = entry()
    drop = {**CANARY_KEYS[0], "path": "packages/b.ts", "changelog_digest": "9" * 64}
    record["drops"] = [drop]
    assert parse_transition(record).drops[0].changelog_digest == "9" * 64
    record["drops"] = [drop, drop]
    with pytest.raises(ValueError, match="drops repeats"):
        parse_transition(record)
    record["drops"] = [{**drop, "rule": "other"}]
    with pytest.raises(ValueError, match="has no canary"):
        parse_transition(record)
    record["drops"] = [CANARY_KEYS[0]]
    with pytest.raises(ValueError, match="drop members"):
        parse_transition(record)


def test_transition_file_envelope_is_validated() -> None:
    """Refuse a wrong schema, a non-list and two exits from one identity."""
    with pytest.raises(ValueError, match="transition file"):
        parse_transitions(b'{"schema": "other", "transitions": []}')
    with pytest.raises(ValueError, match="must be a list"):
        parse_transitions(
            b'{"schema": "quality-ratchet.transitions.v1", "transitions": {}}'
        )
    with pytest.raises(ValueError, match="same profile identity"):
        parse_transitions(file_bytes(entry(), entry()))
    assert parse_transitions(file_bytes()) == ()


def test_transition_lookup_matches_profile_and_both_identities() -> None:
    """Find a record only for exactly its profile and identity pair."""
    transition = parse_transition(entry())
    assert transition_for((transition,), "typescript-docs", OLD_IDENTITY, NEW_IDENTITY)
    assert transition_for((transition,), "go-docs", OLD_IDENTITY, NEW_IDENTITY) is None
    assert (
        transition_for((transition,), "typescript-docs", OLD_IDENTITY, "9" * 64) is None
    )


def test_equal_findings_pass_and_undeclared_changes_refuse() -> None:
    """Pass equal findings; refuse a self-consistent record for another identity."""
    transition = parse_transition(entry())
    before = snapshot(OLD_IDENTITY, finding())
    assert (
        compare_snapshots(before, snapshot(NEW_IDENTITY, finding()), (transition,))
        == ()
    )
    assert compare_snapshots(before, snapshot("9" * 64, finding()), (transition,)) == (
        "Native quality tool, configuration or scope identity changed",
    )
    assert compare_snapshots(before, snapshot(NEW_IDENTITY, finding())) == (
        "Native quality tool, configuration or scope identity changed",
    )
    assert compare_transition(before, snapshot("9" * 64), transition) == (
        "The transition does not declare this identity change",
    )


def test_a_stronger_tool_that_adds_a_finding_is_refused() -> None:
    """Refuse a finding the new identity adds on an unchanged source."""
    transition = parse_transition(entry())
    refusals = compare_transition(
        snapshot(OLD_IDENTITY), snapshot(NEW_IDENTITY, finding()), transition
    )
    assert refusals == (
        "Unchanged source gained a native finding under the new identity: packages/b.ts",
    )


def test_a_weaker_tool_that_drops_a_finding_is_refused() -> None:
    """Refuse an undeclared loss and the loss of a rule's last finding."""
    transition = parse_transition(entry())
    refusals = compare_transition(
        snapshot(OLD_IDENTITY, finding()), snapshot(NEW_IDENTITY), transition
    )
    assert refusals == (
        f"Rule {RULES[0]} lost every finding on the unchanged sources",
        "Unchanged source lost an undeclared native finding: packages/b.ts",
    )


def test_declared_drop_passes_only_while_the_rule_keeps_findings() -> None:
    """Admit a declared drop when the rule keeps another finding; refuse a stale one."""
    record = entry()
    record["drops"] = [
        {
            "path": "packages/b.ts",
            "rule": RULES[0],
            "line": 2,
            "column": 1,
            "cell": None,
            "changelog_digest": "9" * 64,
        }
    ]
    transition = parse_transition(record)
    kept = finding(line=7)
    before = snapshot(OLD_IDENTITY, finding(), kept)
    assert compare_transition(before, snapshot(NEW_IDENTITY, kept), transition) == ()
    assert compare_transition(
        before, snapshot(NEW_IDENTITY, finding(), kept), transition
    ) == ("A declared dropped finding is still reported: packages/b.ts",)


def test_message_only_changes_must_be_declared() -> None:
    """Refuse an undeclared rewording and accept a declared one."""
    reworded = Finding("packages/b.ts", RULES[0], 2, 1, "new wording")
    before = snapshot(OLD_IDENTITY, finding())
    after = snapshot(NEW_IDENTITY, reworded)
    assert compare_transition(before, after, parse_transition(entry())) == (
        "Message-only finding changes differ from the declared list",
    )
    record = entry()
    record["message_only"] = [
        {
            "path": "packages/b.ts",
            "rule": RULES[0],
            "line": 2,
            "column": 1,
            "cell": None,
        }
    ]
    assert compare_transition(before, after, parse_transition(record)) == ()


def test_a_rule_id_change_counts_as_a_loss_and_a_gain() -> None:
    """Treat a renamed rule as one removal plus one addition."""
    renamed = finding(rule=RULES[1])
    refusals = compare_transition(
        snapshot(OLD_IDENTITY, finding()),
        snapshot(NEW_IDENTITY, renamed),
        parse_transition(entry()),
    )
    assert (
        "Unchanged source gained a native finding under the new identity: packages/b.ts"
        in refusals
    )
    assert (
        "Unchanged source lost an undeclared native finding: packages/b.ts" in refusals
    )


def test_modified_sources_still_need_zero_findings() -> None:
    """Refuse a finding on a modified source under a transition too."""
    refusals = compare_transition(
        snapshot(OLD_IDENTITY),
        snapshot(NEW_IDENTITY, finding("packages/a.ts"), digest="c" * 64),
        parse_transition(entry()),
    )
    assert refusals == ("New or modified source has native findings: packages/a.ts",)


def test_finding_key_drops_only_the_message() -> None:
    """Key a finding by path, rule, line, column and cell."""
    assert finding_key(finding()) == FindingKey("packages/b.ts", RULES[0], 2, 1, None)


def test_canary_run_must_reproduce_the_record() -> None:
    """Refuse each canary difference; accept an exact reproduction."""
    transition = parse_transition(entry())
    keys = canary_of(transition).new_findings

    def run(
        *,
        enabled_rules: tuple[str, ...] = RULES,
        fixture_digest: str = "4" * 64,
        platform: str = "linux-x64",
        engine_digest: str = ENGINE,
        findings: tuple[FindingKey, ...] = keys,
    ) -> tuple[str, ...]:
        """Check one run that differs from the record in the given members."""
        return verify_canary_run(
            transition,
            enabled_rules=enabled_rules,
            fixture_digest=fixture_digest,
            platform=platform,
            engine_digest=engine_digest,
            findings=findings,
        )

    assert run() == ()
    cases = (
        (run(enabled_rules=RULES[:1]), "enabled rules differ"),
        (run(fixture_digest="0" * 64), "fixture differs"),
        (run(platform="darwin-arm64"), "platform other"),
        (run(engine_digest="0" * 64), "engine other"),
        (run(findings=keys[:1]), "findings differ"),
    )
    for refusals, message in cases:
        assert len(refusals) == 1 and message in refusals[0]


def test_preimage_profiles_are_the_required_profiles() -> None:
    """Admit transitions for exactly the required native profiles."""
    assert frozenset(REQUIRED_PROFILES) == PREIMAGE_PROFILES


def test_undeclared_added_null_leaf_is_refused() -> None:
    """Refuse a new null-valued key that the record does not declare."""
    record = entry()
    newer = {**NEW_PREIMAGE, "extra": None}
    record["new_preimage"] = newer
    record["new_identity"] = preimage_identity(newer)
    with pytest.raises(ValueError, match="differ from the preimage diff"):
        parse_transition(record)
    fields = members(record, "changed_fields")
    record["changed_fields"] = [*fields, {"path": "/extra", "new": None}]
    assert any(c.path == "/extra" for c in parse_transition(record).changed_fields)


def test_absent_and_null_leaves_differ_in_the_diff() -> None:
    """Report an added null leaf and a removed leaf with the absence marker."""
    added, removed = (
        preimage_changes({"a": 1}, {"a": 1, "b": None}),
        preimage_changes({"a": 1, "b": 2}, {"a": 1}),
    )
    assert added == (FieldChange("/b", ABSENT, None),)
    assert removed == (FieldChange("/b", 2, ABSENT),)
    assert repr(ABSENT) == "ABSENT"


@pytest.mark.parametrize(
    "change",
    ({"path": "/a"}, {"path": "/a", "old": 1, "other": 2}, {"old": 1, "new": 2}),
)
def test_changed_field_members_are_validated(change: dict[str, object]) -> None:
    """Refuse a changed field without a value or with an unknown member."""
    record = entry()
    record["changed_fields"] = [change]
    with pytest.raises(ValueError, match="changed field"):
        parse_transition(record)


def test_same_key_findings_are_compared_as_a_multiset() -> None:
    """Refuse a change in a key's multiplicity; detect a reworded duplicate."""
    transition = parse_transition(entry())
    one = Finding("packages/b.ts", RULES[0], 2, 1, "first")
    two = Finding("packages/b.ts", RULES[0], 2, 1, "second")
    before = snapshot(OLD_IDENTITY, one, two)
    assert compare_transition(before, snapshot(NEW_IDENTITY, one), transition) == (
        "Unchanged source lost an undeclared native finding: packages/b.ts",
    )
    assert compare_transition(
        snapshot(OLD_IDENTITY, one), snapshot(NEW_IDENTITY, one, two), transition
    ) == (
        "Unchanged source gained a native finding under the new identity: packages/b.ts",
    )
    reworded = Finding("packages/b.ts", RULES[0], 2, 1, "third")
    assert compare_transition(
        before, snapshot(NEW_IDENTITY, one, reworded), transition
    ) == ("Message-only finding changes differ from the declared list",)


def test_a_declared_drop_removes_every_occurrence_or_none() -> None:
    """Refuse a partial drop of a duplicated key and a drop never reported."""
    record = entry()
    record["drops"] = [
        {
            "path": "packages/b.ts",
            "rule": RULES[0],
            "line": 2,
            "column": 1,
            "cell": None,
            "changelog_digest": "9" * 64,
        }
    ]
    transition = parse_transition(record)
    one = Finding("packages/b.ts", RULES[0], 2, 1, "first")
    two = Finding("packages/b.ts", RULES[0], 2, 1, "second")
    kept = finding(line=7)
    before = snapshot(OLD_IDENTITY, one, two, kept)
    assert compare_transition(before, snapshot(NEW_IDENTITY, kept), transition) == ()
    partial = compare_transition(before, snapshot(NEW_IDENTITY, one, kept), transition)
    assert partial == (
        "A declared dropped finding is still reported: packages/b.ts",
        "Unchanged source lost an undeclared native finding: packages/b.ts",
    )
    assert compare_transition(
        snapshot(OLD_IDENTITY, kept), snapshot(NEW_IDENTITY, kept), transition
    ) == ("A declared dropped finding was never reported: packages/b.ts",)


def test_canary_findings_keep_their_multiplicity() -> None:
    """Accept a repeated canary key and require the run to repeat it too."""
    record = entry()
    canary = mapping(record, "canary")
    doubled = [*CANARY_KEYS, CANARY_KEYS[0]]
    canary.update(old_findings=doubled, new_findings=doubled)
    record["canary"] = canary
    transition = parse_transition(record)
    assert len(canary_of(transition).new_findings) == 3
    keys = tuple(sorted(set(canary_of(transition).new_findings)))
    refusals = verify_canary_run(
        transition,
        enabled_rules=RULES,
        fixture_digest="4" * 64,
        platform="linux-x64",
        engine_digest=ENGINE,
        findings=keys,
    )
    assert refusals == ("The canary findings differ from the recorded findings",)


@pytest.mark.parametrize(("before", "after"), ((1, True), (1, 1.0), ([1], [True])))
def test_leaves_differ_by_their_json_not_by_python_equality(
    before: object, after: object
) -> None:
    """Report a leaf whose JSON changes even when Python calls the values equal."""
    (change,) = preimage_changes({"flag": before}, {"flag": after})
    assert canonical_leaf(change.old) != canonical_leaf(change.new)


def test_undeclared_int_to_bool_leaf_is_refused() -> None:
    """Refuse an undeclared 1 to true change beside the declared ones."""
    record = entry()
    old = {**OLD_PREIMAGE, "flag": 1}
    new = {**NEW_PREIMAGE, "flag": True}
    record.update(
        old_preimage=old,
        new_preimage=new,
        old_identity=preimage_identity(old),
        new_identity=preimage_identity(new),
    )
    with pytest.raises(ValueError, match="differ from the preimage diff"):
        parse_transition(record)
    fields = members(record, "changed_fields")
    record["changed_fields"] = [*fields, {"path": "/flag", "old": 1, "new": True}]
    assert len(parse_transition(record).changed_fields) == 5
    record["changed_fields"] = [*fields, {"path": "/flag", "old": True, "new": True}]
    with pytest.raises(ValueError, match="differ from the preimage diff"):
        parse_transition(record)
