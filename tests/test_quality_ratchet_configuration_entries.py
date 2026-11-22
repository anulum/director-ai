# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the configuration change entries of the transition file
"""Pin how a ``configuration_change`` entry is parsed and when it needs RIGOR."""

from __future__ import annotations

import pytest

from tests.quality_ratchet_configuration_support import (
    AFTER,
    BEFORE,
    LATER,
    PROFILES,
    configuration_entry,
    pyproject,
    with_field,
)
from tests.quality_ratchet_succession_support import entry, file_bytes
from tools.quality_ratchet_configuration import EXCLUDE_POINTER, PYPROJECT
from tools.quality_ratchet_succession import CONFIGURATION_REASON
from tools.quality_ratchet_succession_load import require_append_only
from tools.quality_ratchet_succession_parse import parse_transition, parse_transitions

FIRST_USE = "A configuration path used for the first time needs the statistics approval"
OUTSIDE = "A configuration transition changes a field outside its configuration files: "


@pytest.mark.parametrize("profile", PROFILES)
def test_a_configuration_entry_parses_without_canary_or_artefacts(
    profile: str,
) -> None:
    """Parse the entry with its recorded pointers and no canary."""
    transition = parse_transition(configuration_entry(profile))
    assert transition.reason == CONFIGURATION_REASON
    assert transition.canary is None
    assert transition.verification_records == {}
    assert transition.drops == () and transition.message_only == ()
    assert transition.configuration_paths == {PYPROJECT: (EXCLUDE_POINTER,)}
    assert transition.rigor_approval_digest == "8" * 64
    assert transition.packet_digest == "6" * 64
    assert transition.authority_approval_digest == "7" * 64


def test_only_a_configuration_entry_has_the_paths_member() -> None:
    """Refuse the member on other reasons and its absence on this one."""
    other = entry()
    other["configuration_paths"] = {}
    missing = configuration_entry()
    del missing["configuration_paths"]
    renamed = configuration_entry()
    renamed["reason"] = "engine_fix"
    for item in (other, missing, renamed):
        with pytest.raises(ValueError, match="Invalid transition entry members"):
            parse_transition(item)
    assert parse_transition(entry()).configuration_paths == {}


@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("canary", {}),
        ("verification_records", {"/configurations/pyproject.toml": "1" * 64}),
        ("drops", [{}]),
        ("message_only", [{}]),
    ],
)
def test_a_configuration_entry_has_no_canary_artefact_drop_or_rewording(
    member: str, value: object
) -> None:
    """Refuse any evidence member that only a tool change can carry."""
    item = configuration_entry()
    item[member] = value
    with pytest.raises(ValueError, match="has no canary, artefact, drop or reworded"):
        parse_transition(item)


@pytest.mark.parametrize(
    ("profile", "pointer"),
    [
        ("python-docs", "/configurations/packages~1director-ai-lite~1pyproject.toml"),
        ("python-docs", "/configurations/requirements~1ci-lint.txt"),
        ("python-docs", "/flags"),
        ("python-types", "/configuration"),
        ("python-types", "/environment"),
        ("notebook-types", "/configuration_hashes/requirements~1ci-dev.txt"),
        ("notebook-types", "/configuration_hashes/MANIFEST.in"),
    ],
)
def test_a_changed_field_outside_the_map_is_refused_by_name(
    profile: str, pointer: str
) -> None:
    """Refuse a packaging, other-file, MyPy-text or environment field change."""
    item = with_field(configuration_entry(profile), pointer, "f" * 64)
    with pytest.raises(ValueError, match=f"^{OUTSIDE}{pointer}$"):
        parse_transition(item)


def test_a_profile_outside_the_map_cannot_use_the_reason() -> None:
    """Refuse a configuration entry for a profile with no reviewed field."""
    item = configuration_entry()
    item["profile"] = "go-docs"
    with pytest.raises(ValueError, match=f"^{OUTSIDE}/configurations/pyproject.toml$"):
        parse_transition(item)


@pytest.mark.parametrize(
    "paths",
    [
        {},
        {PYPROJECT: [EXCLUDE_POINTER], "requirements/ci-lint.txt": []},
        {"requirements/ci-lint.txt": [EXCLUDE_POINTER]},
    ],
)
def test_the_recorded_files_are_exactly_the_changed_files(
    paths: dict[str, object],
) -> None:
    """Refuse a record that names more, fewer or other files than changed."""
    item = configuration_entry()
    item["configuration_paths"] = paths
    with pytest.raises(ValueError, match="must name exactly the changed files"):
        parse_transition(item)


def test_the_recorded_paths_must_be_a_mapping() -> None:
    """Refuse a record that is not a JSON object."""
    item = configuration_entry()
    item["configuration_paths"] = [PYPROJECT]
    with pytest.raises(ValueError):
        parse_transition(item)


@pytest.mark.parametrize(
    "pointers",
    [
        EXCLUDE_POINTER,
        [1],
        [EXCLUDE_POINTER, EXCLUDE_POINTER],
        [EXCLUDE_POINTER + "/1", EXCLUDE_POINTER + "/0"],
        ["tool/setuptools/packages/find/exclude"],
    ],
)
def test_the_recorded_pointers_are_sorted_unique_pointers(pointers: object) -> None:
    """Refuse a non-list, a non-string, a repeat, an unsorted list or a bare name."""
    item = configuration_entry()
    item["configuration_paths"] = {PYPROJECT: pointers}
    with pytest.raises(ValueError, match="^Invalid configuration paths$"):
        parse_transition(item)


@pytest.mark.parametrize(
    "pointer", ["/tool/ruff", "/tool/setuptools", "/tool/setuptools/packages/find"]
)
def test_a_recorded_pointer_must_be_on_the_reviewed_list(pointer: str) -> None:
    """Refuse a pointer that is not under the non-influencing list."""
    item = configuration_entry()
    item["configuration_paths"] = {PYPROJECT: [pointer]}
    with pytest.raises(ValueError, match="not on the non-influencing list"):
        parse_transition(item)


def test_a_pointer_under_a_listed_one_and_an_empty_record_parse() -> None:
    """Accept a descendant of a listed pointer, and a record of no pointer."""
    for pointers in ((EXCLUDE_POINTER + "/0",), ()):
        transition = parse_transition(configuration_entry(pointers=pointers))
        assert transition.configuration_paths == {PYPROJECT: pointers}


def test_the_packet_and_approvals_must_be_distinct_records() -> None:
    """Refuse a reused digest, with or without the statistics approval."""
    alone = configuration_entry(rigor=None)
    alone["authority_approval_digest"] = alone["packet_digest"]
    with pytest.raises(ValueError, match="^The packet and its approval must be"):
        parse_transition(alone)
    both = configuration_entry()
    both["rigor_approval_digest"] = both["packet_digest"]
    with pytest.raises(ValueError, match="^The packet and its two approvals must be"):
        parse_transition(both)
    malformed = configuration_entry(rigor="not a digest")
    with pytest.raises(ValueError, match="Invalid SHA-256 identity"):
        parse_transition(malformed)


def test_only_a_configuration_entry_may_omit_the_statistics_approval() -> None:
    """Refuse a null statistics approval on every other reason."""
    other = entry()
    other["rigor_approval_digest"] = None
    with pytest.raises(ValueError, match="Invalid SHA-256 identity"):
        parse_transition(other)
    assert (
        parse_transition(configuration_entry(rigor=None)).rigor_approval_digest is None
    )


def test_the_first_use_of_a_pointer_needs_the_statistics_approval() -> None:
    """Judge first use from the file's own earlier entries, per profile."""
    with pytest.raises(ValueError, match=FIRST_USE):
        parse_transitions(file_bytes(configuration_entry(rigor=None)))
    reviewed = configuration_entry()
    later = configuration_entry(before=AFTER, after=LATER, rigor=None)
    entries = parse_transitions(file_bytes(reviewed, later))
    assert [e.rigor_approval_digest for e in entries] == ["8" * 64, None]
    with pytest.raises(ValueError, match=FIRST_USE):
        parse_transitions(file_bytes(later, reviewed))
    other = configuration_entry("python-types", rigor=None)
    with pytest.raises(ValueError, match=FIRST_USE):
        parse_transitions(file_bytes(reviewed, other))
    deeper = configuration_entry(
        before=AFTER, after=LATER, pointers=(EXCLUDE_POINTER + "/0",), rigor=None
    )
    with pytest.raises(ValueError, match=FIRST_USE):
        parse_transitions(file_bytes(reviewed, deeper))
    assert len(parse_transitions(file_bytes(entry(), reviewed, later))) == 3


def test_a_change_of_no_pointer_needs_no_statistics_approval() -> None:
    """Accept a comment-only entry without RIGOR; it uses no pointer."""
    comment = configuration_entry(
        before=BEFORE, after=pyproject(comment="reworded"), pointers=(), rigor=None
    )
    (transition,) = parse_transitions(file_bytes(comment))
    assert transition.configuration_paths == {PYPROJECT: ()}


def test_removing_the_reviewed_entry_cannot_make_a_pointer_new_again() -> None:
    """Refuse the removal itself, and judge the shortened file on its own."""
    reviewed = configuration_entry()
    later = configuration_entry(before=AFTER, after=LATER, rigor=None)
    with pytest.raises(ValueError, match="append-only"):
        require_append_only(file_bytes(reviewed, later), file_bytes(later))
    with pytest.raises(ValueError, match=FIRST_USE):
        parse_transitions(file_bytes(later))
