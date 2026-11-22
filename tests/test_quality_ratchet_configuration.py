# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the fail-closed configuration change rules
"""Pin the rules that decide whether a configuration change is raw-only."""

from __future__ import annotations

import dataclasses
import fnmatch
from typing import cast

import pytest

import tools.quality_ratchet_configuration as configuration_module
from tests.quality_ratchet_configuration_support import (
    PROFILES,
    WIDER,
    pyproject,
)
from tools.quality_ratchet_baseline import BASELINE_PATH
from tools.quality_ratchet_configuration import (
    CONFIGURATION_FIELDS,
    EXCLUDE_POINTER,
    INFLUENCING,
    NON_INFLUENCING,
    PACKAGING_METADATA,
    PLACEHOLDER,
    PYPROJECT,
    baseline_delta_refusals,
    canonical_value,
    changed_pointers,
    excluded_packages,
    parse_configuration,
    raw_only_refusals,
    shape_refusals,
    under,
)
from tools.quality_ratchet_contracts import Finding, Snapshot
from tools.quality_ratchet_profiles import PROFILE_CONFIGURATIONS
from tools.quality_ratchet_succession import TRANSITIONS_PATH, pointer_token

EFFECTIVE = "Effective configuration changed; needs canaries: pyproject.toml"
RECORDED = "Recorded configuration paths differ from the configuration diff: "
RECORDED += PYPROJECT
NAMED = frozenset({PYPROJECT})
ADAPTERS = frozenset({"training_adapters"})
PLACEHOLDER_PATH = "src/director_ai/training_adapters/README.md"
EXCLUDE_TABLE = tuple(EXCLUDE_POINTER.split("/")[1:])


def parse(data: bytes) -> dict[str, object]:
    """Parse ``pyproject.toml`` bytes.

    Parameters
    ----------
    data : bytes
        The file's bytes.

    Returns
    -------
    dict[str, object]
        The parsed document.
    """
    return parse_configuration(data, PYPROJECT)


def mutated(path: tuple[str, ...], value: object) -> dict[str, object]:
    """Return the base document with one key set, creating its tables.

    Parameters
    ----------
    path : tuple[str, ...]
        The key's table path and name.
    value : object
        Its new value.

    Returns
    -------
    dict[str, object]
        The changed document.
    """
    document = parse(pyproject())
    node = document
    for key in path[:-1]:
        node = cast("dict[str, object]", node.setdefault(key, {}))
    node[path[-1]] = value
    return document


def test_the_reviewed_lists_are_pinned_and_cover_the_same_files() -> None:
    """Give each mapped profile both lists, over a file its identity binds."""
    reviewed = {p: {PYPROJECT: (EXCLUDE_POINTER,)} for p in PROFILES}
    assert reviewed == NON_INFLUENCING
    assert INFLUENCING == {
        "python-docs": {PYPROJECT: ("/project/requires-python", "/tool/ruff")},
        "python-types": {PYPROJECT: ("/project/requires-python", "/tool/mypy")},
        "notebook-types": {PYPROJECT: ("/project/requires-python", "/tool/mypy")},
    }
    for profile, fields in CONFIGURATION_FIELDS.items():
        files = set(fields.values())
        assert files <= set(PROFILE_CONFIGURATIONS[profile])
        assert set(INFLUENCING[profile]) == set(NON_INFLUENCING[profile]) == files
        for pointer, name in fields.items():
            assert pointer.endswith("/" + pointer_token(name))


def test_the_two_lists_are_disjoint_with_no_prefix_overlap() -> None:
    """Keep every influencing pointer apart from every non-influencing one."""
    for profile, files in NON_INFLUENCING.items():
        for name, allowed in files.items():
            for pointer in allowed:
                for influencing in INFLUENCING[profile][name]:
                    assert not under(pointer, influencing)
                    assert not under(influencing, pointer)


def test_no_packaging_file_or_placeholder_configures_any_profile() -> None:
    """Keep every path the commit shape allows out of all bound configurations."""
    for files in PROFILE_CONFIGURATIONS.values():
        for name in files:
            assert PLACEHOLDER.fullmatch(name) is None
            assert not any(
                fnmatch.fnmatchcase(name, pattern) for pattern in PACKAGING_METADATA
            )


@pytest.mark.parametrize(
    ("pointer", "prefix", "expected"),
    [
        ("/tool/ruff", "/tool/ruff", True),
        ("/tool/ruff/lint/select", "/tool/ruff", True),
        ("/tool/ruffx", "/tool/ruff", False),
        ("/tool", "/tool/ruff", False),
    ],
)
def test_a_pointer_lies_under_a_prefix_by_whole_tokens(
    pointer: str, prefix: str, expected: bool
) -> None:
    """Match the prefix itself or its descendants, never a longer name."""
    assert under(pointer, prefix) is expected


def test_a_configuration_file_must_exist_and_be_utf8_toml() -> None:
    """Refuse a missing or undecodable file."""
    with pytest.raises(ValueError, match="names a missing file: pyproject.toml"):
        parse_configuration(None, PYPROJECT)
    for data in (b"[project\n", b"name = '\xff'\n"):
        with pytest.raises(ValueError, match="is not valid TOML: pyproject.toml"):
            parse_configuration(data, PYPROJECT)


def test_the_diff_covers_the_whole_document_with_arrays_as_leaves() -> None:
    """Report every changed leaf of the two documents, sorted."""
    base = parse(pyproject())
    candidate = parse(
        pyproject(WIDER, ruff=100, extra="[tool.other]\nwhen = 1979-05-27\n")
    )
    assert changed_pointers(base, candidate) == (
        "/tool/other/when",
        "/tool/ruff/line-length",
        EXCLUDE_POINTER,
    )
    assert changed_pointers(base, base) == ()
    assert changed_pointers(parse(b"[a]\n"), parse(b"[a]\nb = 1\n")) == ("/a", "/a/b")
    assert changed_pointers(parse(b"a = {b = 1}\n"), parse(b"a = 1\n")) == (
        "/a",
        "/a/b",
    )
    assert changed_pointers(parse(b'"x/y~" = 1\n'), parse(b'"x/y~" = 2\n')) == (
        "/x~1y~0",
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (b"a = 1", b"a = 1.0"),
        (b"a = 1", b"a = true"),
        (b"a = 0.0", b"a = -0.0"),
        (b'a = "1"', b"a = 1"),
        (b"a = 1979-05-27", b'a = "1979-05-27"'),
        (b"a = 1979-05-27", b'a = {toml-datetime = "1979-05-27"}'),
        (b"a = 1979-05-27", b"a = 1979-05-27T00:00:00"),
        (b"a = 1979-05-27T07:32:00", b"a = 1979-05-27T07:32:00Z"),
        (b"a = 07:32:00", b'a = "07:32:00"'),
        (b"a = [1, 2]", b"a = [2, 1]"),
        (b"a = [[1]]", b"a = [1]"),
        (b"a = []", b"a = {}"),
        (b'a = {b = "1"}', b'a = {b = "2"}'),
    ],
)
def test_values_compare_with_their_toml_type(before: bytes, after: bytes) -> None:
    """Treat values of different TOML types as different, never as equal."""
    first, second = parse(before + b"\n"), parse(after + b"\n")
    assert changed_pointers(first, second) != ()
    assert canonical_value(first["a"]) != canonical_value(second["a"])
    assert changed_pointers(first, parse(before + b"\n")) == ()


def test_an_absent_value_differs_from_every_present_one() -> None:
    """Encode absence apart from any value, the empty table included."""
    assert canonical_value(None) == "absent"
    assert canonical_value({}) != canonical_value(None)
    assert canonical_value({"b": [1, "x"]}) == 'table:{"b"=array:[int:1,str:"x"]}'


@pytest.mark.parametrize("profile", PROFILES)
def test_an_exclusion_change_is_raw_only_for_each_python_profile(
    profile: str,
) -> None:
    """Pass a widened or emptied exclusion that the entry records."""
    base = parse(pyproject())
    for candidate in (parse(pyproject(WIDER)), parse(pyproject(()))):
        assert (
            raw_only_refusals(profile, PYPROJECT, base, candidate, (EXCLUDE_POINTER,))
            == ()
        )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "path",
    [
        ("tool", "black", "line-length"),
        ("tool", "setuptools", "packages", "find", "include"),
        ("tool", "setuptools", "packages", "find", "where"),
        ("tool", "setuptools", "packages", "find", "exclude-extra"),
        ("project", "name"),
        ("build-system", "requires"),
    ],
)
def test_an_unlisted_key_is_refused_fail_closed(
    profile: str, path: tuple[str, ...]
) -> None:
    """Refuse a change nobody listed, even when no influencing pointer moved."""
    candidate = mutated(path, ["changed"])
    pointer = "/" + "/".join(path)
    assert raw_only_refusals(
        profile, PYPROJECT, parse(pyproject()), candidate, (pointer,)
    ) == (f"{EFFECTIVE} {pointer}",)


@pytest.mark.parametrize(
    ("profile", "path", "influencing"),
    [
        ("python-docs", ("tool", "ruff", "line-length"), "/tool/ruff"),
        ("python-types", ("tool", "mypy", "strict"), "/tool/mypy"),
        ("notebook-types", ("tool", "mypy", "mypy_path"), "/tool/mypy"),
        *(
            (profile, ("project", "requires-python"), "/project/requires-python")
            for profile in PROFILES
        ),
    ],
)
def test_an_influencing_change_is_refused_by_both_checks(
    profile: str, path: tuple[str, ...], influencing: str
) -> None:
    """Refuse an influencing change as unlisted and as unequal."""
    candidate = mutated(path, "changed")
    pointer = "/" + "/".join(path)
    assert raw_only_refusals(
        profile, PYPROJECT, parse(pyproject()), candidate, (pointer,)
    ) == (
        f"{EFFECTIVE} {pointer}",
        f"Influencing configuration changed: pyproject.toml {influencing}",
    )


def test_the_equality_check_holds_if_the_lists_ever_overlapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch an influencing change by equality alone, the second check."""
    monkeypatch.setitem(
        configuration_module.NON_INFLUENCING,
        "python-docs",
        {PYPROJECT: (EXCLUDE_POINTER, "/tool/ruff")},
    )
    candidate = parse(pyproject(ruff=120))
    assert raw_only_refusals(
        "python-docs",
        PYPROJECT,
        parse(pyproject()),
        candidate,
        ("/tool/ruff/line-length",),
    ) == ("Influencing configuration changed: pyproject.toml /tool/ruff",)


def test_a_check_of_another_tool_is_not_influencing_but_still_refused() -> None:
    """Refuse a Ruff change for the typing profiles, which only list MyPy."""
    candidate = parse(pyproject(ruff=120))
    for profile in ("python-types", "notebook-types"):
        assert raw_only_refusals(
            profile,
            PYPROJECT,
            parse(pyproject()),
            candidate,
            ("/tool/ruff/line-length",),
        ) == (f"{EFFECTIVE} /tool/ruff/line-length",)


def test_the_recorded_paths_must_equal_the_recomputed_diff() -> None:
    """Refuse a record that names fewer, more or other pointers than changed."""
    base, candidate = parse(pyproject()), parse(pyproject(WIDER))
    for recorded in ((), (EXCLUDE_POINTER, EXCLUDE_POINTER), ("/tool/setuptools",)):
        assert raw_only_refusals(
            "python-docs", PYPROJECT, base, candidate, recorded
        ) == (RECORDED,)
    assert raw_only_refusals(
        "python-docs", PYPROJECT, base, base, (EXCLUDE_POINTER,)
    ) == (RECORDED,)


def test_a_profile_or_file_without_a_reviewed_list_refuses_every_change() -> None:
    """Allow nothing for a profile or a file that no list names."""
    base, candidate = parse(pyproject()), parse(pyproject(WIDER))
    for profile, name in (
        ("go-docs", PYPROJECT),
        ("python-docs", "requirements/ci-lint.txt"),
    ):
        refusals = raw_only_refusals(profile, name, base, candidate, (EXCLUDE_POINTER,))
        assert refusals == (
            f"Effective configuration changed; needs canaries: {name} "
            f"{EXCLUDE_POINTER}",
        )


def test_a_comment_or_layout_change_has_no_pointer() -> None:
    """Pass changed bytes whose parsed documents are equal, with no record."""
    base, edited = pyproject(), pyproject(comment="the project, reformatted")
    assert base != edited
    for profile in PROFILES:
        assert (
            raw_only_refusals(profile, PYPROJECT, parse(base), parse(edited), ()) == ()
        )


def test_the_excluded_packages_come_from_director_ai_patterns_only() -> None:
    """Read top-level subpackage names; ignore every other pattern or value."""
    document = parse(
        pyproject(
            (
                "director_ai.training_adapters*",
                "director_ai.experimental",
                "tests*",
                "director_ai.core.sub*",
                "director_ai.Bad*",
                "director_ai.x**",
            )
        )
    )
    assert excluded_packages(document) == frozenset(
        {"training_adapters", "experimental"}
    )
    assert excluded_packages({}) == frozenset()
    assert excluded_packages(mutated(EXCLUDE_TABLE, "director_ai.x*")) == (frozenset())
    assert excluded_packages(mutated(EXCLUDE_TABLE, [1])) == frozenset()


def test_the_reviewed_paths_pass_the_commit_shape() -> None:
    """Allow the named file, the ratchet files, packaging metadata, one placeholder."""
    changed = {
        PYPROJECT: "M",
        TRANSITIONS_PATH: "A",
        BASELINE_PATH: "M",
        "MANIFEST.in": "M",
        "packages/paid_tier_manifest.json": "D",
        "packages/director-ai-lite/pyproject.toml": "M",
        PLACEHOLDER_PATH: "A",
    }
    assert shape_refusals(changed, NAMED, ADAPTERS) == ()
    assert shape_refusals({}, NAMED, frozenset()) == ()


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("src/director_ai/core/scorer.py", "M"),
        ("src/director_ai/training_adapters/__init__.py", "A"),
        ("tests/test_scorer.py", "A"),
        ("packages/director-ai-lite/src/lite.py", "A"),
        ("packages/director-ai-lite/extra/pyproject.toml", "M"),
        ("packages/pyproject.toml", "A"),
        ("packages/director-ai-lite/setup.cfg", "M"),
        ("docs/MANIFEST.in", "M"),
        ("requirements/ci-lint.txt", "M"),
        (PLACEHOLDER_PATH, "M"),
        (PLACEHOLDER_PATH, "D"),
        ("src/director_ai/core/README.md", "A"),
        ("src/director_ai/training_adapters/sub/README.md", "A"),
        ("src/director_ai/training_adapters/readme.md", "A"),
        ("tools/quality_ratchet_baseline.json.license", "M"),
        (PYPROJECT, "M"),
    ],
)
def test_any_other_path_is_refused_as_a_source_change(path: str, status: str) -> None:
    """Refuse every path outside the shape, and an unnamed configuration file."""
    named = frozenset() if path == PYPROJECT else NAMED
    assert shape_refusals({path: status}, named, ADAPTERS) == (
        f"A configuration transition carries no source change: {path}",
    )


def test_at_most_one_placeholder_is_added() -> None:
    """Refuse a second placeholder even when both packages are excluded."""
    changed = {
        PLACEHOLDER_PATH: "A",
        "src/director_ai/experimental/README.md": "A",
    }
    assert shape_refusals(
        changed, NAMED, frozenset({"training_adapters", "experimental"})
    ) == ("A configuration transition adds at most one placeholder",)
    assert shape_refusals(changed, NAMED, ADAPTERS) == (
        "A configuration transition carries no source change: "
        "src/director_ai/experimental/README.md",
    )


def snapshot(profile: str, identity: str) -> Snapshot:
    """Return a complete snapshot over one source with one finding.

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
    return Snapshot(
        profile,
        identity,
        {"src/a.py": "a" * 64},
        (Finding("src/a.py", "D103", 1, 1, "Missing docstring"),),
        True,
    )


def test_only_the_named_profiles_identity_lines_may_change() -> None:
    """Refuse another identity, any other snapshot field and a new profile set."""
    previous = {p: snapshot(p, "1" * 64) for p in (*PROFILES, "go-docs")}
    candidate = dict(previous)
    for profile in PROFILES:
        candidate[profile] = snapshot(profile, "2" * 64)
    named = frozenset(PROFILES)
    assert baseline_delta_refusals(previous, candidate, named) == ()
    assert baseline_delta_refusals(
        previous, candidate, frozenset({"python-docs", "notebook-types"})
    ) == ("A configuration transition changes another identity: python-types",)
    for change in (
        {"sources": {"src/a.py": "b" * 64}},
        {"findings": ()},
        {"complete": False},
        {"advisories": ("advice",)},
    ):
        for profile in ("python-docs", "go-docs"):
            moved = dict(candidate)
            moved[profile] = dataclasses.replace(candidate[profile], **change)
            assert baseline_delta_refusals(previous, moved, named) == (
                f"A configuration transition changes more than identity: {profile}",
            )
    fewer = {p: s for p, s in candidate.items() if p != "go-docs"}
    assert baseline_delta_refusals(previous, fewer, named) == (
        "A configuration transition changes the profile set: go-docs",
    )
    assert baseline_delta_refusals(fewer, candidate, named) == (
        "A configuration transition changes the profile set: go-docs",
    )
