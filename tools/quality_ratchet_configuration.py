# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the fail-closed configuration change rules of the ratchet
"""Decide whether a configuration file change is raw-only for a profile.

Three Python profiles bind the digest of the whole ``pyproject.toml``, so any
edit changes their identities, even one that no tool of the profile reads. A
``configuration_change`` transition lets such an edit through, under these
rules and nothing weaker:

- the only changed identity fields are the configuration-file digest fields
  of :data:`CONFIGURATION_FIELDS`;
- the whole parsed document is diffed, and every changed path must lie under
  a reviewed pointer of :data:`NON_INFLUENCING` for that profile and file, so
  an influencing key nobody listed is refused, never passed unseen;
- the pointers of :data:`INFLUENCING` must compare equal as a second check,
  and the two lists may not overlap;
- the commit that adds the entry changes only the named configuration files,
  the transitions file, the baseline's identity lines, the reviewed packaging
  metadata of :data:`PACKAGING_METADATA` and at most one placeholder that
  keeps an excluded package directory present.

Adding a pointer to :data:`NON_INFLUENCING` is a reviewed change of this
module, approved by the ratchet authority. This module is no profile's engine
input, so changing it changes no identity.
"""

from __future__ import annotations

import datetime
import fnmatch
import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from typing import cast

from tools.quality_ratchet_baseline import BASELINE_PATH
from tools.quality_ratchet_contracts import Snapshot
from tools.quality_ratchet_succession import (
    TRANSITIONS_PATH,
    pointer_leaf,
    pointer_token,
)

PYPROJECT = "pyproject.toml"

CONFIGURATION_FIELDS: dict[str, dict[str, str]] = {
    "python-docs": {"/configurations/pyproject.toml": PYPROJECT},
    "python-types": {"/configuration_hashes/pyproject.toml": PYPROJECT},
    "notebook-types": {"/configuration_hashes/pyproject.toml": PYPROJECT},
}
"""Per profile, each configuration-file digest field and the file it binds."""

INFLUENCING: dict[str, dict[str, tuple[str, ...]]] = {
    "python-docs": {PYPROJECT: ("/project/requires-python", "/tool/ruff")},
    "python-types": {PYPROJECT: ("/project/requires-python", "/tool/mypy")},
    "notebook-types": {PYPROJECT: ("/project/requires-python", "/tool/mypy")},
}
"""Per profile and file, the configuration a profile's tools read; must stay equal."""

NON_INFLUENCING: dict[str, dict[str, tuple[str, ...]]] = {
    profile: {PYPROJECT: ("/tool/setuptools/packages/find/exclude",)}
    for profile in ("python-docs", "python-types", "notebook-types")
}
"""Per profile and file, the reviewed pointers no tool of the profile reads.

The setuptools package exclusion was reviewed as non-influencing for the three
Python profiles under two conditions, which each need a new review if they
change: (a) no profile environment installs the project distribution, edited
or built, and (b) MyPy resolves project imports from the source tree
(``mypy_path`` and ``explicit_package_bases``). An install would change the
bound environment, a field outside :data:`CONFIGURATION_FIELDS`, and both MyPy
settings lie under the influencing ``/tool/mypy``; both are therefore refused.
"""

PACKAGING_METADATA = (
    "MANIFEST.in",
    "packages/paid_tier_manifest.json",
    "packages/*/pyproject.toml",
)
"""Packaging files a configuration commit may change; ``*`` is one path segment."""

PLACEHOLDER = re.compile(r"src/director_ai/([a-z][a-z0-9_]*)/README\.md")
"""The one placeholder a configuration commit may add, under an excluded package."""

EXCLUDE_POINTER = "/tool/setuptools/packages/find/exclude"
"""Where ``pyproject.toml`` names the packages left out of the public wheel."""


def under(pointer: str, prefix: str) -> bool:
    """Return whether a JSON Pointer names the prefix's value or lies inside it.

    Parameters
    ----------
    pointer, prefix : str
        RFC 6901 pointers.

    Returns
    -------
    bool
        True when the pointer equals the prefix or extends it by whole tokens.
    """
    return pointer == prefix or pointer.startswith(prefix + "/")


def canonical_value(value: object) -> str:
    """Return a TOML value's encoding, which keeps its type.

    Parameters
    ----------
    value : object
        A value ``tomllib`` produced, or None for an absent one.

    Returns
    -------
    str
        A type-tagged encoding. Two values compare equal exactly when their
        encodings do, so ``1``, ``1.0``, ``true``, a date, its ISO string and
        a table holding that string are all different values.
    """
    if value is None:
        return "absent"
    if isinstance(value, bool):
        return "bool:" + json.dumps(value)
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        return f"float:{value!r}"
    if isinstance(value, str):
        return "str:" + json.dumps(value)
    if isinstance(value, datetime.datetime):
        return "datetime:" + value.isoformat()
    if isinstance(value, datetime.date):
        return "date:" + value.isoformat()
    if isinstance(value, datetime.time):
        return "time:" + value.isoformat()
    if isinstance(value, list):
        return "array:[" + ",".join(canonical_value(item) for item in value) + "]"
    table = cast("dict[str, object]", value)
    return (
        "table:{"
        + ",".join(
            json.dumps(key) + "=" + canonical_value(item)
            for key, item in sorted(table.items())
        )
        + "}"
    )


def parse_configuration(data: bytes | None, name: str) -> dict[str, object]:
    """Parse one TOML configuration file.

    Parameters
    ----------
    data : bytes | None
        The file's bytes, or None when the commit has no such file.
    name : str
        The file's repository path, for the refusal.

    Returns
    -------
    dict[str, object]
        The document as ``tomllib`` returns it.

    Raises
    ------
    ValueError
        If the file is missing or is not valid UTF-8 TOML.
    """
    if data is None:
        raise ValueError(f"A configuration transition names a missing file: {name}")
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"A configuration file is not valid TOML: {name}") from error


def _leaves(value: object, prefix: str = "") -> dict[str, str]:
    """Flatten tables into encoded JSON Pointer leaves; arrays stay whole."""
    if not isinstance(value, dict) or not value:
        return {prefix: canonical_value(value)}
    leaves: dict[str, str] = {}
    for key, item in value.items():
        leaves.update(_leaves(item, f"{prefix}/{pointer_token(key)}"))
    return leaves


def changed_pointers(
    base: dict[str, object], candidate: dict[str, object]
) -> tuple[str, ...]:
    """Return every leaf in which two whole configuration documents differ.

    Parameters
    ----------
    base, candidate : dict[str, object]
        Parsed documents.

    Returns
    -------
    tuple[str, ...]
        Sorted RFC 6901 pointers of added, removed and changed leaves. An
        array is one leaf, as in identity preimages, and an empty table is a
        leaf of its own; values compare by :func:`canonical_value`.
    """
    before, after = _leaves(base), _leaves(candidate)
    return tuple(
        sorted(
            pointer
            for pointer in set(before) | set(after)
            if before.get(pointer) != after.get(pointer)
        )
    )


def raw_only_refusals(
    profile: str,
    name: str,
    base: dict[str, object],
    candidate: dict[str, object],
    recorded: Sequence[str],
) -> tuple[str, ...]:
    """Refuse a configuration change that any tool of the profile could read.

    Parameters
    ----------
    profile : str
        The native profile the entry names.
    name : str
        The configuration file.
    base, candidate : dict[str, object]
        The file's parsed documents before and after.
    recorded : Sequence[str]
        The pointers the entry records for this file.

    Returns
    -------
    tuple[str, ...]
        Sorted refusals; empty means the change is raw-only for this profile.
    """
    allowed = NON_INFLUENCING.get(profile, {}).get(name, ())
    pointers = changed_pointers(base, candidate)
    refusals = {
        f"Effective configuration changed; needs canaries: {name} {pointer}"
        for pointer in pointers
        if not any(under(pointer, prefix) for prefix in allowed)
    }
    refusals.update(
        f"Influencing configuration changed: {name} {pointer}"
        for pointer in INFLUENCING.get(profile, {}).get(name, ())
        if canonical_value(pointer_leaf(base, pointer))
        != canonical_value(pointer_leaf(candidate, pointer))
    )
    if tuple(sorted(recorded)) != pointers:
        refusals.add(
            f"Recorded configuration paths differ from the configuration diff: {name}"
        )
    return tuple(sorted(refusals))


def excluded_packages(candidate: dict[str, object]) -> frozenset[str]:
    """Return the ``director_ai`` subpackages the public wheel leaves out.

    Parameters
    ----------
    candidate : dict[str, object]
        The candidate's parsed ``pyproject.toml``.

    Returns
    -------
    frozenset[str]
        Top-level subpackage names from patterns such as ``director_ai.x*``.
    """
    value = pointer_leaf(candidate, EXCLUDE_POINTER)
    names = set()
    for item in value if isinstance(value, list) else ():
        match = re.fullmatch(r"director_ai\.([a-z][a-z0-9_]*)\*?", str(item))
        if match:
            names.add(match.group(1))
    return frozenset(names)


def shape_refusals(
    changed: Mapping[str, str],
    named_files: frozenset[str],
    excluded: frozenset[str],
) -> tuple[str, ...]:
    """Refuse every path a configuration commit may not change.

    Parameters
    ----------
    changed : Mapping[str, str]
        Each changed repository path and its Git status letter (A, M, D, …).
    named_files : frozenset[str]
        The configuration files the commit's entries name.
    excluded : frozenset[str]
        The packages the candidate's configuration excludes from the wheel.

    Returns
    -------
    tuple[str, ...]
        Sorted refusals; empty means the commit has the declared shape.
    """
    refusals: set[str] = set()
    placeholders = []
    for path, status in sorted(changed.items()):
        if path in named_files or path in (TRANSITIONS_PATH, BASELINE_PATH):
            continue
        if any(
            fnmatch.fnmatchcase(path, pattern) and path.count("/") == pattern.count("/")
            for pattern in PACKAGING_METADATA
        ):
            continue
        placeholder = PLACEHOLDER.fullmatch(path)
        if placeholder and status == "A" and placeholder.group(1) in excluded:
            placeholders.append(path)
            continue
        refusals.add(f"A configuration transition carries no source change: {path}")
    if len(placeholders) > 1:
        refusals.add("A configuration transition adds at most one placeholder")
    return tuple(sorted(refusals))


def baseline_delta_refusals(
    previous: Mapping[str, Snapshot],
    candidate: Mapping[str, Snapshot],
    profiles: frozenset[str],
) -> tuple[str, ...]:
    """Refuse any baseline change but the named profiles' identities.

    Parameters
    ----------
    previous, candidate : Mapping[str, Snapshot]
        The baselines' snapshots by profile, before and after.
    profiles : frozenset[str]
        The profiles the commit's configuration entries name.

    Returns
    -------
    tuple[str, ...]
        Sorted refusals; empty means only those identity lines changed.
    """
    refusals = set()
    for name in sorted(set(previous) | set(candidate)):
        before, after = previous.get(name), candidate.get(name)
        if before is None or after is None:
            refusals.add(f"A configuration transition changes the profile set: {name}")
            continue
        if name not in profiles and before.identity != after.identity:
            refusals.add(f"A configuration transition changes another identity: {name}")
        if (before.sources, before.findings, before.complete, before.advisories) != (
            after.sources,
            after.findings,
            after.complete,
            after.advisories,
        ):
            refusals.add(
                f"A configuration transition changes more than identity: {name}"
            )
    return tuple(sorted(refusals))
