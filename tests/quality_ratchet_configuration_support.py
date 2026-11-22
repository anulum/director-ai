# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — shared builders for the configuration change tests
"""Build project files and valid configuration entries that tests then break."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import cast

from tools.quality_ratchet_configuration import (
    CONFIGURATION_FIELDS,
    EXCLUDE_POINTER,
    PYPROJECT,
)
from tools.quality_ratchet_succession import CONFIGURATION_REASON, preimage_identity

PROFILES = tuple(CONFIGURATION_FIELDS)
"""The three Python profiles that bind the whole ``pyproject.toml``."""
EXCLUDED = ("director_ai.experimental*",)
"""The base project's exclusion from the public wheel."""
WIDER = (*EXCLUDED, "director_ai.training_adapters*")
"""The exclusion after the reviewed raw-only change."""
WIDEST = (*WIDER, "director_ai.voice*")
"""A second, later exclusion change."""
OTHER_FILES = {
    "python-docs": "requirements/ci-lint.txt",
    "python-types": "requirements/ci-dev.txt",
    "notebook-types": "requirements/ci-dev.txt",
}


def pyproject(
    exclude: tuple[str, ...] = EXCLUDED,
    *,
    ruff: int = 88,
    comment: str = "the project",
    extra: str = "",
) -> bytes:
    """Return a small ``pyproject.toml`` with every table the rules look at.

    Parameters
    ----------
    exclude : tuple[str, ...]
        The setuptools package exclusion.
    ruff : int
        Ruff's line length, an influencing setting of python-docs.
    comment : str
        The leading comment, which no parser sees.
    extra : str
        TOML appended at the end.

    Returns
    -------
    bytes
        The file's bytes.
    """
    lines = (
        f"# {comment}",
        "[project]",
        'name = "director-ai"',
        'requires-python = ">=3.11"',
        "",
        "[tool.setuptools.packages.find]",
        'where = ["src"]',
        f"exclude = {json.dumps(list(exclude))}",
        "",
        "[tool.ruff]",
        f"line-length = {ruff}",
        "",
        "[tool.mypy]",
        "strict = true",
        'mypy_path = "src"',
        extra,
    )
    return ("\n".join(lines) + "\n").encode()


BEFORE = pyproject()
AFTER = pyproject(WIDER)
LATER = pyproject(WIDEST)


def sha(data: bytes) -> str:
    """Return the SHA-256 hex digest of bytes.

    Parameters
    ----------
    data : bytes
        The bytes.

    Returns
    -------
    str
        The digest.
    """
    return hashlib.sha256(data).hexdigest()


def preimage(profile: str, digest: str) -> dict[str, object]:
    """Return an identity preimage that binds ``pyproject.toml`` by digest.

    Parameters
    ----------
    profile : str
        One of :data:`PROFILES`.
    digest : str
        The ``pyproject.toml`` digest.

    Returns
    -------
    dict[str, object]
        A preimage shaped like the profile adapter's.
    """
    member = next(iter(CONFIGURATION_FIELDS[profile])).split("/")[1]
    value: dict[str, object] = {
        "profile": profile,
        member: {PYPROJECT: digest, OTHER_FILES[profile]: "b" * 64},
    }
    if profile != "python-docs":
        value["configuration"] = "[mypy]\nstrict = True\n"
    return value


def configuration_entry(
    profile: str = "python-docs",
    *,
    before: bytes = BEFORE,
    after: bytes = AFTER,
    pointers: tuple[str, ...] = (EXCLUDE_POINTER,),
    rigor: str | None = "8" * 64,
) -> dict[str, object]:
    """Return a complete valid configuration entry, as decoded JSON.

    Parameters
    ----------
    profile : str
        One of :data:`PROFILES`.
    before, after : bytes
        The ``pyproject.toml`` bytes the two identities bind.
    pointers : tuple[str, ...]
        The recorded TOML pointers.
    rigor : str | None
        The statistics approval digest, or None.

    Returns
    -------
    dict[str, object]
        A fresh entry that callers may mutate.
    """
    old, new = preimage(profile, sha(before)), preimage(profile, sha(after))
    return copy.deepcopy(
        {
            "profile": profile,
            "reason": CONFIGURATION_REASON,
            "old_identity": preimage_identity(old),
            "new_identity": preimage_identity(new),
            "old_preimage": old,
            "new_preimage": new,
            "changed_fields": [
                {
                    "path": next(iter(CONFIGURATION_FIELDS[profile])),
                    "old": sha(before),
                    "new": sha(after),
                }
            ],
            "verification_records": {},
            "evidence_digest": "3" * 64,
            "message_only": [],
            "drops": [],
            "canary": None,
            "packet_digest": "6" * 64,
            "authority_approval_digest": "7" * 64,
            "rigor_approval_digest": rigor,
            "configuration_paths": {PYPROJECT: list(pointers)},
        }
    )


def with_field(entry: dict[str, object], pointer: str, value: str) -> dict[str, object]:
    """Return an entry whose new preimage also changes one more leaf.

    Parameters
    ----------
    entry : dict[str, object]
        A configuration entry.
    pointer : str
        An RFC 6901 pointer into the preimage, its tokens escaped.
    value : str
        The leaf's new value.

    Returns
    -------
    dict[str, object]
        A copy whose new preimage, new identity and declared fields agree.
    """
    changed = copy.deepcopy(entry)
    tokens = [
        token.replace("~1", "/").replace("~0", "~") for token in pointer.split("/")[1:]
    ]
    new = cast("dict[str, object]", changed["new_preimage"])
    old: object = changed["old_preimage"]
    for token in tokens[:-1]:
        new = cast("dict[str, object]", new.setdefault(token, {}))
        old = old.get(token) if isinstance(old, dict) else None
    field: dict[str, object] = {"path": pointer, "new": value}
    if isinstance(old, dict) and tokens[-1] in old:
        field["old"] = old[tokens[-1]]
    new[tokens[-1]] = value
    cast("list[object]", changed["changed_fields"]).append(field)
    changed["new_identity"] = preimage_identity(
        cast("dict[str, object]", changed["new_preimage"])
    )
    return changed
