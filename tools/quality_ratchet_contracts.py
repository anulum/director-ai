# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native quality snapshot contracts
"""Validate source-bound native diagnostics before ratchet comparison."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

SCHEMA = "quality-ratchet.snapshot.v2"


def repository_path(value: object) -> str:
    """Validate one canonical repository-relative POSIX source path.

    Parameters
    ----------
    value : object
        Decoded path, without aliases, traversal or platform separators.

    Returns
    -------
    str
        Original canonical relative path.

    Raises
    ------
    ValueError
        If the value is not a canonical relative source path.
    """
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ("\\", ":"))
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("Invalid repository source path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Invalid repository source path")
    if value == ".":
        raise ValueError("Invalid repository source path")
    return value


def sha256_digest(value: object) -> str:
    """Validate a lowercase SHA-256 source or native-profile identity.

    Parameters
    ----------
    value : object
        Decoded identity.

    Returns
    -------
    str
        Exactly 64 hexadecimal characters.

    Raises
    ------
    ValueError
        If the identity is absent or malformed.
    """
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid SHA-256 identity")
    return value


def _mapping(value: object) -> dict[str, object]:
    """Require a decoded object with string keys, retaining original values."""
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("Expected a native quality object")
    return {k: v for k, v in value.items()}


def _text(value: object) -> str:
    """Require nonempty native text without a NUL byte."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("Expected nonempty native diagnostic text")
    return value


def _position(value: object) -> int:
    """Require a positive native position, excluding Boolean integers."""
    if type(value) is not int or value < 1:
        raise ValueError("Invalid native diagnostic position")
    return value


@dataclass(frozen=True)
class Finding:
    """Identify one native diagnostic without collapsing its location or text.

    Attributes
    ----------
    path : str
        Canonical source path present in the snapshot's source inventory.
    rule : str
        Native rule or diagnostic category.
    line : int
        One-based native source line.
    column : int | None
        One-based native source column, or None when the compiler omits it.
    message : str
        Exact original diagnostic text, without presentation colouring.
    cell : int | None
        One-based notebook cell identity, or None for ordinary source files.
    """

    path: str
    rule: str
    line: int
    column: int | None
    message: str
    cell: int | None = None


@dataclass(frozen=True)
class Snapshot:
    """Bind complete native findings to source and effective profile identities.

    Attributes
    ----------
    profile : str
        Stable native profile name.
    identity : str
        Digest of effective tool, configuration, dependencies and scope inputs.
    sources : dict[str, str]
        Complete inventoried source paths mapped to their byte digests.
    findings : tuple[Finding, ...]
        Original individual diagnostics, including paths and positions.
    complete : bool
        Native adapter's analysis-completion result; false never qualifies.
    advisories : tuple[str, ...]
        Validated native configuration information, separate from error debt.
    """

    profile: str
    identity: str
    sources: dict[str, str]
    findings: tuple[Finding, ...]
    complete: bool
    advisories: tuple[str, ...] = ()


def parse_snapshot(value: object) -> Snapshot:
    """Validate the exact native snapshot envelope and source-bound findings.

    Parameters
    ----------
    value : object
        Decoded native snapshot. Unknown fields and duplicated findings refuse.

    Returns
    -------
    Snapshot
        Validated carrier; completion still requires the native adapter and
        comparison against the trusted original baseline.

    Raises
    ------
    ValueError
        If the schema, sources, identities or diagnostic members are invalid.
    """
    data = _mapping(value)
    if (
        set(data)
        != {
            "schema",
            "profile",
            "identity",
            "sources",
            "findings",
            "complete",
            "advisories",
        }
        or data["schema"] != SCHEMA
    ):
        raise ValueError("Invalid native quality snapshot schema")
    profile = _text(data["profile"])
    if re.fullmatch(r"[a-z][a-z0-9-]*", profile) is None:
        raise ValueError("Invalid native quality profile name")
    if type(data["complete"]) is not bool:
        raise ValueError("Invalid native analysis completion value")
    sources = {
        repository_path(path): sha256_digest(digest)
        for path, digest in _mapping(data["sources"]).items()
    }
    if not sources:
        raise ValueError("Native quality source inventory is empty")
    raw_advisories = data["advisories"]
    if not isinstance(raw_advisories, list):
        raise ValueError("Expected original native advisories")
    advisories = tuple(_text(advisory) for advisory in raw_advisories)
    if len(set(advisories)) != len(advisories):
        raise ValueError("Repeated original native advisory identity")
    raw_findings = data["findings"]
    if not isinstance(raw_findings, list):
        raise ValueError("Expected original native findings")
    findings = []
    for raw in raw_findings:
        item = _mapping(raw)
        if set(item) != {"path", "rule", "line", "column", "message", "cell"}:
            raise ValueError("Invalid native diagnostic members")
        finding = Finding(
            repository_path(item["path"]),
            _text(item["rule"]),
            _position(item["line"]),
            None if item["column"] is None else _position(item["column"]),
            _text(item["message"]),
            None if item["cell"] is None else _position(item["cell"]),
        )
        if finding.path not in sources:
            raise ValueError("Native diagnostic lies outside the source inventory")
        findings.append(finding)
    if len(set(findings)) != len(findings):
        raise ValueError("Repeated original native diagnostic identity")
    return Snapshot(
        profile,
        sha256_digest(data["identity"]),
        sources,
        tuple(findings),
        data["complete"],
        advisories,
    )


def snapshot_payload(snapshot: Snapshot) -> dict[str, object]:
    """Serialize individual original findings without count-only projection.

    Parameters
    ----------
    snapshot : Snapshot
        Validated native carrier.

    Returns
    -------
    dict[str, object]
        Stable JSON-compatible carrier retaining source, position and cell IDs.
    """
    return {
        "schema": SCHEMA,
        "profile": snapshot.profile,
        "identity": snapshot.identity,
        "sources": dict(snapshot.sources),
        "complete": snapshot.complete,
        "advisories": list(snapshot.advisories),
        "findings": [
            {
                "path": f.path,
                "rule": f.rule,
                "line": f.line,
                "column": f.column,
                "message": f.message,
                "cell": f.cell,
            }
            for f in sorted(
                snapshot.findings,
                key=lambda f: (
                    f.path,
                    f.line,
                    -1 if f.column is None else f.column,
                    f.cell or 0,
                    f.rule,
                    f.message,
                ),
            )
        ],
    }


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse repeated JSON object members before they can overwrite identities."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Repeated native quality JSON member")
        result[key] = value
    return result


def decode_quality_json(data: bytes) -> object:
    """Decode quality JSON without allowing duplicate object keys.

    Parameters
    ----------
    data : bytes
        Original JSON bytes.

    Returns
    -------
    object
        Original decoded carrier, before schema validation.

    Raises
    ------
    ValueError
        If encoding, JSON or member identity is invalid.
    """
    value: object = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs)
    return value
