# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — shared builders for the tool succession tests
"""Build valid transition entries and snapshots that individual tests then break."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping

from tools.quality_ratchet_contracts import Finding, Snapshot
from tools.quality_ratchet_succession import TRANSITIONS_SCHEMA, preimage_identity

ADAPTER = "tools/quality_ratchet_typescript.py"
OLD_PREIMAGE: dict[str, object] = {
    "profile": "typescript-docs",
    "native": {"node": "26.13.0", "typescript": "6.0.3"},
    "engine": {ADAPTER: "a" * 64, "packages/vercel-ai/package-lock.json": "b" * 64},
    "roots": ["packages", "tools"],
}
ENGINE = "e" * 64
NEW_PREIMAGE: dict[str, object] = {
    "profile": "typescript-docs",
    "native": {"node": "26.13.1", "typescript": "6.0.3"},
    "engine": {ADAPTER: "c" * 64, "packages/vercel-ai/package-lock.json": "b" * 64},
    "roots": ["packages", "tools"],
    "artefacts": {"engine": {"linux-x64": {"binary": ENGINE, "tarball": "f" * 64}}},
}
OLD_IDENTITY = preimage_identity(OLD_PREIMAGE)
NEW_IDENTITY = preimage_identity(NEW_PREIMAGE)
RULES = ("declaration-documentation", "source-directive")
CANARY_KEYS = [
    {
        "path": "tests/fixtures/canary.ts",
        "rule": RULES[0],
        "line": 3,
        "column": 1,
        "cell": None,
    },
    {
        "path": "tests/fixtures/canary.ts",
        "rule": RULES[1],
        "line": 5,
        "column": 1,
        "cell": None,
    },
]


def members(record: Mapping[str, object], name: str) -> list[dict[str, object]]:
    """Return a record's list of objects, narrowed to typed mappings.

    Parameters
    ----------
    record : Mapping[str, object]
        A decoded record.
    name : str
        The list member.

    Returns
    -------
    list[dict[str, object]]
        Fresh copies of its items.
    """
    value = record[name]
    assert isinstance(value, list)
    items = []
    for item in value:
        assert isinstance(item, dict)
        items.append(dict(item))
    return items


def mapping(record: Mapping[str, object], name: str) -> dict[str, object]:
    """Return a record's object member, narrowed to a typed mapping.

    Parameters
    ----------
    record : Mapping[str, object]
        A decoded record.
    name : str
        The object member.

    Returns
    -------
    dict[str, object]
        A fresh copy.
    """
    value = record[name]
    assert isinstance(value, dict)
    return dict(value)


def entry() -> dict[str, object]:
    """Return a complete valid transition entry, as decoded JSON.

    Returns
    -------
    dict[str, object]
        A fresh copy that callers may mutate.
    """
    return copy.deepcopy(
        {
            "profile": "typescript-docs",
            "reason": "security_release",
            "old_identity": OLD_IDENTITY,
            "new_identity": NEW_IDENTITY,
            "old_preimage": OLD_PREIMAGE,
            "new_preimage": NEW_PREIMAGE,
            "changed_fields": [
                {
                    "path": "/artefacts/engine/linux-x64/binary",
                    "new": ENGINE,
                },
                {
                    "path": "/artefacts/engine/linux-x64/tarball",
                    "new": "f" * 64,
                },
                {
                    "path": "/engine/tools~1quality_ratchet_typescript.py",
                    "old": "a" * 64,
                    "new": "c" * 64,
                },
                {"path": "/native/node", "old": "26.13.0", "new": "26.13.1"},
            ],
            "verification_records": {
                "/artefacts/engine/linux-x64/binary": "1" * 64,
                "/artefacts/engine/linux-x64/tarball": "2" * 64,
            },
            "evidence_digest": "3" * 64,
            "message_only": [],
            "drops": [],
            "canary": {
                "fixture_digest": "4" * 64,
                "platform": "linux-x64",
                "enabled_rules": list(RULES),
                "old_engine_digest": "5" * 64,
                "new_engine_digest": ENGINE,
                "old_findings": CANARY_KEYS,
                "new_findings": CANARY_KEYS,
            },
            "packet_digest": "6" * 64,
            "authority_approval_digest": "7" * 64,
            "rigor_approval_digest": "8" * 64,
        }
    )


def file_bytes(*entries: dict[str, object]) -> bytes:
    """Encode a transition file.

    Parameters
    ----------
    *entries : dict[str, object]
        The decoded entries, in order.

    Returns
    -------
    bytes
        The file's exact bytes.
    """
    return json.dumps(
        {"schema": TRANSITIONS_SCHEMA, "transitions": list(entries)}, sort_keys=True
    ).encode()


def snapshot(identity: str, *findings: Finding, digest: str = "a" * 64) -> Snapshot:
    """Return a typescript-docs snapshot over two sources.

    Parameters
    ----------
    identity : str
        The snapshot's identity.
    *findings : Finding
        Its findings.
    digest : str
        The byte digest of ``packages/a.ts``; ``packages/b.ts`` is fixed.

    Returns
    -------
    Snapshot
        A complete snapshot.
    """
    return Snapshot(
        "typescript-docs",
        identity,
        {"packages/a.ts": digest, "packages/b.ts": "b" * 64},
        tuple(findings),
        True,
    )


def finding(
    path: str = "packages/b.ts", rule: str = RULES[0], line: int = 2
) -> Finding:
    """Return one finding with a fixed message.

    Parameters
    ----------
    path : str
        Source path.
    rule : str
        Rule identifier.
    line : int
        Line.

    Returns
    -------
    Finding
        The finding.
    """
    return Finding(path, rule, line, 1, "requires explanatory JSDoc")
