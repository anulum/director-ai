# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — declared native tool succession records
"""Parse the append-only records that let a profile's tool identity change.

A native profile's identity binds its tool versions, engine sources and scope.
The ordinary comparison refuses any identity change. A transition record
declares one reviewed change: both identity preimages, which must hash to the
two identities, the exact fields that differ, a reason from a closed set, the
signature-verification record of every changed tool artefact, a seeded canary
run under both identities, any findings dropped by a documented analyser fix,
and two approvals of the evidence packet, one from the ratchet authority and
one from the statistics reviewer.

Trust: until a profile's first transition, its tool identity rests on the
version strings its adapter records; artefact digests enter the preimage at
that first transition. The candidate's own ratchet code checks the
candidate; the authority is the base commit's baseline findings, the base-code
differential in the packet and the review of the code. The comparison,
gate, baseline and command-line modules are no profile's engine input, so a
change to them, this module included, is caught only by exact-tree review.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

TRANSITIONS_PATH = "tools/quality_ratchet_transitions.json"
TRANSITIONS_SCHEMA = "quality-ratchet.transitions.v1"
REASONS = frozenset(
    {"security_release", "compatibility_release", "engine_fix", "engine_hardening"}
)
PREIMAGE_PROFILES = frozenset(
    {
        "python-docs",
        "python-types",
        "notebook-types",
        "go-docs",
        "typescript-docs",
        "protobuf-docs",
        "lean-docs",
        "rust-docs",
        "julia-docs",
    }
)
"""Profiles whose identity is the SHA-256 of their sorted JSON preimage.

All nine required adapters compute their identity this way; the landing
packet's preimage census recovered each preimage and matched its identity.
A profile still needs its canary runner before any transition can apply.
"""
ARTEFACTS_POINTER = "/artefacts/"
"""Prefix of every preimage leaf that binds a verified tool artefact."""
ENGINE_FIELD = "/artefacts/engine/{platform}/binary"
"""JSON Pointer of the verified engine binary digest for one platform."""

ENTRY_MEMBERS = frozenset(
    {
        "profile",
        "reason",
        "old_identity",
        "new_identity",
        "old_preimage",
        "new_preimage",
        "changed_fields",
        "verification_records",
        "evidence_digest",
        "message_only",
        "drops",
        "canary",
        "packet_digest",
        "authority_approval_digest",
        "rigor_approval_digest",
    }
)
KEY_MEMBERS = frozenset({"path", "rule", "line", "column", "cell"})
CANARY_MEMBERS = frozenset(
    {
        "fixture_digest",
        "platform",
        "enabled_rules",
        "old_engine_digest",
        "new_engine_digest",
        "old_findings",
        "new_findings",
    }
)


@dataclass(frozen=True, order=True)
class FindingKey:
    """Identify a native finding without its message text.

    Attributes
    ----------
    path : str
        Canonical source path.
    rule : str
        Native rule identifier.
    line : int
        One-based line.
    column : int | None
        One-based column, or None when the tool omits it.
    cell : int | None
        One-based notebook cell, or None for ordinary sources.
    """

    path: str
    rule: str
    line: int
    column: int | None
    cell: int | None


class _Absent:
    """Mark a preimage leaf that does not exist, distinct from JSON null."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the marker's name."""
        return "ABSENT"


ABSENT: object = _Absent()
"""The value of an absent leaf; a record states absence by omitting the member."""


@dataclass(frozen=True)
class FieldChange:
    """One differing leaf of the two identity preimages.

    Attributes
    ----------
    path : str
        RFC 6901 JSON Pointer of the leaf within the preimage.
    old : object
        The leaf's value before, or :data:`ABSENT` when the leaf is new.
    new : object
        The leaf's value after, or :data:`ABSENT` when the leaf is removed.
    """

    path: str
    old: object
    new: object


@dataclass(frozen=True)
class Drop:
    """One finding an analyser fix removes from a byte-identical source.

    Attributes
    ----------
    key : FindingKey
        The dropped finding.
    changelog_digest : str
        SHA-256 of the vendor changelog entry naming the fix.
    """

    key: FindingKey
    changelog_digest: str


@dataclass(frozen=True)
class CanaryEvidence:
    """The seeded known-positive canary run under both identities.

    Attributes
    ----------
    fixture_digest : str
        Digest of the canary fixture's file set.
    platform : str
        The platform both runs executed on, for example ``linux-x64``.
    enabled_rules : tuple[str, ...]
        Every rule the profile enforces; each must fire on the canaries.
    old_engine_digest, new_engine_digest : str
        Digest of the engine binary each run actually executed.
    old_findings : tuple[FindingKey, ...]
        Canary findings under the old identity.
    new_findings : tuple[FindingKey, ...]
        Canary findings under the new identity.
    """

    fixture_digest: str
    platform: str
    enabled_rules: tuple[str, ...]
    old_engine_digest: str
    new_engine_digest: str
    old_findings: tuple[FindingKey, ...]
    new_findings: tuple[FindingKey, ...]


@dataclass(frozen=True)
class Transition:
    """One declared, reviewed change of a profile's tool identity.

    Attributes
    ----------
    profile : str
        The native profile.
    reason : str
        One of :data:`REASONS`.
    old_identity, new_identity : str
        The identities before and after.
    old_preimage, new_preimage : dict[str, object]
        The JSON preimages hashing to those identities.
    changed_fields : tuple[FieldChange, ...]
        Exactly the leaves in which the preimages differ.
    verification_records : dict[str, str]
        Signature-verification record digest per changed artefact field.
    evidence_digest : str
        Digest of the vendor's release notes or the engine change record.
    message_only : tuple[FindingKey, ...]
        Findings on unchanged sources whose message text alone changed.
    drops : tuple[Drop, ...]
        Findings removed from unchanged sources by a documented fix.
    canary : CanaryEvidence
        The canary run under both identities.
    packet_digest : str
        Digest of the approved transition evidence packet.
    authority_approval_digest, rigor_approval_digest : str
        The two approval records, each bound to the packet.
    """

    profile: str
    reason: str
    old_identity: str
    new_identity: str
    old_preimage: dict[str, object]
    new_preimage: dict[str, object]
    changed_fields: tuple[FieldChange, ...]
    verification_records: dict[str, str]
    evidence_digest: str
    message_only: tuple[FindingKey, ...]
    drops: tuple[Drop, ...]
    canary: CanaryEvidence
    packet_digest: str
    authority_approval_digest: str
    rigor_approval_digest: str


def preimage_identity(preimage: dict[str, object]) -> str:
    """Return the identity an adapter computes from its preimage.

    Parameters
    ----------
    preimage : dict[str, object]
        Decoded preimage.

    Returns
    -------
    str
        SHA-256 of the sorted-key JSON encoding.
    """
    return hashlib.sha256(json.dumps(preimage, sort_keys=True).encode()).hexdigest()


def pointer_token(name: str) -> str:
    """Escape one member name as an RFC 6901 JSON Pointer reference token.

    Parameters
    ----------
    name : str
        A preimage member name, which may hold any character, such as the
        ``/`` and ``.`` of the file paths that key engine digests.

    Returns
    -------
    str
        The name with ``~`` written ``~0`` and ``/`` written ``~1``.
    """
    return name.replace("~", "~0").replace("/", "~1")


def pointer_leaf(preimage: dict[str, object], pointer: str) -> object:
    """Return the value a JSON Pointer names in a preimage, or None if absent.

    Parameters
    ----------
    preimage : dict[str, object]
        Decoded preimage.
    pointer : str
        RFC 6901 pointer, empty or starting with ``/``.

    Returns
    -------
    object
        The value, or None when a member on the way is missing.
    """
    value: object = preimage
    for token in pointer.split("/")[1:]:
        name = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or name not in value:
            return None
        value = value[name]
    return value


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten nested objects into JSON Pointer leaves; lists stay whole leaves."""
    if not isinstance(value, dict) or not value:
        return {prefix: value}
    leaves: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("Invalid identity preimage member name")
        leaves.update(_leaves(item, f"{prefix}/{pointer_token(key)}"))
    return leaves


def canonical_leaf(value: object) -> str:
    """Return the form in which a leaf enters the identity.

    Parameters
    ----------
    value : object
        A decoded leaf, or :data:`ABSENT`.

    Returns
    -------
    str
        ``ABSENT`` for a missing leaf, otherwise its sorted JSON encoding. Two
        leaves differ exactly when their encodings do, so ``1``, ``true`` and
        ``1.0`` are three different values, as they are to the identity.
    """
    return "ABSENT" if value is ABSENT else json.dumps(value, sort_keys=True)


def preimage_changes(
    old: dict[str, object], new: dict[str, object]
) -> tuple[FieldChange, ...]:
    """Return every leaf in which two preimages differ, in path order.

    Parameters
    ----------
    old, new : dict[str, object]
        The two decoded preimages.

    Returns
    -------
    tuple[FieldChange, ...]
        Added, removed and changed leaves.
    """
    before, after = _leaves(old), _leaves(new)
    return tuple(
        FieldChange(path, before.get(path, ABSENT), after.get(path, ABSENT))
        for path in sorted(set(before) | set(after))
        if canonical_leaf(before.get(path, ABSENT))
        != canonical_leaf(after.get(path, ABSENT))
    )
