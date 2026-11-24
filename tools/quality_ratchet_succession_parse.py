# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — validation of declared native tool succession records
"""Validate the append-only transition file before any comparison uses it.

Every refusal is raised as :class:`ValueError` with a reason naming the
member at fault. Parsing alone grants nothing: a record applies only when the
base snapshot's identity and the identity the candidate's adapter recomputes
match it (:mod:`tools.quality_ratchet_succession_check`).

A ``configuration_change`` entry has no canary, no artefact, no drop and no
reworded message. Its changed fields are configuration-file digests only, and
it records the TOML pointers it changes, each on the profile's reviewed
non-influencing list; the checks recompute them from the files. Its statistics
approval may be absent only when every pointer it records was recorded by an
earlier configuration entry of the same file, never on a pointer's first use.
"""

from __future__ import annotations

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_configuration import (
    CONFIGURATION_FIELDS,
    NON_INFLUENCING,
    under,
)
from tools.quality_ratchet_contracts import (
    decode_quality_json,
    repository_path,
    sha256_digest,
)
from tools.quality_ratchet_succession import (
    ABSENT,
    ARTEFACTS_POINTER,
    CANARY_MEMBERS,
    CONFIGURATION_MEMBERS,
    CONFIGURATION_REASON,
    ENGINE_FIELD,
    ENTRY_MEMBERS,
    KEY_MEMBERS,
    PREIMAGE_PROFILES,
    REASONS,
    TRANSITIONS_SCHEMA,
    CanaryEvidence,
    Drop,
    FieldChange,
    FindingKey,
    Transition,
    canonical_leaf,
    pointer_leaf,
    preimage_changes,
    preimage_identity,
)


def _line(value: object) -> int:
    """Return a required one-based position."""
    if type(value) is not int or value < 1:
        raise ValueError("Invalid native finding key position")
    return value


def _optional(value: object) -> int | None:
    """Return an optional one-based position."""
    return None if value is None else _line(value)


def parse_finding_key(value: object) -> FindingKey:
    """Validate one message-free finding key.

    Parameters
    ----------
    value : object
        Decoded key with exactly path, rule, line, column and cell.

    Returns
    -------
    FindingKey
        The validated key.

    Raises
    ------
    ValueError
        If a member is missing, unknown or malformed.
    """
    item = object_mapping(value)
    if set(item) != KEY_MEMBERS:
        raise ValueError("Invalid native finding key members")
    rule = item["rule"]
    if not isinstance(rule, str) or not rule:
        raise ValueError("Invalid native finding key rule")
    return FindingKey(
        repository_path(item["path"]),
        rule,
        _line(item["line"]),
        _optional(item["column"]),
        _optional(item["cell"]),
    )


def _keys(value: object, member: str, *, unique: bool = True) -> tuple[FindingKey, ...]:
    """Parse a list of finding keys in sorted order; a multiset unless unique.

    Canary findings are a multiset: one source position may carry two findings
    with different messages, and each occurrence counts.
    """
    if not isinstance(value, list):
        raise ValueError(f"Transition member {member} must be a list")
    keys = tuple(parse_finding_key(item) for item in value)
    if unique and len(set(keys)) != len(keys):
        raise ValueError(f"Transition member {member} repeats a finding")
    return tuple(sorted(keys))


def _preimage(value: object, identity: str, member: str) -> dict[str, object]:
    """Require a nonempty preimage that hashes to the identity it claims."""
    preimage = object_mapping(value)
    if not preimage:
        raise ValueError(f"Transition {member} is empty")
    if preimage_identity(preimage) != identity:
        raise ValueError(f"Transition {member} does not hash to its identity")
    return preimage


def _canonical(changes: tuple[FieldChange, ...]) -> list[tuple[str, str, str]]:
    """Return changes as their paths and canonical JSON values, for comparison."""
    return [(c.path, canonical_leaf(c.old), canonical_leaf(c.new)) for c in changes]


def _changes(value: object) -> tuple[FieldChange, ...]:
    """Parse the declared changed preimage leaves."""
    if not isinstance(value, list) or not value:
        raise ValueError("Transition changed_fields must be a nonempty list")
    changes = []
    for raw in value:
        item = object_mapping(raw)
        values = set(item) - {"path"}
        if (
            "path" not in item
            or not isinstance(item["path"], str)
            or not values
            or not values <= {"old", "new"}
        ):
            raise ValueError("Invalid transition changed field")
        changes.append(
            FieldChange(item["path"], item.get("old", ABSENT), item.get("new", ABSENT))
        )
    return tuple(changes)


def _canary(
    value: object, old: dict[str, object], new: dict[str, object]
) -> CanaryEvidence:
    """Parse the canary runs; both fire every enabled rule on one platform.

    The new identity's preimage must hold the executed engine's digest for the
    run's platform. The old one must too once a previous transition added it;
    on a profile's first transition it holds none, and the old run's digest
    is evidence only.
    """
    item = object_mapping(value)
    if set(item) != CANARY_MEMBERS:
        raise ValueError("Invalid transition canary members")
    platform = item["platform"]
    if not isinstance(platform, str) or not platform or {"/", "~"} & set(platform):
        raise ValueError("Invalid transition canary platform")
    field = ENGINE_FIELD.format(platform=platform)
    old_engine = sha256_digest(item["old_engine_digest"])
    new_engine = sha256_digest(item["new_engine_digest"])
    if pointer_leaf(new, field) != new_engine:
        raise ValueError("The new canary run's engine differs from its preimage")
    recorded = pointer_leaf(old, field)
    if recorded is not None and recorded != old_engine:
        raise ValueError("The old canary run's engine differs from its preimage")
    rules = item["enabled_rules"]
    if (
        not isinstance(rules, list)
        or not rules
        or any(not isinstance(rule, str) or not rule for rule in rules)
        or len(set(rules)) != len(rules)
    ):
        raise ValueError("Transition canary needs unique enabled rules")
    old_keys = _keys(item["old_findings"], "canary.old_findings", unique=False)
    new_keys = _keys(item["new_findings"], "canary.new_findings", unique=False)
    if old_keys != new_keys:
        raise ValueError("Transition canary findings differ between the identities")
    if {key.rule for key in new_keys} != set(rules):
        raise ValueError("Transition canary must fire exactly the enabled rules")
    return CanaryEvidence(
        sha256_digest(item["fixture_digest"]),
        platform,
        tuple(sorted(rules)),
        old_engine,
        new_engine,
        old_keys,
        new_keys,
    )


def _drops(value: object, rules: tuple[str, ...]) -> tuple[Drop, ...]:
    """Parse the declared drops; each names a changelog and a rule with a live canary."""
    if not isinstance(value, list):
        raise ValueError("Transition member drops must be a list")
    drops = []
    for raw in value:
        item = object_mapping(raw)
        if set(item) != KEY_MEMBERS | {"changelog_digest"}:
            raise ValueError("Invalid transition drop members")
        key = parse_finding_key({name: item[name] for name in KEY_MEMBERS})
        if key.rule not in rules:
            raise ValueError("A dropped finding's rule has no canary")
        drops.append(Drop(key, sha256_digest(item["changelog_digest"])))
    if len({drop.key for drop in drops}) != len(drops):
        raise ValueError("Transition member drops repeats a finding")
    return tuple(sorted(drops, key=lambda drop: drop.key))


def _configuration_paths(
    value: object, profile: str, changes: tuple[FieldChange, ...]
) -> dict[str, tuple[str, ...]]:
    """Parse a configuration entry's changed TOML pointers, per changed file."""
    fields = CONFIGURATION_FIELDS.get(profile, {})
    stray = [change.path for change in changes if change.path not in fields]
    if stray:
        raise ValueError(
            "A configuration transition changes a field outside its configuration "
            f"files: {stray[0]}"
        )
    files = {fields[change.path] for change in changes}
    item = object_mapping(value)
    if set(item) != files:
        raise ValueError("Configuration paths must name exactly the changed files")
    paths = {}
    for name in sorted(files):
        raw = item[name]
        if (
            not isinstance(raw, list)
            or any(not isinstance(pointer, str) for pointer in raw)
            or raw != sorted(set(raw))
            or any(not pointer.startswith("/") for pointer in raw)
        ):
            raise ValueError("Invalid configuration paths")
        allowed = NON_INFLUENCING.get(profile, {}).get(name, ())
        if any(not any(under(pointer, a) for a in allowed) for pointer in raw):
            raise ValueError(
                "A recorded configuration path is not on the non-influencing list"
            )
        paths[name] = tuple(raw)
    return paths


def _approvals(
    item: dict[str, object], *, optional_rigor: bool
) -> tuple[str, str, str | None]:
    """Parse the packet and approval digests; they must be distinct records."""
    packet = sha256_digest(item["packet_digest"])
    authority = sha256_digest(item["authority_approval_digest"])
    raw = item["rigor_approval_digest"]
    rigor = None if raw is None and optional_rigor else sha256_digest(raw)
    digests = [packet, authority, *([rigor] if rigor is not None else [])]
    if len(set(digests)) != len(digests):
        raise ValueError(
            "The packet and its two approvals must be distinct records"
            if rigor is not None
            else "The packet and its approval must be distinct records"
        )
    return packet, authority, rigor


def _configuration_transition(
    item: dict[str, object],
    profile: str,
    identities: tuple[str, str],
    preimages: tuple[dict[str, object], dict[str, object]],
    changes: tuple[FieldChange, ...],
) -> Transition:
    """Validate the members only a configuration entry has, and build it."""
    if (
        item["canary"] is not None
        or item["verification_records"] != {}
        or item["drops"] != []
        or item["message_only"] != []
    ):
        raise ValueError(
            "A configuration transition has no canary, artefact, drop or reworded "
            "message"
        )
    paths = _configuration_paths(item["configuration_paths"], profile, changes)
    packet, authority, rigor = _approvals(item, optional_rigor=True)
    return Transition(
        profile,
        CONFIGURATION_REASON,
        identities[0],
        identities[1],
        preimages[0],
        preimages[1],
        changes,
        {},
        sha256_digest(item["evidence_digest"]),
        (),
        (),
        None,
        packet,
        authority,
        rigor,
        paths,
    )


def parse_transition(value: object) -> Transition:
    """Validate one transition entry completely.

    Parameters
    ----------
    value : object
        One decoded entry of the transition file.

    Returns
    -------
    Transition
        The validated record.

    Raises
    ------
    ValueError
        If any member is missing, unknown or inconsistent: an adapter that does
        not hash a preimage, a preimage that does not hash to its identity,
        declared fields that differ from the preimage diff, a changed artefact
        without its verification record, a canary that fails, a drop without a
        changelog, or approvals that are not three distinct digests.
    """
    item = object_mapping(value)
    configuration = item.get("reason") == CONFIGURATION_REASON
    if set(item) != (CONFIGURATION_MEMBERS if configuration else ENTRY_MEMBERS):
        raise ValueError("Invalid transition entry members")
    profile, reason = item["profile"], item["reason"]
    if not isinstance(profile, str) or profile not in PREIMAGE_PROFILES:
        raise ValueError("The profile's adapter does not hash an identity preimage")
    if not isinstance(reason, str) or reason not in REASONS:
        raise ValueError("Unknown transition reason")
    old_identity = sha256_digest(item["old_identity"])
    new_identity = sha256_digest(item["new_identity"])
    if old_identity == new_identity:
        raise ValueError("A transition must change the identity")
    old = _preimage(item["old_preimage"], old_identity, "old_preimage")
    new = _preimage(item["new_preimage"], new_identity, "new_preimage")
    changes = tuple(sorted(_changes(item["changed_fields"]), key=lambda c: c.path))
    if _canonical(changes) != _canonical(preimage_changes(old, new)):
        raise ValueError("Declared changed fields differ from the preimage diff")
    if configuration:
        return _configuration_transition(
            item, profile, (old_identity, new_identity), (old, new), changes
        )
    records = {
        path: sha256_digest(digest)
        for path, digest in object_mapping(item["verification_records"]).items()
    }
    artefacts = {
        c.path
        for c in changes
        if c.path.startswith(ARTEFACTS_POINTER) and c.new is not ABSENT
    }
    if set(records) != artefacts:
        raise ValueError("Every changed artefact needs exactly one verification record")
    canary = _canary(item["canary"], old, new)
    approvals = _approvals(item, optional_rigor=False)
    return Transition(
        profile,
        reason,
        old_identity,
        new_identity,
        old,
        new,
        changes,
        records,
        sha256_digest(item["evidence_digest"]),
        _keys(item["message_only"], "message_only"),
        _drops(item["drops"], canary.enabled_rules),
        canary,
        *approvals,
    )


def parse_transitions(data: bytes) -> tuple[Transition, ...]:
    """Validate the whole transition file.

    Parameters
    ----------
    data : bytes
        Exact file bytes.

    Returns
    -------
    tuple[Transition, ...]
        Entries in file order.

    Raises
    ------
    ValueError
        If the envelope or any entry is invalid, or two entries leave the same
        identity of one profile.
    """
    value = object_mapping(decode_quality_json(data))
    if set(value) != {"schema", "transitions"} or value["schema"] != TRANSITIONS_SCHEMA:
        raise ValueError("Invalid native quality transition file")
    raw = value["transitions"]
    if not isinstance(raw, list):
        raise ValueError("Native quality transitions must be a list")
    entries = tuple(parse_transition(entry) for entry in raw)
    starts = [(entry.profile, entry.old_identity) for entry in entries]
    if len(set(starts)) != len(starts):
        raise ValueError("Two transitions leave the same profile identity")
    used: set[tuple[str, str, str]] = set()
    for entry in entries:
        recorded = {
            (entry.profile, name, pointer)
            for name, pointers in entry.configuration_paths.items()
            for pointer in pointers
        }
        if entry.rigor_approval_digest is None and not recorded <= used:
            raise ValueError(
                "A configuration path used for the first time needs the statistics "
                "approval"
            )
        used |= recorded
    return entries
