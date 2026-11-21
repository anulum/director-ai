# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Custody Tier Journal

"""Custody tier journal entries, and the scoped pins derived from them.

Each pin act is a tier journal entry, ``tier_pinned`` or ``tier_unpinned``,
with a ``pin_scope``. A digest D is pinned in scope S while the latest entry
for ``(D, S)`` is ``tier_pinned``; one scope's unpin never unpins another.
The pin set is derived from the journal only, never kept separately.

A ``tier_released`` entry resets its digest's pins: only pin entries after
the digest's last release count. A pinned digest is never released, so a
release means every scope had already unpinned it, or the journal broke that
rule; either way no earlier pin may count again. A recall does not revive
them: a recalled digest needs a new ``tier_pinned`` entry.

The journal is per programme, starts at ``TIER_JOURNAL_ORIGIN_SEQ`` and its
``tier_seq`` rises by exactly one, so a missing or repeated entry anywhere
before the tail is refused, its head included. Only entries at or below the
decision's tier position count, and a journal that ends before that
position is refused, since it provably misses entries the decision saw. A
truncated tail is not detectable here: that the journal passed in is
complete up to that position, and that the position is the decision's true
one, stay custody trusts, as they are for the look journal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_choice,
    require_int,
    require_sha256,
    require_text,
    require_utc_timestamp,
)
from .errors import CustodyRecordError, UnpinnedPromotionDependencyError
from .look_admission import CustodyProgrammeRoles
from .promotion_records import PromotionDependencyPins, validate_dependency_pins

TierAction = Literal[
    "tier_pinned",
    "tier_unpinned",
    "tier_release_staged",
    "tier_released",
    "tier_recalled",
]
TIER_ACTIONS: frozenset[str] = frozenset(
    {
        "tier_pinned",
        "tier_unpinned",
        "tier_release_staged",
        "tier_released",
        "tier_recalled",
    }
)
PIN_ACTIONS: frozenset[str] = frozenset({"tier_pinned", "tier_unpinned"})
"""The only actions that take a ``pin_scope``."""
TIER_JOURNAL_ORIGIN_SEQ = 1
"""The ``tier_seq`` of a programme's first tier journal entry."""

E = CustodyRecordError


@dataclass(frozen=True, slots=True)
class CustodyTierJournalEntry:
    """One entry of a programme's custody tier journal.

    Attributes
    ----------
    schema : Literal["custody-tier-journal-entry.v1"]
        Record schema.
    programme : str
        The programme whose journal holds the entry.
    tier_seq : int
        Position in the journal, rising by one from the origin.
    action : TierAction
        The tier act.
    digest : str
        Content digest the act names.
    pin_scope : str
        The slot or service a pin act is scoped to; empty for other acts.
    role : str
        The role that entered the act.
    holder : str
        The holder of that role.
    reason : str
        Why the act was entered.
    recorded_at_utc : str
        Informational time; order is ``tier_seq``.
    """

    schema: Literal["custody-tier-journal-entry.v1"]
    programme: str
    tier_seq: int
    action: TierAction
    digest: str
    pin_scope: str
    role: str
    holder: str
    reason: str
    recorded_at_utc: str


def validate_tier_journal_entry(
    entry: CustodyTierJournalEntry,
) -> CustodyTierJournalEntry:
    """Check one tier journal entry's shape.

    Parameters
    ----------
    entry : CustodyTierJournalEntry
        The entry to check.

    Returns
    -------
    CustodyTierJournalEntry
        The same entry.

    Raises
    ------
    CustodyRecordError
        If a pin act has no scope, another act has one, or a field is
        malformed.
    MalformedDigestError
        If the digest is malformed.
    """
    if not isinstance(entry, CustodyTierJournalEntry):
        raise E("a CustodyTierJournalEntry is required")
    require_choice(entry.schema, "schema", {"custody-tier-journal-entry.v1"}, E)
    require_text(entry.programme, "programme", E)
    require_int(entry.tier_seq, "tier_seq", E, minimum=TIER_JOURNAL_ORIGIN_SEQ)
    action = require_choice(entry.action, "action", TIER_ACTIONS, E)
    require_sha256(entry.digest, "digest")
    if action in PIN_ACTIONS:
        require_text(entry.pin_scope, "pin_scope", E)
    elif entry.pin_scope != "":
        raise E("only a pin or unpin act names a pin_scope")
    for name in ("role", "holder", "reason"):
        require_text(getattr(entry, name), name, E)
    require_utc_timestamp(entry.recorded_at_utc, "recorded_at_utc", E)
    return entry


def check_tier_journal_order(
    tier_journal: Sequence[CustodyTierJournalEntry],
) -> Sequence[CustodyTierJournalEntry]:
    """Check that a tier journal is one programme's, gap-free from its origin.

    Parameters
    ----------
    tier_journal : Sequence[CustodyTierJournalEntry]
        The programme's tier journal, in order.

    Returns
    -------
    Sequence[CustodyTierJournalEntry]
        The same journal.

    Raises
    ------
    CustodyRecordError
        If an entry is malformed or belongs to another programme, or the
        journal does not start at ``TIER_JOURNAL_ORIGIN_SEQ`` and rise by
        exactly one.
    """
    if tier_journal and tier_journal[0].tier_seq != TIER_JOURNAL_ORIGIN_SEQ:
        raise E(f"the tier journal must start at tier_seq {TIER_JOURNAL_ORIGIN_SEQ}")
    for position, entry in enumerate(tier_journal):
        validate_tier_journal_entry(entry)
        if entry.programme != tier_journal[0].programme:
            raise E("a tier journal entry belongs to another programme")
        if entry.tier_seq != tier_journal[0].tier_seq + position:
            raise E("tier_seq must rise by exactly one")
    return tier_journal


def _deciding_pin_entries(
    tier_journal: Sequence[CustodyTierJournalEntry], through_seq: int
) -> dict[tuple[str, str], CustodyTierJournalEntry]:
    """Return the latest pin act per ``(digest, scope)`` since its last release."""
    position = require_int(through_seq, "through_seq", E, minimum=0)
    check_tier_journal_order(tier_journal)
    if position > (tier_journal[-1].tier_seq if tier_journal else 0):
        raise E("the tier journal ends before the decision's position")
    counted = [entry for entry in tier_journal if entry.tier_seq <= through_seq]
    deciding: dict[tuple[str, str], CustodyTierJournalEntry] = {}
    for entry in counted:
        if entry.action in PIN_ACTIONS:
            deciding[(entry.digest, entry.pin_scope)] = entry
        elif entry.action == "tier_released":
            for key in [key for key in deciding if key[0] == entry.digest]:
                del deciding[key]
    return deciding


def active_pins(
    tier_journal: Sequence[CustodyTierJournalEntry], *, through_seq: int
) -> Mapping[str, frozenset[str]]:
    """Return each scope's pinned digests at a tier position.

    Parameters
    ----------
    tier_journal : Sequence[CustodyTierJournalEntry]
        The programme's tier journal, in order.
    through_seq : int
        The tier position of the decision; later entries do not count.

    Returns
    -------
    Mapping[str, frozenset[str]]
        Scope to the digests whose latest entry in it is ``tier_pinned``.

    Raises
    ------
    CustodyRecordError
        If the journal or the position is malformed, or the journal ends
        before the position.
    """
    scopes: dict[str, set[str]] = {}
    for (digest, scope), entry in _deciding_pin_entries(
        tier_journal, through_seq
    ).items():
        if entry.action == "tier_pinned":
            scopes.setdefault(scope, set()).add(digest)
    return {scope: frozenset(digests) for scope, digests in scopes.items()}


def require_dependencies_pinned(
    pins: PromotionDependencyPins,
    tier_journal: Sequence[CustodyTierJournalEntry],
    *,
    through_seq: int,
    roles: CustodyProgrammeRoles,
) -> PromotionDependencyPins:
    """Refuse a promotion whose dependencies are not pinned in its scope (Q-9).

    Parameters
    ----------
    pins : PromotionDependencyPins
        The promotion's dependency digests and its pin scope.
    tier_journal : Sequence[CustodyTierJournalEntry]
        The programme's tier journal, in order.
    through_seq : int
        The tier position of the decision.
    roles : CustodyProgrammeRoles
        The programme's configured roles.

    Returns
    -------
    PromotionDependencyPins
        The same pins.

    Raises
    ------
    UnpinnedPromotionDependencyError
        If a family contract, secondary contract or witness digest is not
        pinned in the promotion's scope at that position, or its pin was
        entered by a role other than the deployment operator.
    CustodyRecordError
        If the journal, the position or the roles are malformed, or the
        journal is another programme's.
    PromotionDecisionError
        If the pins are malformed.
    """
    validate_dependency_pins(pins)
    operator = require_text(
        roles.deployment_operator_role, "deployment_operator_role", E
    )
    if tier_journal and tier_journal[0].programme != roles.programme:
        raise E("the tier journal belongs to another programme")
    deciding = _deciding_pin_entries(tier_journal, through_seq)
    for digest in (
        pins.family_contract_digest,
        *pins.secondary_family_contract_digests,
        *pins.witness_and_parity_digests,
    ):
        entry = deciding.get((digest, pins.pin_scope))
        if entry is None or entry.action != "tier_pinned":
            raise UnpinnedPromotionDependencyError(
                f"dependency {digest[:12]} is not pinned in scope {pins.pin_scope!r}"
            )
        if entry.role != operator:
            raise UnpinnedPromotionDependencyError(
                f"the pin of {digest[:12]} was not entered by the deployment operator"
            )
    return pins
