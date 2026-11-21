# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Tier Journal and Promotion Pin Tests

"""Scoped pins derived from the custody tier journal, and Q-9's pin check."""

from __future__ import annotations

from dataclasses import replace

import pytest

from director_ai.model_lifecycle import (
    PIN_ACTIONS,
    TIER_JOURNAL_ORIGIN_SEQ,
    CustodyProgrammeRoles,
    CustodyRecordError,
    CustodyTierJournalEntry,
    MalformedDigestError,
    PromotionDecisionError,
    PromotionDependencyPins,
    UnpinnedPromotionDependencyError,
    active_pins,
    canonical_json_bytes,
    check_tier_journal_order,
    from_canonical_json_bytes,
    require_dependencies_pinned,
    validate_tier_journal_entry,
)
from tests.model_lifecycle_support import PROGRAMME, UTC, digest_of

OPERATOR = "deployment-operator"
ROLES = CustodyProgrammeRoles(PROGRAMME, "custodian", "approver", OPERATOR)
FAMILY, SECOND, WITNESS = digest_of("family"), digest_of("second"), digest_of("witness")
DEPENDENCIES = (FAMILY, SECOND, WITNESS)
Act = tuple[str, str, str]


def entry(
    seq: int, action: str, digest: str, scope: str = "slot-a"
) -> CustodyTierJournalEntry:
    """Return a valid tier journal entry; only pin acts carry the scope."""
    return CustodyTierJournalEntry(
        "custody-tier-journal-entry.v1",
        PROGRAMME,
        seq,
        action,  # type: ignore[arg-type]
        digest,
        scope if action in PIN_ACTIONS else "",
        OPERATOR,
        "holder-1",
        "rollback target",
        UTC,
    )


def journal(*acts: Act) -> list[CustodyTierJournalEntry]:
    """Return ``acts`` as entries numbered from the origin."""
    return [entry(TIER_JOURNAL_ORIGIN_SEQ + i, *act) for i, act in enumerate(acts)]


def pinned(scope: str = "slot-a") -> tuple[Act, ...]:
    """Pin every dependency in ``scope``."""
    return tuple(("tier_pinned", digest, scope) for digest in DEPENDENCIES)


def pins(scope: str = "slot-a") -> PromotionDependencyPins:
    """Return the promotion's dependency pins in ``scope``."""
    return PromotionDependencyPins(
        "promotion-dependency-pins.v1", scope, FAMILY, (SECOND,), (WITNESS,)
    )


def check(
    tier: list[CustodyTierJournalEntry], *, through: int | None = None
) -> PromotionDependencyPins:
    """Run Q-9 at ``through``, by default the journal's last position."""
    position = len(tier) if through is None else through
    return require_dependencies_pinned(pins(), tier, through_seq=position, roles=ROLES)


class TestScopes:
    """CT-5a: a digest is pinned per scope; one scope's unpin leaves the others."""

    def test_pinned_in_scope_passes(self) -> None:
        """Every dependency pinned in the promotion's scope passes."""
        assert check(journal(*pinned())) == pins()

    def test_unpinned_in_scope_fails(self) -> None:
        """The latest entry decides: a pin followed by an unpin is not pinned."""
        with pytest.raises(UnpinnedPromotionDependencyError, match="not pinned"):
            check(journal(*pinned(), ("tier_unpinned", WITNESS, "slot-a")))

    def test_a_pin_after_an_unpin_counts(self) -> None:
        """An unpin followed by a pin is pinned again."""
        acts = (("tier_unpinned", WITNESS, "slot-a"), *pinned())
        assert check(journal(*acts)) == pins()

    def test_a_pin_in_another_scope_only_fails(self) -> None:
        """A digest pinned only in another scope does not count."""
        with pytest.raises(UnpinnedPromotionDependencyError, match="'slot-a'"):
            check(journal(*pinned("slot-b")))

    def test_another_scopes_unpin_leaves_this_scope_pinned(self) -> None:
        """TT-6b: pinned in two scopes, unpinned in one, still pinned in the other."""
        tier = journal(
            *pinned(),
            ("tier_pinned", WITNESS, "slot-b"),
            ("tier_unpinned", WITNESS, "slot-b"),
        )
        assert check(tier) == pins()
        assert active_pins(tier, through_seq=len(tier)) == {
            "slot-a": frozenset(DEPENDENCIES)
        }

    def test_active_pins_maps_each_scope(self) -> None:
        """Only latest pins count, per scope; an empty or zero view holds none."""
        tier = journal(
            *pinned(),
            ("tier_pinned", WITNESS, "slot-b"),
            ("tier_unpinned", FAMILY, "slot-a"),
        )
        assert active_pins(tier, through_seq=len(tier)) == {
            "slot-a": frozenset({SECOND, WITNESS}),
            "slot-b": frozenset({WITNESS}),
        }
        assert active_pins(tier, through_seq=0) == {}
        assert active_pins([], through_seq=0) == {}


class TestWatermark:
    """Only entries at or below the decision's tier position count."""

    def test_a_pin_after_the_decision_fails(self) -> None:
        """A dependency pinned after the decision's position is not pinned at it."""
        tier = journal(*pinned())
        with pytest.raises(UnpinnedPromotionDependencyError):
            check(tier, through=2)

    def test_an_unpin_after_the_decision_does_not_count(self) -> None:
        """An unpin after the decision's position leaves the pin in force at it."""
        tier = journal(*pinned(), ("tier_unpinned", WITNESS, "slot-a"))
        assert check(tier, through=3) == pins()

    @pytest.mark.parametrize("journal_length", [0, 3])
    def test_a_journal_ending_before_the_decision_is_refused(
        self, journal_length: int
    ) -> None:
        """A journal that stops before the stated position is a visible truncation."""
        tier = journal(*pinned())[:journal_length]
        with pytest.raises(CustodyRecordError, match="ends before the decision"):
            check(tier, through=50)
        with pytest.raises(CustodyRecordError, match="ends before the decision"):
            active_pins(tier, through_seq=journal_length + 1)

    @pytest.mark.parametrize("through", [-1, True, "3"])
    def test_a_malformed_position_is_refused(self, through: object) -> None:
        """The position is an integer of at least zero."""
        with pytest.raises(CustodyRecordError, match="through_seq"):
            active_pins(journal(*pinned()), through_seq=through)  # type: ignore[arg-type]


class TestColdDigests:
    """A released digest is pinned in no scope until it is recalled."""

    def test_a_released_witness_fails(self) -> None:
        """M-B2-PIN-1: a witness released to the cold tier is not pinned."""
        with pytest.raises(UnpinnedPromotionDependencyError, match="not pinned"):
            check(journal(*pinned(), ("tier_released", WITNESS, "")))

    def test_a_recall_does_not_revive_the_earlier_pins(self) -> None:
        """Pinned, released, recalled, with no new pin: not pinned in any scope."""
        acts = (
            *pinned(),
            ("tier_pinned", WITNESS, "slot-b"),
            ("tier_released", WITNESS, ""),
            ("tier_recalled", WITNESS, ""),
        )
        tier = journal(*acts)
        with pytest.raises(UnpinnedPromotionDependencyError, match="not pinned"):
            check(tier)
        assert active_pins(tier, through_seq=len(tier)) == {
            "slot-a": frozenset({FAMILY, SECOND})
        }

    def test_a_new_pin_after_the_recall_counts(self) -> None:
        """Pinned again after the recall: pinned."""
        acts = (
            *pinned(),
            ("tier_released", WITNESS, ""),
            ("tier_recalled", WITNESS, ""),
            ("tier_pinned", WITNESS, "slot-a"),
        )
        assert check(journal(*acts)) == pins()

    def test_a_release_resets_only_its_own_digest(self) -> None:
        """Releasing an unrelated digest leaves the dependencies pinned."""
        assert (
            check(journal(*pinned(), ("tier_released", digest_of("other"), "")))
            == pins()
        )

    def test_a_staged_release_leaves_the_pin(self) -> None:
        """Staging a release removes nothing."""
        assert check(journal(*pinned(), ("tier_release_staged", WITNESS, ""))) == pins()


class TestOrderAndProgramme:
    """The journal is one programme's and gap-free from its origin."""

    @pytest.mark.parametrize(
        ("seqs", "fragment"),
        [
            ((1, 3), "exactly one"),
            ((1, 1), "exactly one"),
            ((1, 2, 1), "exactly one"),
            ((2, 3), "must start at tier_seq 1"),
            ((7,), "must start at tier_seq 1"),
        ],
        ids=["gap", "repeat", "back", "head-missing", "head-at-7"],
    )
    def test_order_refusals(self, seqs: tuple[int, ...], fragment: str) -> None:
        """A gap, repeat or step back, or a missing head, is refused."""
        tier = [
            entry(seq, "tier_pinned", digest_of(str(i))) for i, seq in enumerate(seqs)
        ]
        with pytest.raises(CustodyRecordError, match=fragment):
            active_pins(tier, through_seq=9)
        with pytest.raises(CustodyRecordError, match=fragment):
            require_dependencies_pinned(pins(), tier, through_seq=9, roles=ROLES)

    def test_an_entry_of_another_programme_is_refused(self) -> None:
        """Every entry is the first entry's programme."""
        tier = journal(*pinned())
        tier[2] = replace(tier[2], programme="pilot-b")
        with pytest.raises(CustodyRecordError, match="another programme"):
            check_tier_journal_order(tier)
        with pytest.raises(CustodyRecordError, match="another programme"):
            active_pins(tier, through_seq=3)

    def test_another_programmes_journal_is_refused(self) -> None:
        """The journal must be the configured programme's."""
        tier = [replace(item, programme="pilot-b") for item in journal(*pinned())]
        with pytest.raises(CustodyRecordError, match="another programme"):
            check(tier)
        assert check_tier_journal_order(tier) == tier


class TestRolesAndShape:
    """Pins count only from the deployment operator; entries are well formed."""

    def test_a_pin_by_another_role_fails(self) -> None:
        """A pin entered by the custodian does not count for a promotion."""
        tier = journal(*pinned())
        tier[2] = replace(tier[2], role="custodian")
        with pytest.raises(
            UnpinnedPromotionDependencyError, match="deployment operator"
        ):
            check(tier)

    def test_the_operator_role_must_be_configured(self) -> None:
        """An empty deployment-operator role is refused."""
        with pytest.raises(CustodyRecordError, match="deployment_operator_role"):
            require_dependencies_pinned(
                pins(),
                journal(*pinned()),
                through_seq=3,
                roles=replace(ROLES, deployment_operator_role=""),
            )

    def test_malformed_pins_are_refused(self) -> None:
        """The pins are validated first."""
        with pytest.raises(PromotionDecisionError):
            require_dependencies_pinned(
                replace(pins(), witness_and_parity_digests=()),
                journal(*pinned()),
                through_seq=3,
                roles=ROLES,
            )

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"pin_scope": ""}, CustodyRecordError),
            ({"action": "tier_moved"}, CustodyRecordError),
            ({"schema": "tier-entry.v2"}, CustodyRecordError),
            ({"tier_seq": 0}, CustodyRecordError),
            ({"role": ""}, CustodyRecordError),
            ({"holder": " h"}, CustodyRecordError),
            ({"reason": ""}, CustodyRecordError),
            ({"recorded_at_utc": "today"}, CustodyRecordError),
            ({"programme": ""}, CustodyRecordError),
            ({"digest": "x"}, MalformedDigestError),
        ],
    )
    def test_entry_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """A pin act names a scope; every field is checked."""
        with pytest.raises(error):
            validate_tier_journal_entry(
                replace(entry(1, "tier_pinned", FAMILY), **changes)  # type: ignore[arg-type]
            )

    def test_only_pin_acts_name_a_scope(self) -> None:
        """A release naming a scope is refused; one without passes."""
        released = entry(1, "tier_released", FAMILY)
        assert validate_tier_journal_entry(released) == released
        with pytest.raises(CustodyRecordError, match="only a pin or unpin"):
            validate_tier_journal_entry(replace(released, pin_scope="slot-a"))

    def test_records_of_other_types_are_refused_and_entries_round_trip(self) -> None:
        """The validator checks its type; the codec round-trips an entry."""
        with pytest.raises(CustodyRecordError):
            validate_tier_journal_entry(pins())  # type: ignore[arg-type]
        item = entry(1, "tier_unpinned", WITNESS)
        assert (
            from_canonical_json_bytes(
                CustodyTierJournalEntry, canonical_json_bytes(item)
            )
            == item
        )
