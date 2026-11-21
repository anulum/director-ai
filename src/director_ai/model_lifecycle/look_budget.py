# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Look Budget

"""The sealed look budget of a lineage on one split.

``released`` and ``candidate_failed`` consume the one confirmatory look.
The two attested faults share one ``fault_relook``. The budget is checked by
intersection: any earlier entry on the split that shares a member with the
look's lineage class counts, so a merge can never reopen a budget.
"""

from __future__ import annotations

from collections.abc import Sequence

from .errors import LookBudgetExhaustedError
from .lineage import LineageComponent
from .sealed_looks import (
    CONSUMING_OUTCOMES,
    FAULT_OUTCOMES,
    LookKind,
    SealedLookJournalEntry,
)


def check_look_budget(
    component: LineageComponent,
    *,
    split_digest: str,
    look_kind: LookKind,
    prior_journal: Sequence[SealedLookJournalEntry],
) -> None:
    """Refuse a look the lineage has no budget for on the split.

    Parameters
    ----------
    component : LineageComponent
        The look's lineage class.
    split_digest : str
        The requested split.
    look_kind : LookKind
        ``initial_confirmatory`` or ``fault_relook``.
    prior_journal : Sequence[SealedLookJournalEntry]
        Every earlier journal entry.

    Raises
    ------
    LookBudgetExhaustedError
        If an earlier look on the split shares a member with the class and
        consumed the look; if an initial look follows any earlier look; or
        if a relook has no earlier fault or follows a relook.
    """
    shared = [
        entry
        for entry in prior_journal
        if entry.split_digest == split_digest
        and component.members & set(entry.frozen_artefact_digests)
    ]
    if any(entry.status in CONSUMING_OUTCOMES for entry in shared):
        raise LookBudgetExhaustedError(
            "the lineage already consumed its confirmatory look on this split"
        )
    if look_kind == "initial_confirmatory":
        if shared:
            raise LookBudgetExhaustedError(
                "an earlier look on this split faulted; only a fault_relook may follow"
            )
        return
    if not any(entry.status in FAULT_OUTCOMES for entry in shared):
        raise LookBudgetExhaustedError(
            "a fault_relook needs an earlier attested fault on this split"
        )
    if any(entry.look_kind == "fault_relook" for entry in shared):
        raise LookBudgetExhaustedError(
            "the lineage already used its one fault_relook on this split"
        )
