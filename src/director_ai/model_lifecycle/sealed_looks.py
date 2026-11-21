# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Sealed Looks

"""Sealed look requests, evaluator outputs, results and journal entries.

A look has four outcomes, never merged:

* ``released`` and ``candidate_failed`` follow exposure of the sealed split
  to candidate code or weights. Both consume the single confirmatory look of
  the lineage on that split, and both taint the lineage after the look.
* ``infrastructure_fault`` and ``voided_evaluator_fault`` are attested as
  not attributable to the candidate. They give no result, carry a
  ``fault_attestation_digest`` and share one ``fault_relook``.

When attribution is uncertain the outcome is ``candidate_failed``. A refusal
before the split is opened consumes nothing and writes no entry.

An :class:`EvaluatorOutput` is what the evaluator emits: per-slice aggregate
estimates and intervals, or a test suite's pass count and run count, never
item-level data. A :class:`SealedLookResult` wraps it after journalling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_choice,
    require_digests,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_utc_timestamp,
)
from .envelopes import LineageEdge, validate_lineage_edge
from .errors import EvaluationContractError

LookKind = Literal["initial_confirmatory", "fault_relook"]
LOOK_KINDS: frozenset[str] = frozenset({"initial_confirmatory", "fault_relook"})
SealedLookOutcomeStatus = Literal[
    "released", "candidate_failed", "infrastructure_fault", "voided_evaluator_fault"
]
CONSUMING_OUTCOMES: frozenset[str] = frozenset({"released", "candidate_failed"})
"""Outcomes that consume the confirmatory look and taint the lineage."""
FAULT_OUTCOMES: frozenset[str] = frozenset(
    {"infrastructure_fault", "voided_evaluator_fault"}
)
"""Attested non-candidate faults; they share one fault relook."""


@dataclass(frozen=True, slots=True)
class SealedLookRequest:
    """A request to evaluate frozen artefacts on one sealed split.

    The requester's role is durable and its holder is the current seat.
    Declared edges may only add merges to the custody-derived lineage.
    """

    schema: Literal["sealed-look-request.v1"]
    preregistration_digest: str
    split_digest: str
    frozen_artefact_digests: tuple[str, ...]
    declared_lineage_edges: tuple[LineageEdge, ...]
    requester_role: str
    requester_holder: str
    look_kind: LookKind
    programme: str
    evaluator_name: str
    evaluator_digest: str


@dataclass(frozen=True, slots=True)
class SealedLookJournalEntry:
    """Custody's journal entry for one look that opened a split.

    ``custody_seq_at_release`` is the custody sequence watermark when the
    look finished; it anchors post-look taint.
    """

    schema: Literal["sealed-look-journal-entry.v1"]
    journal_seq: int
    custody_seq_at_release: int
    derived_proposer_lineage_id: str
    split_digest: str
    frozen_artefact_digests: tuple[str, ...]
    requester_role: str
    requester_holder: str
    look_kind: LookKind
    status: SealedLookOutcomeStatus
    fault_attestation_digest: str
    recorded_at_utc: str
    programme: str
    evaluator_name: str
    evaluator_digest: str


def validate_look_request(request: SealedLookRequest) -> SealedLookRequest:
    """Check a look request's own fields.

    Parameters
    ----------
    request : SealedLookRequest
        The request to check.

    Returns
    -------
    SealedLookRequest
        The same request.

    Raises
    ------
    EvaluationContractError
        If a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(request, SealedLookRequest):
        raise EvaluationContractError("a SealedLookRequest is required")
    require_choice(
        request.schema, "schema", {"sealed-look-request.v1"}, EvaluationContractError
    )
    require_sha256(request.preregistration_digest, "preregistration_digest")
    require_sha256(request.split_digest, "split_digest")
    require_digests(
        request.frozen_artefact_digests,
        "frozen_artefact_digests",
        EvaluationContractError,
        non_empty=True,
    )
    for edge in require_tuple(
        request.declared_lineage_edges,
        "declared_lineage_edges",
        EvaluationContractError,
    ):
        if not isinstance(edge, LineageEdge):
            raise EvaluationContractError(
                "declared_lineage_edges must hold LineageEdge records"
            )
        validate_lineage_edge(edge)
    require_text(request.requester_role, "requester_role", EvaluationContractError)
    require_text(request.requester_holder, "requester_holder", EvaluationContractError)
    require_choice(request.look_kind, "look_kind", LOOK_KINDS, EvaluationContractError)
    require_text(request.programme, "programme", EvaluationContractError)
    require_text(request.evaluator_name, "evaluator_name", EvaluationContractError)
    require_sha256(request.evaluator_digest, "evaluator_digest")
    return request


def validate_journal_entry(entry: SealedLookJournalEntry) -> SealedLookJournalEntry:
    """Check a journal entry, including its fault attestation rule.

    Parameters
    ----------
    entry : SealedLookJournalEntry
        The entry to check.

    Returns
    -------
    SealedLookJournalEntry
        The same entry.

    Raises
    ------
    EvaluationContractError
        If a field is malformed, or the attestation digest is not present
        exactly for the two fault outcomes.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(entry, SealedLookJournalEntry):
        raise EvaluationContractError("a SealedLookJournalEntry is required")
    require_choice(
        entry.schema,
        "schema",
        {"sealed-look-journal-entry.v1"},
        EvaluationContractError,
    )
    require_int(entry.journal_seq, "journal_seq", EvaluationContractError, minimum=0)
    require_int(
        entry.custody_seq_at_release,
        "custody_seq_at_release",
        EvaluationContractError,
        minimum=0,
    )
    require_sha256(entry.derived_proposer_lineage_id, "derived_proposer_lineage_id")
    require_sha256(entry.split_digest, "split_digest")
    require_digests(
        entry.frozen_artefact_digests,
        "frozen_artefact_digests",
        EvaluationContractError,
        non_empty=True,
    )
    require_text(entry.requester_role, "requester_role", EvaluationContractError)
    require_text(entry.requester_holder, "requester_holder", EvaluationContractError)
    require_choice(entry.look_kind, "look_kind", LOOK_KINDS, EvaluationContractError)
    status = require_choice(
        entry.status,
        "status",
        CONSUMING_OUTCOMES | FAULT_OUTCOMES,
        EvaluationContractError,
    )
    if status in FAULT_OUTCOMES:
        require_sha256(entry.fault_attestation_digest, "fault_attestation_digest")
    elif entry.fault_attestation_digest != "":
        raise EvaluationContractError(
            "only fault outcomes carry a fault_attestation_digest"
        )
    require_utc_timestamp(
        entry.recorded_at_utc, "recorded_at_utc", EvaluationContractError
    )
    require_text(entry.programme, "programme", EvaluationContractError)
    require_text(entry.evaluator_name, "evaluator_name", EvaluationContractError)
    require_sha256(entry.evaluator_digest, "evaluator_digest")
    return entry
