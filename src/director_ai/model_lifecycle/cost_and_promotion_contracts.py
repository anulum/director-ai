# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Promotion Decisions

"""Promotion decision records and their production pre-state.

A promotion decision names the serving slot's expected active generation
and production digest: the generation chain's required pre-state. It binds
the candidate to its preregistration, its released look, the evaluation
review's admission and the custody blobs of the candidate and the rollback
target. Publication itself belongs to the generation chain, not to this
record.

Generation 0 has no production artefact, so its expected digest is 64
zeros. A deployment applies the decision only while the slot still shows
exactly that pre-state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_choice,
    require_int,
    require_sha256,
    require_text,
    require_utc_timestamp,
)
from .custody_records import CustodyIngestRecord
from .digests import record_digest
from .errors import EvaluationContractError, PromotionDecisionError
from .evaluator_outputs import SealedLookResult, validate_look_result
from .sealed_looks import SealedLookJournalEntry

EMPTY_PRODUCTION_DIGEST = "0" * 64
"""Expected production digest of a slot that has no generation yet."""


@dataclass(frozen=True, slots=True)
class PromotionDecisionRecord:
    """A decision to promote one candidate into one serving slot.

    Attributes
    ----------
    schema : Literal["model-promotion-decision.v1"]
        Record schema.
    intent_sha256 : str
        SHA-256 of the promotion intent.
    model_slot : str
        The serving slot.
    expected_active_generation : int
        The slot's active generation the decision expects.
    expected_production_digest : str
        The production artefact digest it expects, 64 zeros at generation 0.
    candidate_content_digest : str
        Content digest of the candidate.
    preregistration_digest : str
        Record digest of the study's preregistration.
    sealed_look_result_digest : str
        Record digest of the released look's result.
    rigor_admission_digest : str
        SHA-256 of the independent evaluation review's admission.
    remanentia_control_packet_digest : str
        SHA-256 of the custody control packet.
    remanentia_candidate_blob_digest : str
        SHA-256 of the candidate's custody blob.
    remanentia_rollback_blob_digest : str
        SHA-256 of the rollback target's custody blob.
    decided_at_utc : str
        Informational decision time.
    """

    schema: Literal["model-promotion-decision.v1"]
    intent_sha256: str
    model_slot: str
    expected_active_generation: int
    expected_production_digest: str
    candidate_content_digest: str
    preregistration_digest: str
    sealed_look_result_digest: str
    rigor_admission_digest: str
    remanentia_control_packet_digest: str
    remanentia_candidate_blob_digest: str
    remanentia_rollback_blob_digest: str
    decided_at_utc: str


def validate_promotion_decision(
    decision: PromotionDecisionRecord,
    *,
    look_result: SealedLookResult | None = None,
    journal_entry: SealedLookJournalEntry | None = None,
    opening_record: CustodyIngestRecord | None = None,
) -> PromotionDecisionRecord:
    """Check a promotion decision, and its look when given.

    Parameters
    ----------
    decision : PromotionDecisionRecord
        The decision to check.
    look_result : SealedLookResult, optional
        The released look's result; when given, its record digest must
        equal ``sealed_look_result_digest``.
    journal_entry : SealedLookJournalEntry, optional
        The look's journal entry, required with ``look_result``. The
        result must belong to it, and the candidate must be among its
        frozen artefacts, for slice estimates and test-suite counts alike.
    opening_record : CustodyIngestRecord, optional
        Custody's record of the look's opening, required with
        ``look_result``; it must be ingested no later than the release.

    Returns
    -------
    PromotionDecisionRecord
        The same decision.

    Raises
    ------
    PromotionDecisionError
        If a field is malformed, generation 0 does not expect 64 zeros, the
        candidate equals the expected production artefact, or the look
        result does not belong to the decision.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(decision, PromotionDecisionRecord):
        raise PromotionDecisionError("a PromotionDecisionRecord is required")
    require_choice(
        decision.schema,
        "schema",
        {"model-promotion-decision.v1"},
        PromotionDecisionError,
    )
    require_sha256(decision.intent_sha256, "intent_sha256")
    require_text(decision.model_slot, "model_slot", PromotionDecisionError)
    generation = require_int(
        decision.expected_active_generation,
        "expected_active_generation",
        PromotionDecisionError,
        minimum=0,
    )
    production = require_sha256(
        decision.expected_production_digest, "expected_production_digest"
    )
    if (generation == 0) != (production == EMPTY_PRODUCTION_DIGEST):
        raise PromotionDecisionError(
            "generation 0, and only generation 0, expects the empty production digest"
        )
    candidate = require_sha256(
        decision.candidate_content_digest, "candidate_content_digest"
    )
    if candidate == production:
        raise PromotionDecisionError(
            "the candidate is already the expected production artefact"
        )
    for name in (
        "preregistration_digest",
        "sealed_look_result_digest",
        "rigor_admission_digest",
        "remanentia_control_packet_digest",
        "remanentia_candidate_blob_digest",
        "remanentia_rollback_blob_digest",
    ):
        require_sha256(getattr(decision, name), name)
    require_utc_timestamp(
        decision.decided_at_utc, "decided_at_utc", PromotionDecisionError
    )
    given = {item is None for item in (look_result, journal_entry, opening_record)}
    if len(given) != 1:
        raise PromotionDecisionError(
            "a look result, its journal entry and its opening record go together"
        )
    if (
        look_result is not None
        and journal_entry is not None
        and opening_record is not None
    ):
        if record_digest(look_result) != decision.sealed_look_result_digest:
            raise PromotionDecisionError(
                "the look result does not match sealed_look_result_digest"
            )
        try:
            validate_look_result(
                look_result, journal_entry, opening_record=opening_record
            )
        except EvaluationContractError as exc:
            raise PromotionDecisionError(
                f"the look result is not bound: {exc}"
            ) from exc
        if candidate not in journal_entry.frozen_artefact_digests:
            raise PromotionDecisionError("the look did not freeze the candidate")
        reported = {
            item.artefact_digest
            for item in look_result.evaluator_output.slice_estimates
        }
        if (
            look_result.evaluator_output.output_mode == "slice_estimates"
            and candidate not in reported
        ):
            raise PromotionDecisionError(
                "the look result does not report the candidate"
            )
    return decision


def check_promotion_pre_state(
    decision: PromotionDecisionRecord, *, active_generation: int, production_digest: str
) -> None:
    """Refuse a decision whose expected pre-state the slot no longer shows.

    Parameters
    ----------
    decision : PromotionDecisionRecord
        A validated decision.
    active_generation : int
        The slot's current active generation.
    production_digest : str
        The slot's current production digest, 64 zeros at generation 0.

    Raises
    ------
    PromotionDecisionError
        If the generation is not an integer, or the generation or the
        production digest differs from the decision's expectation.
    MalformedDigestError
        If the production digest is not 64 hex characters.
    """
    validate_promotion_decision(decision)
    if type(active_generation) is not int:
        raise PromotionDecisionError("the slot's active generation must be an integer")
    require_sha256(production_digest, "production_digest")
    if active_generation != decision.expected_active_generation:
        raise PromotionDecisionError(
            "the slot's active generation differs from the decision's"
        )
    if production_digest != decision.expected_production_digest:
        raise PromotionDecisionError(
            "the slot's production digest differs from the decision's"
        )
