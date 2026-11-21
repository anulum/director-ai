# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Study Binding of a Promotion or Decline

"""Binding a promotion or a decline to its study, and checking a decline.

Both ends of a study bind to its preregistration, its build-2 terms, its
released confirmatory look, that look's custodied opening and its result.
A decline then names every declared arm but the terms' baseline (E-8).
"""

from __future__ import annotations

from .custody_records import CustodyIngestRecord
from .digests import record_digest
from .errors import EvaluationContractError, PromotionDecisionError
from .evaluation_contracts import PreregistrationRecord, validate_preregistration
from .evaluator_outputs import SealedLookResult, validate_look_result
from .look_admission import CustodyProgrammeRoles
from .promotion_evidence import PromotionEvidence
from .promotion_records import StudyDeclineRecord, validate_study_decline
from .sealed_looks import SealedLookJournalEntry
from .study_contracts import validate_build2_terms
from .study_records import PreregistrationBuild2Terms

E = PromotionDecisionError


def check_study_binding(
    terms: PreregistrationBuild2Terms,
    preregistration: PreregistrationRecord,
    journal_entry: SealedLookJournalEntry,
    look_result: SealedLookResult,
    roles: CustodyProgrammeRoles,
    opening_record: CustodyIngestRecord,
) -> str:
    """Bind the terms, the look, its opening and its result to the plan.

    Parameters
    ----------
    terms : PreregistrationBuild2Terms
        The study's build-2 terms.
    preregistration : PreregistrationRecord
        The study's preregistration.
    journal_entry : SealedLookJournalEntry
        The confirmatory look's released journal entry.
    look_result : SealedLookResult
        The look's result.
    roles : CustodyProgrammeRoles
        The programme's configured roles.
    opening_record : CustodyIngestRecord
        Custody's record of the look's opening.

    Returns
    -------
    str
        The preregistration's record digest.

    Raises
    ------
    PromotionDecisionError
        If the terms, the look, its opening or its result are not the
        preregistration's.
    """
    validate_preregistration(preregistration)
    validate_build2_terms(terms)
    if record_digest(terms) != preregistration.search_spaces_and_budgets_digest:
        raise E("the terms are not the preregistration's search_spaces_and_budgets")
    if {arm for arm, _ in terms.arm_matched_budgets} != set(
        preregistration.declared_arms
    ):
        raise E("the terms' arms are not the preregistration's declared arms")
    try:
        validate_look_result(look_result, journal_entry, opening_record=opening_record)
    except EvaluationContractError as exc:
        raise E(f"the look result is not bound: {exc}") from exc
    digest = record_digest(preregistration)
    if journal_entry.study_id != preregistration.study_id:
        raise E("the look belongs to another study")
    if journal_entry.preregistration_digest != digest:
        raise E("the look was admitted under another preregistration")
    if journal_entry.programme != roles.programme:
        raise E("the look belongs to another programme")
    return digest


def check_decline(
    decline: StudyDeclineRecord,
    evidence: PromotionEvidence,
    candidates: set[str],
    preregistration_digest: str,
    look_result: SealedLookResult,
    study_id: str,
) -> StudyDeclineRecord:
    """Check that a decline is the study's and names every candidate arm.

    Parameters
    ----------
    decline : StudyDeclineRecord
        The study decline.
    evidence : PromotionEvidence
        The arms' artefacts; a decline promotes none.
    candidates : set[str]
        Every declared arm but the baseline.
    preregistration_digest : str
        The preregistration's record digest.
    look_result : SealedLookResult
        The look's result.
    study_id : str
        The study.

    Returns
    -------
    StudyDeclineRecord
        The same decline.

    Raises
    ------
    PromotionDecisionError
        If the decline promotes an arm, is another study's, names another
        look result, or does not name exactly the candidate arms (E-8).
    """
    if evidence.promoted_arm is not None:
        raise E("a decline promotes no arm")
    validate_study_decline(decline)
    if (decline.study_id, decline.preregistration_digest) != (
        study_id,
        preregistration_digest,
    ):
        raise E("the decline names another study or preregistration")
    if decline.sealed_look_result_digest != record_digest(look_result):
        raise E("the decline names another look result")
    failed = {arm for arm, _ in decline.failed_arm_reasons}
    if failed != candidates:
        raise E(
            "E-8: a decline names every declared arm but the baseline, and no other"
        )
    return decline
