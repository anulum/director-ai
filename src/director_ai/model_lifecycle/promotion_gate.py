# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Promotion or Decline

"""The gate that ends a study: one promotion decision, or one complete decline.

Exactly one of a promotion decision and a study decline is given (E-8).
Both bind to the study's preregistration, its build-2 terms (by the
preregistration's ``search_spaces_and_budgets_digest``), its released
confirmatory look and the look's result.

The baseline arm is the terms', fixed before any look. A decline names
every declared arm except that baseline, each with reasons: no candidate
arm cleared every gate. A decline is not a finding that an arm is worse;
failing to show non-inferiority is not inferiority. A promotion passes
every kernel gate for its promoted arm, which is not the baseline, or it is
refused:

- E-4, E-5 and M-J4-12, the study gates;
- the look measured the baseline's artefact on every ``(slice, metric)``
  pair it measured the candidate on, so the comparator is concurrent;
- the serving memory budget;
- Q-9, its dependencies pinned in its scope by the deployment operator;
- its family's witnesses, from the terms' family: the candidate's pinned
  envelope, and for a spiking or variational quantum candidate its passed,
  bound and pinned parity report, or spec, receipts and preflight;
- its export parity reports: of the candidate, pinned as witnesses, each
  covering every ``(slice, metric)`` pair the look reported for the
  candidate, with every non-rate slice's allowed margin at least three
  noise floors, and, for a dynamics study, a rollout horizon at least the
  dynamics contract's.

Two inputs are caller-supplied trusts here. The memory budget is checked
for its own sum, not tied to the candidate, and the pins are scoped to the
deployment scope the operator pinned in, not to the decision's slot; the
deployment port's ``declare_deployment`` binds both to the decision.

Whether the promoted arm passed its statistical gate is not computed here:
the decision carries the independent evaluation review's admission, and the
paired-difference interval it rests on is in that admission, since the
sealed result holds per-arm estimates only.
"""

from __future__ import annotations

from collections.abc import Sequence

from .cost_and_promotion_contracts import (
    PromotionDecisionRecord,
    validate_promotion_decision,
)
from .custody_records import CustodyIngestRecord
from .digests import record_digest
from .errors import PromotionDecisionError
from .evaluation_contracts import PreregistrationRecord
from .evaluator_outputs import SealedLookResult
from .look_admission import CustodyProgrammeRoles
from .promotion_evidence import (
    PromotionEvidence,
    check_held_equal_axes,
    check_inference_envelope,
    check_noise_floors,
    evidence_arm_digests,
)
from .promotion_exports import check_promotion_exports
from .promotion_records import (
    PromotionDependencyPins,
    ServingMemoryBudget,
    StudyDeclineRecord,
    validate_serving_memory_budget,
)
from .promotion_study import check_decline, check_study_binding
from .promotion_witnesses import FamilyWitnesses, check_family_witnesses
from .sealed_looks import SealedLookJournalEntry
from .study_records import (
    DynamicsEvaluationContract,
    ExportParityReport,
    PreregistrationBuild2Terms,
)
from .tier_journal import CustodyTierJournalEntry, require_dependencies_pinned

E = PromotionDecisionError


def _check_concurrent_baseline(
    look_result: SealedLookResult, candidate: str, baseline: str
) -> None:
    """Require the look to measure the baseline wherever it measured the candidate."""
    estimates = look_result.evaluator_output.slice_estimates
    measured = {
        artefact: {
            (item.slice_id, item.metric_id)
            for item in estimates
            if item.artefact_digest == artefact
        }
        for artefact in (candidate, baseline)
    }
    if not measured[candidate] <= measured[baseline]:
        raise E(
            "the look did not measure the baseline wherever it measured the candidate"
        )


def validate_promotion_or_decline(
    *,
    decision: PromotionDecisionRecord | None,
    decline: StudyDeclineRecord | None,
    evidence: PromotionEvidence,
    terms: PreregistrationBuild2Terms,
    preregistration: PreregistrationRecord,
    look_result: SealedLookResult,
    journal_entry: SealedLookJournalEntry,
    opening_record: CustodyIngestRecord,
    memory_budget: ServingMemoryBudget | None,
    pins: PromotionDependencyPins | None,
    export_reports: Sequence[ExportParityReport],
    dynamics_contract: DynamicsEvaluationContract | None,
    family_witnesses: FamilyWitnesses | None,
    tier_journal: Sequence[CustodyTierJournalEntry],
    tier_seq_at_decision: int,
    roles: CustodyProgrammeRoles,
) -> PromotionDecisionRecord | StudyDeclineRecord:
    """Admit a study's one promotion decision or its one complete decline.

    Parameters
    ----------
    decision : PromotionDecisionRecord or None
        The promotion decision, or ``None`` for a decline.
    decline : StudyDeclineRecord or None
        The study decline, or ``None`` for a promotion.
    evidence : PromotionEvidence
        The arms' artefacts, and the promoted arm for a promotion.
    terms : PreregistrationBuild2Terms
        The study's build-2 terms.
    preregistration : PreregistrationRecord
        The study's preregistration.
    look_result : SealedLookResult
        The confirmatory look's result.
    journal_entry : SealedLookJournalEntry
        The confirmatory look's released journal entry.
    opening_record : CustodyIngestRecord
        Custody's record of the look's opening, ingested by its release.
    memory_budget : ServingMemoryBudget or None
        The promoted model's serving memory budget; not read for a decline.
    pins : PromotionDependencyPins or None
        The promotion's dependency pins; not read for a decline.
    export_reports : Sequence[ExportParityReport]
        The candidate's serving-binary parity reports; not read for a decline.
    dynamics_contract : DynamicsEvaluationContract or None
        The terms' dynamics contract, when they name one; not read for a
        decline.
    family_witnesses : FamilyWitnesses or None
        The candidate's envelope and its family's witnesses; required for a
        promotion, not read for a decline.
    tier_journal : Sequence[CustodyTierJournalEntry]
        The programme's custody tier journal.
    tier_seq_at_decision : int
        The tier journal position of the decision.
    roles : CustodyProgrammeRoles
        The programme's configured roles.

    Returns
    -------
    PromotionDecisionRecord or StudyDeclineRecord
        The admitted decision or decline.

    Raises
    ------
    PromotionDecisionError
        If both or neither of a decision and a decline is given, a record
        binds to another study, preregistration, look or candidate, a
        decline is incomplete, or a promotion fails a gate.
    UnpinnedPromotionDependencyError
        If a promotion's dependency is not pinned in its scope (Q-9).
    EvaluationContractError, RecipeContractError, ExportParityError
        If a preregistration, terms, contract or report is malformed.
    CustodyRecordError
        If the tier journal or the roles are malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    digest = check_study_binding(
        terms, preregistration, journal_entry, look_result, roles, opening_record
    )
    arms = evidence_arm_digests(evidence, preregistration)
    if decline is not None and decision is None:
        candidates = set(arms) - {terms.baseline_arm}
        return check_decline(
            decline, evidence, candidates, digest, look_result, preregistration.study_id
        )
    if decision is None or decline is not None:
        raise E("E-8: give exactly one of a promotion decision and a study decline")
    if evidence.promoted_arm is None:
        raise E("a promotion names its promoted arm")
    if evidence.promoted_arm == terms.baseline_arm:
        raise E("the promoted arm is the study's baseline")
    validate_promotion_decision(
        decision,
        look_result=look_result,
        journal_entry=journal_entry,
        opening_record=opening_record,
    )
    if decision.preregistration_digest != digest:
        raise E("the decision names another preregistration")
    candidate = decision.candidate_content_digest
    if arms[evidence.promoted_arm] != candidate:
        raise E("the promoted arm's artefact is not the decision's candidate")
    check_noise_floors(terms, preregistration)
    check_held_equal_axes(terms)
    _check_concurrent_baseline(look_result, candidate, arms[terms.baseline_arm])
    check_inference_envelope(terms, promoted_arm=evidence.promoted_arm)
    if memory_budget is None or pins is None:
        raise E("a promotion needs its serving memory budget and dependency pins")
    validate_serving_memory_budget(memory_budget)
    require_dependencies_pinned(
        pins, tier_journal, through_seq=tier_seq_at_decision, roles=roles
    )
    if family_witnesses is None:
        raise E("a promotion needs its candidate's family witnesses")
    check_family_witnesses(
        family_witnesses,
        family=terms.family,
        candidate=candidate,
        baseline=arms[terms.baseline_arm],
        pins=pins,
        quantum_spec_digest=terms.quantum_acceptance_spec_digest,
        split_pool_digest=record_digest(preregistration.split_pool),
    )
    check_promotion_exports(
        tuple(export_reports),
        candidate=candidate,
        pins=pins,
        look_result=look_result,
        terms=terms,
        dynamics_contract=dynamics_contract,
    )
    return decision
