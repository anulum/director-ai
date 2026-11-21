# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Family Witnesses at Promotion

"""The witnesses a promoted candidate's family needs, and how they bind.

The family comes from the study's terms, which the preregistration froze,
never from a record the caller supplies. The candidate's envelope is itself
a witness: it is the candidate, of the terms' family, under the pins'
family contract, and pinned. Pinning (Q-9) only proves that a witness is
held, so each witness here must also have passed and must bind to the
study:

- a spiking candidate's lowering parity report is of the candidate, pinned
  and passed (TSN-3), and its hardware numeric contract is a pinned
  companion contract;
- a variational quantum candidate's acceptance spec is the terms', its
  paired receipts match it (Q-6) and are the candidate's and the terms'
  baseline's, its barren-plateau witness passed for the candidate's
  contract on the study's sealed pool, and the spec, both receipts, the
  witness and the classical baseline (Q-5) are all pinned.

A language or dynamics candidate carries none of these, so a stray witness
is refused. The seeds (Q-7) are checked where runs start, and the quantum
lower-bound conjunct is applied by the evaluation review's admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import require_choice
from .digests import record_digest
from .envelopes import ModelArtefactEnvelope, validate_envelope
from .errors import PromotionDecisionError
from .family_witnesses import (
    require_spiking_parity_passed,
    validate_barren_plateau_receipt,
    validate_spiking_lowering_parity,
)
from .promotion_records import PromotionDependencyPins
from .quantum_contracts import (
    validate_paired_receipts,
    validate_receipt_shape,
    validate_shot_noise_spec,
)
from .quantum_records import (
    BarrenPlateauWitnessReceipt,
    QuantumCircuitExecutionReceipt,
    QuantumShotNoiseAcceptanceSpec,
    SpikingLoweringParityReport,
)

E = PromotionDecisionError
QUANTUM_WITNESSES = (
    "barren_plateau_receipt",
    "acceptance_spec",
    "candidate_receipt",
    "baseline_receipt",
)
"""The witnesses a variational quantum candidate needs."""
FAMILY_WITNESSES: dict[str, tuple[str, ...]] = {
    "spiking": ("spiking_parity_report",),
    "quantum_variational": QUANTUM_WITNESSES,
}
"""Each family's witnesses; a family not listed needs none."""


@dataclass(frozen=True, slots=True)
class FamilyWitnesses:
    """The candidate's envelope and the witnesses its family needs at promotion.

    Attributes
    ----------
    schema : Literal["promotion-family-witnesses.v1"]
        Record schema.
    candidate_envelope : ModelArtefactEnvelope
        The promoted candidate's envelope.
    spiking_parity_report : SpikingLoweringParityReport or None
        A spiking candidate's lowering parity report.
    barren_plateau_receipt : BarrenPlateauWitnessReceipt or None
        A variational quantum candidate's preflight witness.
    acceptance_spec : QuantumShotNoiseAcceptanceSpec or None
        A variational quantum study's acceptance spec.
    candidate_receipt : QuantumCircuitExecutionReceipt or None
        The candidate arm's execution receipt.
    baseline_receipt : QuantumCircuitExecutionReceipt or None
        The baseline arm's execution receipt.
    """

    schema: Literal["promotion-family-witnesses.v1"]
    candidate_envelope: ModelArtefactEnvelope
    spiking_parity_report: SpikingLoweringParityReport | None
    barren_plateau_receipt: BarrenPlateauWitnessReceipt | None
    acceptance_spec: QuantumShotNoiseAcceptanceSpec | None
    candidate_receipt: QuantumCircuitExecutionReceipt | None
    baseline_receipt: QuantumCircuitExecutionReceipt | None


def validate_family_witnesses(witnesses: FamilyWitnesses) -> FamilyWitnesses:
    """Check the shape of a candidate's envelope and of each witness given.

    Parameters
    ----------
    witnesses : FamilyWitnesses
        The witnesses to check.

    Returns
    -------
    FamilyWitnesses
        The same witnesses.

    Raises
    ------
    PromotionDecisionError
        If the record or its schema is malformed.
    EnvelopeError, SpikingParityError, QuantumAcceptanceError
        If the envelope or a witness is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(witnesses, FamilyWitnesses):
        raise E("a FamilyWitnesses is required")
    require_choice(witnesses.schema, "schema", {"promotion-family-witnesses.v1"}, E)
    validate_envelope(witnesses.candidate_envelope)
    if witnesses.spiking_parity_report is not None:
        validate_spiking_lowering_parity(witnesses.spiking_parity_report)
    if witnesses.barren_plateau_receipt is not None:
        validate_barren_plateau_receipt(witnesses.barren_plateau_receipt)
    if witnesses.acceptance_spec is not None:
        validate_shot_noise_spec(witnesses.acceptance_spec)
    for receipt in (witnesses.candidate_receipt, witnesses.baseline_receipt):
        if receipt is not None:
            validate_receipt_shape(receipt)
    return witnesses


def _pinned(record: object, pinned: set[str], what: str) -> None:
    """Refuse a witness whose record digest the operator has not pinned."""
    if record_digest(record) not in pinned:
        raise E(f"the {what} is not a pinned witness")


def _check_spiking(
    witnesses: FamilyWitnesses,
    candidate: str,
    pins: PromotionDependencyPins,
    pinned: set[str],
) -> None:
    """Bind a spiking candidate's passed parity report (TSN-3)."""
    report = witnesses.spiking_parity_report
    if report is None:
        raise E("a spiking candidate needs its lowering parity report")
    require_spiking_parity_passed(report)
    if report.checkpoint_content_digest != candidate:
        raise E("the parity report is of another checkpoint than the candidate")
    _pinned(report, pinned, "spiking lowering parity report")
    if (
        report.hardware_numeric_contract_digest
        not in pins.secondary_family_contract_digests
    ):
        raise E("the hardware numeric contract is not a pinned companion contract")


def _check_quantum(
    witnesses: FamilyWitnesses,
    *,
    candidate: str,
    baseline: str,
    pinned: set[str],
    spec_digest: str | None,
    split_pool_digest: str,
) -> None:
    """Bind a variational quantum candidate's spec, receipts and preflight."""
    spec, preflight = witnesses.acceptance_spec, witnesses.barren_plateau_receipt
    candidate_receipt, baseline_receipt = (
        witnesses.candidate_receipt,
        witnesses.baseline_receipt,
    )
    if (
        spec is None
        or preflight is None
        or candidate_receipt is None
        or baseline_receipt is None
    ):
        raise E(
            "a variational quantum candidate needs its spec, receipts and preflight"
        )
    if record_digest(spec) != spec_digest:
        raise E("the acceptance spec is not the terms'")
    validate_paired_receipts(candidate_receipt, baseline_receipt, spec)
    if candidate_receipt.arm_content_digest != candidate:
        raise E("the candidate receipt is of another arm than the candidate")
    if baseline_receipt.arm_content_digest != baseline:
        raise E("the baseline receipt is of another arm than the terms' baseline")
    if preflight.passed is not True:
        raise E("the barren-plateau preflight did not pass")
    if (
        preflight.companion_digest
        != witnesses.candidate_envelope.family_contract_digest
    ):
        raise E("the preflight is of another family contract than the candidate's")
    if preflight.split_pool_digest != split_pool_digest:
        raise E("the preflight sampled another split pool than the study's")
    for record, what in (
        (spec, "acceptance spec"),
        (candidate_receipt, "candidate receipt"),
        (baseline_receipt, "baseline receipt"),
        (preflight, "barren-plateau witness"),
    ):
        _pinned(record, pinned, what)
    if spec.classical_baseline_digest not in pinned:
        raise E("the classical baseline is not a pinned witness")


def check_family_witnesses(
    witnesses: FamilyWitnesses,
    *,
    family: str,
    candidate: str,
    baseline: str,
    pins: PromotionDependencyPins,
    quantum_spec_digest: str | None,
    split_pool_digest: str,
) -> FamilyWitnesses:
    """Refuse a promotion whose candidate's family witnesses do not hold.

    Parameters
    ----------
    witnesses : FamilyWitnesses
        The candidate's envelope and witnesses.
    family : str
        The terms' family.
    candidate : str
        The decision's candidate content digest.
    baseline : str
        The terms' baseline arm's artefact digest.
    pins : PromotionDependencyPins
        The promotion's validated pins.
    quantum_spec_digest : str or None
        The terms' quantum acceptance spec digest.
    split_pool_digest : str
        The record digest of the preregistration's sealed split pool.

    Returns
    -------
    FamilyWitnesses
        The same witnesses.

    Raises
    ------
    PromotionDecisionError
        If the envelope is not the pinned candidate of the terms' family,
        a witness the family needs is missing, unbound, unpinned or did
        not pass, or a witness the family does not use is given.
    EnvelopeError, SpikingParityError, QuantumAcceptanceError
        If the envelope or a witness is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    validate_family_witnesses(witnesses)
    envelope, pinned = (
        witnesses.candidate_envelope,
        set(pins.witness_and_parity_digests),
    )
    if envelope.content_digest != candidate:
        raise E("the envelope is not the decision's candidate")
    if envelope.family != family:
        raise E("the candidate is of another family than the study's")
    if envelope.family_contract_digest != pins.family_contract_digest:
        raise E("the candidate names another family contract than the pins'")
    _pinned(envelope, pinned, "candidate envelope")
    used = FAMILY_WITNESSES.get(family, ())
    for name in ("spiking_parity_report", *QUANTUM_WITNESSES):
        if name not in used and getattr(witnesses, name) is not None:
            raise E(f"a {family} candidate carries no {name}")
    if family == "spiking":
        _check_spiking(witnesses, candidate, pins, pinned)
    elif family == "quantum_variational":
        _check_quantum(
            witnesses,
            candidate=candidate,
            baseline=baseline,
            pinned=pinned,
            spec_digest=quantum_spec_digest,
            split_pool_digest=split_pool_digest,
        )
    return witnesses
