# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Family Verifier Ports

"""Ports that evaluate quantum and physical-dynamics arms for a sealed look.

The quantum verifier runs the barren-plateau preflight and evaluates a
paired candidate and baseline under the study's acceptance spec; the
bootstrap bounds the kernel's admission gate consumes are its output. The
dynamics verifier evaluates trajectories, conformal coverage and shift
deferral under the dynamics contract. Adapters outside the kernel implement
them, and the kernel's guarantees on their results are conditional on them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .encoding_records import DynamicsNormaliserEnvelope
from .envelopes import ModelArtefactEnvelope
from .evaluation_contracts import PreregistrationRecord
from .evaluator_outputs import SealedLookResult
from .quantum_records import (
    BarrenPlateauWitnessReceipt,
    QuantumCircuitExecutionReceipt,
    QuantumShotNoiseAcceptanceSpec,
    QuantumVariationalSeedTuple,
)
from .spend_records import SpendReservationBinding
from .study_records import DynamicsEvaluationContract


@runtime_checkable
class QuantumVariationalVerifierPort(Protocol):
    """Evaluates barren-plateau preflight and paired variational quantum arms."""

    def verify_barren_plateau_gate(
        self,
        *,
        companion_digest: str,
        split_pool_digest: str,
        seeds: tuple[QuantumVariationalSeedTuple, ...],
        threshold_grad_variance: float,
    ) -> BarrenPlateauWitnessReceipt:
        """Sample the parameter landscape and witness its gradient variance.

        Parameters
        ----------
        companion_digest : str
            SHA-256 of the ansatz's family companion contract.
        split_pool_digest : str
            SHA-256 of the split pool the preflight samples.
        seeds : tuple[QuantumVariationalSeedTuple, ...]
            The runs' seeds.
        threshold_grad_variance : float
            The smallest acceptable gradient variance.

        Returns
        -------
        BarrenPlateauWitnessReceipt
            The preflight witness.
        """
        ...

    def evaluate_variational_arm(
        self,
        *,
        candidate: ModelArtefactEnvelope,
        baseline: ModelArtefactEnvelope,
        candidate_receipt: QuantumCircuitExecutionReceipt,
        baseline_receipt: QuantumCircuitExecutionReceipt,
        split_digest: str,
        preregistration: PreregistrationRecord,
        acceptance_spec: QuantumShotNoiseAcceptanceSpec,
        seeds: tuple[QuantumVariationalSeedTuple, ...],
        spend_binding: SpendReservationBinding | None,
    ) -> SealedLookResult:
        """Evaluate a paired candidate and baseline on one sealed split.

        Parameters
        ----------
        candidate : ModelArtefactEnvelope
            The candidate arm.
        baseline : ModelArtefactEnvelope
            The baseline arm.
        candidate_receipt : QuantumCircuitExecutionReceipt
            The candidate's execution receipt.
        baseline_receipt : QuantumCircuitExecutionReceipt
            The baseline's execution receipt.
        split_digest : str
            SHA-256 of the sealed split.
        preregistration : PreregistrationRecord
            The study's preregistration.
        acceptance_spec : QuantumShotNoiseAcceptanceSpec
            The study's acceptance spec.
        seeds : tuple[QuantumVariationalSeedTuple, ...]
            The runs' seeds.
        spend_binding : SpendReservationBinding or None
            The reservation a credit-funded or paid run spends against.

        Returns
        -------
        SealedLookResult
            The look's result.
        """
        ...


@runtime_checkable
class PhysicsDynamicsVerifierPort(Protocol):
    """Evaluates physical dynamics trajectories, conformal coverage, and shift deferral."""

    def verify_dynamics_arm(
        self,
        *,
        candidate: ModelArtefactEnvelope,
        contract: DynamicsEvaluationContract,
        normaliser: DynamicsNormaliserEnvelope,
        split_digest: str,
    ) -> SealedLookResult:
        """Evaluate a dynamics arm under its evaluation contract.

        Parameters
        ----------
        candidate : ModelArtefactEnvelope
            The arm.
        contract : DynamicsEvaluationContract
            The dynamics evaluation contract.
        normaliser : DynamicsNormaliserEnvelope
            The train-only channel normaliser.
        split_digest : str
            SHA-256 of the sealed split.

        Returns
        -------
        SealedLookResult
            The look's result.
        """
        ...
