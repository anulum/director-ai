# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Training and Evaluation Ports

"""Ports that train, merge, export and evaluate models.

A training backend runs one recipe under one spend binding. A merge and
export port combines compatible adapters and produces a serving binary with
its parity report. An evaluation grader scores an arm on a non-sealed
selection split only, and a test-time search stays inside its inference
envelope. Adapters outside the kernel implement them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from .envelopes import ModelArtefactEnvelope
from .evaluator_outputs import SliceAggregateEstimate
from .family_companions import LanguageFamilyContract
from .graders import GraderSpec
from .ports_rewards import VerifiableRewardPort
from .recipe_records import PretrainingCheckpointState, TrainingRecipeManifest
from .spend_records import SpendReservationBinding
from .study_records import (
    ExportParityReport,
    InferenceComputeEnvelope,
    OnnxExportProfileSpec,
)


@runtime_checkable
class TrainingBackendPort(Protocol):
    """Executes reproducible pretraining, fine-tuning, or RLVR recipes."""

    def train(
        self,
        *,
        recipe: TrainingRecipeManifest,
        spend_binding: SpendReservationBinding,
        resume_checkpoint: PretrainingCheckpointState | None = None,
        output_dir: Path,
    ) -> ModelArtefactEnvelope:
        """Run one recipe and return the trained artefact's envelope.

        Parameters
        ----------
        recipe : TrainingRecipeManifest
            The recipe.
        spend_binding : SpendReservationBinding
            The reservation the run spends against.
        resume_checkpoint : PretrainingCheckpointState or None
            The checkpoint to resume from, if any.
        output_dir : Path
            Where the artefact is written.

        Returns
        -------
        ModelArtefactEnvelope
            The trained artefact.
        """
        ...


@runtime_checkable
class AdapterMergeAndExportPort(Protocol):
    """Merges compatible adapters and exports GGUF or ONNX serving artefacts."""

    def merge_adapters(
        self,
        *,
        parents: tuple[ModelArtefactEnvelope, ...],
        parent_contracts: tuple[LanguageFamilyContract, ...],
        method: Literal["linear", "ties", "dare_ties"],
        weights: tuple[float, ...],
        spend_binding: SpendReservationBinding,
        output_dir: Path,
    ) -> ModelArtefactEnvelope:
        """Merge compatible parent adapters into one artefact.

        Parameters
        ----------
        parents : tuple[ModelArtefactEnvelope, ...]
            The parents.
        parent_contracts : tuple[LanguageFamilyContract, ...]
            Their family contracts, in the same order.
        method : str
            The merge method.
        weights : tuple[float, ...]
            One weight per parent.
        spend_binding : SpendReservationBinding
            The reservation the merge spends against.
        output_dir : Path
            Where the merged artefact is written.

        Returns
        -------
        ModelArtefactEnvelope
            The merged artefact.
        """
        ...

    def export_and_verify(
        self,
        *,
        parent: ModelArtefactEnvelope,
        export_format: Literal["gguf", "onnx"],
        onnx_profile: OnnxExportProfileSpec | None,
        serving_runtime_id_and_version: str,
        probe_split_path: Path,
        spend_binding: SpendReservationBinding,
        output_dir: Path,
    ) -> tuple[ModelArtefactEnvelope, ExportParityReport]:
        """Export a serving binary and check its parity with the parent.

        Parameters
        ----------
        parent : ModelArtefactEnvelope
            The parent artefact.
        export_format : str
            GGUF or ONNX.
        onnx_profile : OnnxExportProfileSpec or None
            The ONNX profile, required for ONNX.
        serving_runtime_id_and_version : str
            The serving runtime.
        probe_split_path : Path
            The probe split the parity is measured on.
        spend_binding : SpendReservationBinding
            The reservation the export spends against.
        output_dir : Path
            Where the binary is written.

        Returns
        -------
        tuple[ModelArtefactEnvelope, ExportParityReport]
            The exported artefact and its parity report.
        """
        ...


@runtime_checkable
class EvaluationGraderPort(Protocol):
    """Evaluates candidate arms strictly on non-sealed selection splits."""

    def evaluate_selection(
        self,
        *,
        artefact: ModelArtefactEnvelope,
        split_path: Path,
        graders: tuple[GraderSpec, ...],
        inference_envelope: InferenceComputeEnvelope,
    ) -> tuple[SliceAggregateEstimate, ...]:
        """Score an arm on a selection split within its inference envelope.

        Parameters
        ----------
        artefact : ModelArtefactEnvelope
            The arm.
        split_path : Path
            A non-sealed selection split.
        graders : tuple[GraderSpec, ...]
            The graders.
        inference_envelope : InferenceComputeEnvelope
            The arm's test-time compute caps.

        Returns
        -------
        tuple[SliceAggregateEstimate, ...]
            The slice estimates.
        """
        ...


@runtime_checkable
class TestTimeVerifierSearchPort(Protocol):
    """Runs bounded best-of-N or verifier-guided search under an inference envelope."""

    def search_with_verifier(
        self,
        *,
        model: ModelArtefactEnvelope,
        prompt: str,
        envelope: InferenceComputeEnvelope,
        verifier: VerifiableRewardPort,
        seed: int,
    ) -> tuple[str, int]:
        """Search for the best verified response within the envelope.

        Parameters
        ----------
        model : ModelArtefactEnvelope
            The model.
        prompt : str
            The prompt.
        envelope : InferenceComputeEnvelope
            The test-time compute caps.
        verifier : VerifiableRewardPort
            The verifier that scores candidates.
        seed : int
            The search seed.

        Returns
        -------
        tuple[str, int]
            The chosen response and the number of verifier calls used.
        """
        ...
