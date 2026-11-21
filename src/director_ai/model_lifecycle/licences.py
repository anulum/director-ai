# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Licences and Budgets

"""Dual-axis licence records, compute budgets and the output-training check.

A :class:`DualAxisLicenceRecord` states, for one artefact, the licence of
its weights and the terms of its outputs, the class of its training data,
and three permissions. ``output_training_permitted`` is the single source of
truth for whether the artefact's outputs may train another model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_bool,
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
)
from .errors import LicenceRecordError, OutputTrainingNotPermittedError

CustodyDataClass = Literal[
    "licensed_public", "pilot", "internal_exception", "not_applicable"
]
CUSTODY_DATA_CLASSES: frozenset[str] = frozenset(
    {"licensed_public", "pilot", "internal_exception", "not_applicable"}
)
"""Every data class a custody record or licence record may name."""

MODEL_DATA_CLASSES: frozenset[str] = frozenset(
    {"licensed_public", "pilot", "internal_exception"}
)
"""Data classes allowed for model weights and the data they came from."""


@dataclass(frozen=True, slots=True)
class DualAxisLicenceRecord:
    """The weight licence and output terms of one model artefact.

    Attributes
    ----------
    schema : Literal["dual-axis-licence-record.v1"]
        Record schema.
    spdx_expression : str
        SPDX licence expression of the weights.
    weight_and_output_terms_digest : str
        SHA-256 of the full weight and output terms text.
    tdm_opted_out : bool
        Whether the rights holder reserved text and data mining.
    data_class : CustodyDataClass
        Class of the data the artefact was trained on.
    commercial_use_permitted : bool
        Whether commercial use is permitted.
    derivative_weights_permitted : bool
        Whether derived weights may be made and distributed.
    output_training_permitted : bool
        Whether the artefact's outputs may train another model.
    """

    schema: Literal["dual-axis-licence-record.v1"]
    spdx_expression: str
    weight_and_output_terms_digest: str
    tdm_opted_out: bool
    data_class: CustodyDataClass
    commercial_use_permitted: bool
    derivative_weights_permitted: bool
    output_training_permitted: bool


@dataclass(frozen=True, slots=True)
class ComputeAndTuningBudget:
    """The compute and tuning budget of a run, or what it spent.

    Attributes
    ----------
    schema : Literal["compute-and-tuning-budget.v1"]
        Record schema.
    max_wall_seconds : int
        Wall-clock seconds.
    max_accelerator_hours : float
        Accelerator hours.
    max_hyperparameter_trials : int
        Hyperparameter trials.
    max_sealed_looks : int
        Sealed looks.
    min_training_seeds : int
        Training seeds.
    """

    schema: Literal["compute-and-tuning-budget.v1"]
    max_wall_seconds: int
    max_accelerator_hours: float
    max_hyperparameter_trials: int
    max_sealed_looks: int
    min_training_seeds: int


def validate_licence_record(record: DualAxisLicenceRecord) -> DualAxisLicenceRecord:
    """Check a licence record's fields.

    Parameters
    ----------
    record : DualAxisLicenceRecord
        The record to check.

    Returns
    -------
    DualAxisLicenceRecord
        The same record.

    Raises
    ------
    LicenceRecordError
        If a field is malformed or the data class is not a model data class.
    MalformedDigestError
        If the terms digest is malformed.
    """
    if not isinstance(record, DualAxisLicenceRecord):
        raise LicenceRecordError("a DualAxisLicenceRecord is required")
    require_choice(
        record.schema, "schema", {"dual-axis-licence-record.v1"}, LicenceRecordError
    )
    require_text(record.spdx_expression, "spdx_expression", LicenceRecordError)
    require_sha256(
        record.weight_and_output_terms_digest, "weight_and_output_terms_digest"
    )
    require_choice(
        record.data_class, "data_class", MODEL_DATA_CLASSES, LicenceRecordError
    )
    for name in (
        "tdm_opted_out",
        "commercial_use_permitted",
        "derivative_weights_permitted",
        "output_training_permitted",
    ):
        require_bool(getattr(record, name), name, LicenceRecordError)
    return record


def validate_compute_budget(budget: ComputeAndTuningBudget) -> ComputeAndTuningBudget:
    """Check that every budget axis is a non-negative number.

    Parameters
    ----------
    budget : ComputeAndTuningBudget
        The budget to check.

    Returns
    -------
    ComputeAndTuningBudget
        The same budget.

    Raises
    ------
    LicenceRecordError
        If an axis is negative, non-finite or of the wrong type.
    """
    if not isinstance(budget, ComputeAndTuningBudget):
        raise LicenceRecordError("a ComputeAndTuningBudget is required")
    require_choice(
        budget.schema, "schema", {"compute-and-tuning-budget.v1"}, LicenceRecordError
    )
    for name in (
        "max_wall_seconds",
        "max_hyperparameter_trials",
        "max_sealed_looks",
        "min_training_seeds",
    ):
        require_int(getattr(budget, name), name, LicenceRecordError, minimum=0)
    hours = require_finite(
        budget.max_accelerator_hours, "max_accelerator_hours", LicenceRecordError
    )
    if hours < 0.0:
        raise LicenceRecordError("max_accelerator_hours must not be negative")
    return budget


def require_output_training_permitted(teacher: DualAxisLicenceRecord) -> None:
    """Refuse training on a teacher's outputs unless its licence permits it.

    Parameters
    ----------
    teacher : DualAxisLicenceRecord
        The licence record of the artefact whose outputs would train
        another model.

    Raises
    ------
    OutputTrainingNotPermittedError
        If ``output_training_permitted`` is not ``True``.
    LicenceRecordError
        If the record itself is malformed.
    """
    validate_licence_record(teacher)
    if teacher.output_training_permitted is not True:
        raise OutputTrainingNotPermittedError(
            "the teacher's licence does not permit training on its outputs"
        )
