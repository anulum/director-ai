# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Contract Errors

"""Typed refusals raised by the model lifecycle contract kernel.

Every refusal derives from :class:`ModelLifecycleError`, a ``ValueError``,
so callers can catch the whole family or one named case. Messages name the
field or rule that failed and never carry sealed content.
"""

from __future__ import annotations


class ModelLifecycleError(ValueError):
    """Base class of every model lifecycle contract refusal."""


class NonCanonicalJsonError(ModelLifecycleError):
    """A value cannot be encoded or decoded under the canonical JSON rule."""


class NonCanonicalFloatError(NonCanonicalJsonError):
    """A JSON float is not finite or is a negative zero."""


class MalformedDigestError(ModelLifecycleError):
    """A digest field is not 64 lowercase hexadecimal characters."""


class UnsupportedSafetensorsDtypeError(ModelLifecycleError):
    """A safetensors tensor uses a dtype outside the closed set."""


class MalformedWeightsError(ModelLifecycleError):
    """A safetensors file or weight archive breaks the hardened parser."""


class NonFiniteTensorError(MalformedWeightsError):
    """A floating-point tensor holds a NaN or an infinity."""


class NegativeZeroTensorError(MalformedWeightsError):
    """A floating-point tensor holds a negative zero where its family forbids it."""


class UnsupportedWeightFormatError(MalformedWeightsError):
    """Weights are in a format the artefact's family does not accept."""


class MissingRequiredTensorPrefixError(MalformedWeightsError):
    """A family's required tensor prefix has no tensor, or a tensor has no allowed prefix."""


class MissingFixedPointDeclarationError(MalformedWeightsError):
    """An integer ``qat.`` spiking tensor has no fixed-point declaration."""


class InconsistentFixedPointDeclarationError(MalformedWeightsError):
    """A fixed-point declaration disagrees with its tensor or names no such tensor."""


class ArchitectureSpecError(ModelLifecycleError):
    """A transformer architecture spec breaks a numeric rule or its wire profile."""


class LicenceRecordError(ModelLifecycleError):
    """A dual-axis licence record or compute budget is malformed."""


class OutputTrainingNotPermittedError(ModelLifecycleError):
    """A teacher's licence does not permit training on its outputs."""


class EnvelopeError(ModelLifecycleError):
    """A model artefact envelope or lineage edge is malformed."""


class CustodyRecordError(ModelLifecycleError):
    """A custody ingest record breaks its schema or the kind rules."""


class MissingTensorSetDigestError(CustodyRecordError):
    """A family model identity needs a tensor-set digest the ingest lacks."""


class UnverifiedCustodyArtefactError(ModelLifecycleError):
    """A digest named by a request has no custody ingest record."""


class EvaluationContractError(ModelLifecycleError):
    """A preregistration, grader, pool, look or result record is malformed."""


class SealedPoolCustodyError(ModelLifecycleError):
    """A sealed split pool or its approval is not held by the right roles."""


class PreregistrationOrderError(ModelLifecycleError):
    """Custody ingest order contradicts the preregistration."""


class LookBudgetExhaustedError(ModelLifecycleError):
    """The lineage has no look left on the requested split."""


class SampleSizeError(ModelLifecycleError):
    """A paired sample-size input is outside its valid range."""


class PromotionDecisionError(ModelLifecycleError):
    """A promotion decision is malformed or its pre-state does not hold."""
