# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Family Weight Checks

"""Family weight checks and the family model identity.

:func:`compute_family_tensor_set_identity` reads a model's weights under
its family's rule: the allowed formats, the negative-zero refusal of
``spiking`` and ``quantum_variational``, the tensor-name prefixes, the
dtypes per prefix and the spiking fixed-point table. NaN and infinities are
refused in every family by the reader itself.

:func:`compute_family_model_identity_digest` binds a model's tensor-set
digest to its family and companion contract.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from .custody_records import CustodyIngestRecord, validate_custody_ingest_record
from .digests import record_digest
from .envelopes import ModelArtefactEnvelope, validate_envelope
from .errors import (
    EnvelopeError,
    InconsistentFixedPointDeclarationError,
    MalformedWeightsError,
    MissingFixedPointDeclarationError,
    MissingRequiredTensorPrefixError,
    MissingTensorSetDigestError,
    UnsupportedWeightFormatError,
)
from .family_registry import (
    FIXED_POINT_PREFIX,
    INTEGER_BITS,
    FixedPointFormatDecl,
    RegisteredFamilyContractRule,
    family_rule,
)
from .safetensors_parser import read_tensor_set_descriptors
from .tensors import TensorDescriptor, TensorSetIdentity, tensor_set_identity
from .weight_archives import read_archive_tensor_descriptors

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
"""Leading bytes of an HDF5 file, such as a NIR ``.nir`` graph."""


def _refuse_hdf5(path: str | os.PathLike[str]) -> None:
    """Refuse a weights file that is HDF5, whatever its declared format."""
    with Path(path).open("rb") as handle:
        if handle.read(len(HDF5_MAGIC)) == HDF5_MAGIC:
            raise UnsupportedWeightFormatError(
                "HDF5 weights are not accepted; export NIR graphs as safetensors"
            )


def _validate_fixed_point(descriptor: TensorDescriptor, decl: object) -> None:
    """Check one fixed-point declaration against its integer ``qat.`` tensor."""
    bits = INTEGER_BITS[descriptor.dtype]
    if (
        not isinstance(decl, FixedPointFormatDecl)
        or decl.signed is not True
        or decl.total_bits != bits
        or type(decl.fraction_bits) is not int
        or not 0 <= decl.fraction_bits < decl.total_bits
    ):
        raise InconsistentFixedPointDeclarationError(
            f"tensor {descriptor.name!r} needs a signed {bits}-bit declaration"
            " with 0 <= fraction_bits < total_bits"
        )


def _allowed_dtypes(rule: RegisteredFamilyContractRule, name: str) -> tuple[str, ...]:
    """Return the dtypes a tensor of this name may use under the family's rule."""
    for prefix, dtypes in rule.prefix_allowed_dtypes:
        if name.startswith(prefix):
            return dtypes
    return rule.default_allowed_tensor_dtypes


def validate_family_tensor_layout(
    family: object,
    descriptors: Sequence[TensorDescriptor],
    *,
    fixed_point_map: Mapping[str, FixedPointFormatDecl],
) -> None:
    """Check a model's tensors against its family's prefixes, dtypes and fixed points.

    Parameters
    ----------
    family : object
        The artefact's family.
    descriptors : Sequence[TensorDescriptor]
        Every tensor of the model.
    fixed_point_map : Mapping[str, FixedPointFormatDecl]
        The companion contract's per-tensor fixed-point declarations; empty
        for a family whose tensors carry none.

    Raises
    ------
    MissingRequiredTensorPrefixError
        If a tensor has no allowed prefix, or a required prefix has no tensor.
    MalformedWeightsError
        If a tensor's dtype is not allowed under its prefix.
    MissingFixedPointDeclarationError
        If an integer ``qat.`` tensor has no declaration.
    InconsistentFixedPointDeclarationError
        If a declaration disagrees with its tensor or names any other tensor.
    EnvelopeError
        If the family is not registered.
    """
    rule = family_rule(family)
    prefixes = rule.required_tensor_prefixes + rule.optional_tensor_prefixes
    declared: set[str] = set()
    for descriptor in descriptors:
        name = descriptor.name
        if prefixes and not name.startswith(prefixes):
            raise MissingRequiredTensorPrefixError(
                f"tensor {name!r} has no prefix allowed in {rule.family_id!r}"
            )
        if descriptor.dtype not in _allowed_dtypes(rule, name):
            raise MalformedWeightsError(
                f"tensor {name!r} may not use dtype {descriptor.dtype!r}"
            )
        if (
            FIXED_POINT_PREFIX in prefixes
            and name.startswith(FIXED_POINT_PREFIX)
            and descriptor.dtype in INTEGER_BITS
        ):
            if name not in fixed_point_map:
                raise MissingFixedPointDeclarationError(
                    f"tensor {name!r} has no fixed-point declaration"
                )
            _validate_fixed_point(descriptor, fixed_point_map[name])
            declared.add(name)
    for prefix in rule.required_tensor_prefixes:
        if not any(item.name.startswith(prefix) for item in descriptors):
            raise MissingRequiredTensorPrefixError(
                f"family {rule.family_id!r} needs a {prefix!r} tensor"
            )
    extraneous = sorted(set(fixed_point_map) - declared)
    if extraneous:
        raise InconsistentFixedPointDeclarationError(
            f"fixed-point declarations name no integer qat. tensor: {extraneous}"
        )


def compute_family_tensor_set_identity(
    family: object,
    weight_format: str,
    paths: Sequence[str | os.PathLike[str]],
    *,
    fixed_point_map: Mapping[str, FixedPointFormatDecl] | None = None,
) -> TensorSetIdentity:
    """Read a model's weights under its family's rule and return their identity.

    Parameters
    ----------
    family : object
        The artefact's family.
    weight_format : str
        ``"safetensors"`` (one file or its shards) or ``"safetensors_tar"``
        (exactly one archive).
    paths : Sequence[str | os.PathLike[str]]
        The weights files.
    fixed_point_map : Mapping[str, FixedPointFormatDecl] or None, optional
        The companion contract's fixed-point table; ``None`` means empty.

    Returns
    -------
    TensorSetIdentity
        The identity of the checked tensor set.

    Raises
    ------
    UnsupportedWeightFormatError
        If the family does not accept the format or a file is HDF5.
    MalformedWeightsError
        If no file is given, a ``safetensors_tar`` is not one archive, or
        the weights break the reader or the family's tensor rules. A
        NaN, an infinity, or a forbidden negative zero raises the matching
        subclass.
    EnvelopeError
        If the family is not registered.
    """
    rule = family_rule(family)
    if weight_format not in rule.allowed_weight_formats:
        raise UnsupportedWeightFormatError(
            f"family {rule.family_id!r} does not accept weight format {weight_format!r}"
        )
    if not paths:
        raise MalformedWeightsError("a model needs at least one weights file")
    for path in paths:
        _refuse_hdf5(path)
    forbid = rule.forbid_negative_zero
    if weight_format == "safetensors":
        descriptors = read_tensor_set_descriptors(paths, forbid_negative_zero=forbid)
    elif len(paths) == 1:
        descriptors = read_archive_tensor_descriptors(
            paths[0], forbid_negative_zero=forbid
        )
    else:
        raise MalformedWeightsError("a safetensors_tar model is exactly one archive")
    validate_family_tensor_layout(
        family, descriptors, fixed_point_map=fixed_point_map or {}
    )
    return tensor_set_identity(descriptors)


def compute_family_model_identity_digest(
    envelope: ModelArtefactEnvelope, ingest_record: CustodyIngestRecord
) -> str:
    """Return the digest that binds a model's tensors to its family and contract.

    Parameters
    ----------
    envelope : ModelArtefactEnvelope
        The artefact's envelope.
    ingest_record : CustodyIngestRecord
        Its custody ingest record.

    Returns
    -------
    str
        SHA-256 of the canonical ``family-model-identity.v1`` document.

    Raises
    ------
    EnvelopeError
        If the envelope is malformed, the record belongs to another
        artefact, or the companion schema lacks the family's prefix.
    CustodyRecordError
        If the ingest record is malformed.
    MissingTensorSetDigestError
        If the ingest recorded no tensor-set digest (weight format ``none``).
    """
    validate_envelope(envelope)
    validate_custody_ingest_record(ingest_record)
    if ingest_record.content_digest != envelope.content_digest:
        raise EnvelopeError("the ingest record belongs to another artefact")
    if ingest_record.tensor_set_digest is None:
        raise MissingTensorSetDigestError(
            "a family model identity needs a tensor-set digest"
        )
    rule = family_rule(envelope.family)
    if not envelope.family_contract_schema.startswith(rule.companion_schema_prefix):
        raise EnvelopeError(
            f"family_contract_schema must start with {rule.companion_schema_prefix!r}"
        )
    return record_digest(
        {
            "schema": "family-model-identity.v1",
            "family": envelope.family,
            "tensor_set_digest": ingest_record.tensor_set_digest,
            "family_contract_schema": envelope.family_contract_schema,
            "family_contract_digest": envelope.family_contract_digest,
        }
    )
