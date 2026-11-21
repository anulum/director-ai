# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Family Registry

"""The registered model families and their tensor rules.

Each family has a companion contract whose schema starts with the family's
prefix, and a rule for its weight formats, tensor-name prefixes, dtypes and
whether a negative zero is refused in its float tensors. NaN and infinities
are refused in every family. Negative zero is refused only for
``quantum_variational`` and ``spiking``, whose exporters canonicalise it;
trained language and dynamics weights keep it.

Spiking tensors may be ``I16`` only under ``qat.``, and every integer
``qat.`` tensor needs a signed fixed-point declaration in the companion
contract; no other tensor may carry one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .envelopes import ActiveModelFamily, require_active_family


@dataclass(frozen=True, slots=True)
class FixedPointFormatDecl:
    """Signed fixed-point format required on every integer ``qat.`` spiking tensor."""

    signed: Literal[True]
    total_bits: Literal[16, 32, 64]
    fraction_bits: int


@dataclass(frozen=True, slots=True)
class RegisteredFamilyContractRule:
    """Custody and validator rule for one registered model family."""

    family_id: ActiveModelFamily
    companion_schema_prefix: str
    allowed_weight_formats: tuple[
        Literal["safetensors", "safetensors_tar", "none"], ...
    ]
    required_tensor_prefixes: tuple[str, ...]
    optional_tensor_prefixes: tuple[str, ...]
    default_allowed_tensor_dtypes: tuple[str, ...]
    prefix_allowed_dtypes: tuple[tuple[str, tuple[str, ...]], ...]
    forbid_negative_zero: bool
    requires_non_null_tensor_set_digest: bool


LANGUAGE_FAMILY_RULE = RegisteredFamilyContractRule(
    family_id="language",
    companion_schema_prefix="language-family-contract.",
    allowed_weight_formats=("safetensors", "safetensors_tar"),
    required_tensor_prefixes=(),
    optional_tensor_prefixes=(),
    default_allowed_tensor_dtypes=(
        "F64",
        "F32",
        "BF16",
        "F16",
        "F8_E4M3",
        "F8_E5M2",
        "I64",
        "I32",
        "I16",
        "I8",
        "U8",
    ),
    prefix_allowed_dtypes=(),
    forbid_negative_zero=False,
    requires_non_null_tensor_set_digest=True,
)

DYNAMICS_FAMILY_RULE = RegisteredFamilyContractRule(
    family_id="dynamics",
    companion_schema_prefix="dynamics-family-contract.",
    allowed_weight_formats=("safetensors", "safetensors_tar"),
    required_tensor_prefixes=(),
    optional_tensor_prefixes=(),
    default_allowed_tensor_dtypes=("F64", "F32", "BF16", "F16", "I64", "I32"),
    prefix_allowed_dtypes=(),
    forbid_negative_zero=False,
    requires_non_null_tensor_set_digest=True,
)

SPIKING_FAMILY_RULE = RegisteredFamilyContractRule(
    family_id="spiking",
    companion_schema_prefix="sc-neurocore.spiking-artefact-contract.",
    allowed_weight_formats=("safetensors", "safetensors_tar"),
    required_tensor_prefixes=("pop.", "edge."),
    optional_tensor_prefixes=("qat.", "readout."),
    default_allowed_tensor_dtypes=("F64", "F32", "I64", "I32"),
    prefix_allowed_dtypes=(("qat.", ("F64", "F32", "I64", "I32", "I16")),),
    forbid_negative_zero=True,
    requires_non_null_tensor_set_digest=True,
)

QUANTUM_VARIATIONAL_FAMILY_RULE = RegisteredFamilyContractRule(
    family_id="quantum_variational",
    companion_schema_prefix="scpn-qc.variational-artefact-contract.",
    allowed_weight_formats=("safetensors", "safetensors_tar"),
    required_tensor_prefixes=("ansatz.theta.",),
    optional_tensor_prefixes=(
        "observable.coeff.",
        "classical_head.",
        "discrete_structure.gate_idx.",
    ),
    default_allowed_tensor_dtypes=("F64", "F32", "I64"),
    prefix_allowed_dtypes=(),
    forbid_negative_zero=True,
    requires_non_null_tensor_set_digest=True,
)

FAMILY_CONTRACT_REGISTRY: dict[str, RegisteredFamilyContractRule] = {
    rule.family_id: rule
    for rule in (
        LANGUAGE_FAMILY_RULE,
        DYNAMICS_FAMILY_RULE,
        SPIKING_FAMILY_RULE,
        QUANTUM_VARIATIONAL_FAMILY_RULE,
    )
}
"""The rule of every registered family, by family name."""

FIXED_POINT_PREFIX = "qat."
"""The only tensor prefix whose integer tensors carry fixed-point declarations."""

INTEGER_BITS: dict[str, int] = {"I16": 16, "I32": 32, "I64": 64}
"""Bit width of each integer dtype that may carry a fixed-point declaration."""


def family_rule(family: object) -> RegisteredFamilyContractRule:
    """Return the registered rule of a family.

    Parameters
    ----------
    family : object
        A family name.

    Returns
    -------
    RegisteredFamilyContractRule
        The family's rule.

    Raises
    ------
    EnvelopeError
        If the family is not registered.
    """
    return FAMILY_CONTRACT_REGISTRY[require_active_family(family)]
