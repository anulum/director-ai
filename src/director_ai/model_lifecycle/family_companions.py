# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Family Companion Contracts

"""Companion contracts of the language and dynamics families.

Every artefact names its family's companion contract by schema and digest
on its envelope. The kernel owns the language and dynamics contracts; the
spiking and quantum contracts belong to their domain repositories, and the
kernel checks only what their tensors must satisfy.

A hybrid model names auxiliary contracts of other families as
:class:`SecondaryFamilyContractRef` entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_bool,
    require_choice,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
)
from .errors import EnvelopeError


@dataclass(frozen=True, slots=True)
class SecondaryFamilyContractRef:
    """Reference to an auxiliary family companion contract on a hybrid model."""

    schema: str
    digest: str


@dataclass(frozen=True, slots=True)
class LanguageFamilyContract:
    """Companion contract required for every language-family artefact."""

    schema: Literal["language-family-contract.v1"]
    tokeniser_envelope_digest: str
    architecture_spec_digest: str | None
    embedding_space_digest: str | None
    tie_word_embeddings: bool
    secondary_family_contracts: tuple[SecondaryFamilyContractRef, ...]


@dataclass(frozen=True, slots=True)
class DynamicsFamilyContract:
    """Companion contract required for every physical-dynamics artefact."""

    schema: Literal["dynamics-family-contract.v1"]
    normaliser_envelope_digest: str
    architecture_spec_digest: str | None
    timestep_microseconds: int | None
    state_units_ucum: tuple[str, ...]
    action_units_ucum: tuple[str, ...]
    actuation_mode: Literal["review_only_non_actuating", "simulator_closed_loop_pilot"]
    secondary_family_contracts: tuple[SecondaryFamilyContractRef, ...]


ACTUATION_MODES: frozenset[str] = frozenset(
    {"review_only_non_actuating", "simulator_closed_loop_pilot"}
)
"""How a dynamics model may act: review only, or a closed loop in a simulator."""


def _validate_secondary(refs: object) -> None:
    """Check a tuple of secondary companion contract references."""
    items = require_tuple(refs, "secondary_family_contracts", EnvelopeError)
    digests: list[object] = []
    for ref in items:
        if not isinstance(ref, SecondaryFamilyContractRef):
            raise EnvelopeError(
                "secondary_family_contracts must hold SecondaryFamilyContractRef"
            )
        require_text(ref.schema, "secondary contract schema", EnvelopeError)
        digests.append(require_sha256(ref.digest, "secondary contract digest"))
    require_unique(digests, "secondary_family_contracts", EnvelopeError)


def _optional_digest(value: object, field: str) -> None:
    """Check a digest that may be ``None``."""
    if value is not None:
        require_sha256(value, field)


def validate_language_family_contract(
    contract: LanguageFamilyContract,
) -> LanguageFamilyContract:
    """Check a language family companion contract.

    Parameters
    ----------
    contract : LanguageFamilyContract
        The contract to check.

    Returns
    -------
    LanguageFamilyContract
        The same contract.

    Raises
    ------
    EnvelopeError
        If a field is malformed or a secondary contract repeats.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(contract, LanguageFamilyContract):
        raise EnvelopeError("a LanguageFamilyContract is required")
    require_choice(
        contract.schema, "schema", {"language-family-contract.v1"}, EnvelopeError
    )
    require_sha256(contract.tokeniser_envelope_digest, "tokeniser_envelope_digest")
    _optional_digest(contract.architecture_spec_digest, "architecture_spec_digest")
    _optional_digest(contract.embedding_space_digest, "embedding_space_digest")
    require_bool(contract.tie_word_embeddings, "tie_word_embeddings", EnvelopeError)
    _validate_secondary(contract.secondary_family_contracts)
    return contract


def _require_units(value: object, field: str) -> tuple[object, ...]:
    """Check a tuple of non-empty UCUM unit strings; units may repeat."""
    units = require_tuple(value, field, EnvelopeError)
    for unit in units:
        require_text(unit, field, EnvelopeError)
    return units


def validate_dynamics_family_contract(
    contract: DynamicsFamilyContract,
) -> DynamicsFamilyContract:
    """Check a dynamics family companion contract.

    Parameters
    ----------
    contract : DynamicsFamilyContract
        The contract to check.

    Returns
    -------
    DynamicsFamilyContract
        The same contract.

    Raises
    ------
    EnvelopeError
        If a field is malformed, the state has no unit, or a secondary
        contract repeats.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(contract, DynamicsFamilyContract):
        raise EnvelopeError("a DynamicsFamilyContract is required")
    require_choice(
        contract.schema, "schema", {"dynamics-family-contract.v1"}, EnvelopeError
    )
    require_sha256(contract.normaliser_envelope_digest, "normaliser_envelope_digest")
    _optional_digest(contract.architecture_spec_digest, "architecture_spec_digest")
    if contract.timestep_microseconds is not None:
        require_int(
            contract.timestep_microseconds,
            "timestep_microseconds",
            EnvelopeError,
            minimum=1,
        )
    if not _require_units(contract.state_units_ucum, "state_units_ucum"):
        raise EnvelopeError("a dynamics model needs at least one state unit")
    _require_units(contract.action_units_ucum, "action_units_ucum")
    require_choice(
        contract.actuation_mode, "actuation_mode", ACTUATION_MODES, EnvelopeError
    )
    _validate_secondary(contract.secondary_family_contracts)
    return contract
