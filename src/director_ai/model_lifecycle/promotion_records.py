# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Promotion Records

"""Records a promotion or a study decline depends on, with their shape checks.

:class:`ServingMemoryBudget` is the memory a candidate needs where it will
serve: its weights, cache and runtime overhead must fit the hard ceiling.
:class:`PromotionDependencyPins` names the digests a promotion depends on,
which custody must keep pinned. :class:`StudyDeclineRecord` is written when
no arm clears every gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_bool,
    require_choice,
    require_digests,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
    require_utc_timestamp,
)
from .errors import PromotionDecisionError

E = PromotionDecisionError


@dataclass(frozen=True, slots=True)
class ServingMemoryBudget:
    """Vendor-neutral device VRAM or host RAM budget gate."""

    schema: Literal["serving-memory-budget.v1"]
    target_host: Literal["gpu_vram", "host_ram"]
    weights_bytes: int
    kv_or_state_cache_bytes: int
    runtime_overhead_bytes: int
    hard_ceiling_bytes: int


@dataclass(frozen=True, slots=True)
class PromotionDependencyPins:
    """Scoped hot-tier custody pins required across all promotion witnesses."""

    schema: Literal["promotion-dependency-pins.v1"]
    pin_scope: str
    family_contract_digest: str
    secondary_family_contract_digests: tuple[str, ...]
    witness_and_parity_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StudyDeclineRecord:
    """Explicit study decline record emitted when zero arms pass all gates."""

    schema: Literal["model-study-decline.v1"]
    study_id: str
    preregistration_digest: str
    sealed_look_result_digest: str
    failed_arm_reasons: tuple[tuple[str, tuple[str, ...]], ...]
    null_control_rejected: bool
    negative_controls_failed_target_gates: bool
    declined_at_utc: str


def validate_serving_memory_budget(budget: ServingMemoryBudget) -> ServingMemoryBudget:
    """Check that a candidate's serving memory fits its hard ceiling.

    Parameters
    ----------
    budget : ServingMemoryBudget
        The budget to check.

    Returns
    -------
    ServingMemoryBudget
        The same budget.

    Raises
    ------
    PromotionDecisionError
        If weights, cache and overhead exceed the ceiling, or a field is
        malformed.
    """
    if not isinstance(budget, ServingMemoryBudget):
        raise E("a ServingMemoryBudget is required")
    require_choice(budget.schema, "schema", {"serving-memory-budget.v1"}, E)
    require_choice(budget.target_host, "target_host", {"gpu_vram", "host_ram"}, E)
    weights = require_int(budget.weights_bytes, "weights_bytes", E, minimum=1)
    cache = require_int(
        budget.kv_or_state_cache_bytes, "kv_or_state_cache_bytes", E, minimum=0
    )
    overhead = require_int(
        budget.runtime_overhead_bytes, "runtime_overhead_bytes", E, minimum=0
    )
    ceiling = require_int(budget.hard_ceiling_bytes, "hard_ceiling_bytes", E, minimum=1)
    if weights + cache + overhead > ceiling:
        raise E("weights, cache and overhead exceed the serving memory ceiling")
    return budget


def validate_dependency_pins(pins: PromotionDependencyPins) -> PromotionDependencyPins:
    """Check a promotion's dependency pins.

    Parameters
    ----------
    pins : PromotionDependencyPins
        The pins to check.

    Returns
    -------
    PromotionDependencyPins
        The same pins.

    Raises
    ------
    PromotionDecisionError
        If the scope is empty, no witness is pinned, or a digest repeats.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(pins, PromotionDependencyPins):
        raise E("a PromotionDependencyPins is required")
    require_choice(pins.schema, "schema", {"promotion-dependency-pins.v1"}, E)
    require_text(pins.pin_scope, "pin_scope", E)
    require_sha256(pins.family_contract_digest, "family_contract_digest")
    secondary = require_digests(
        pins.secondary_family_contract_digests,
        "secondary_family_contract_digests",
        E,
        non_empty=False,
    )
    witnesses = require_digests(
        pins.witness_and_parity_digests, "witness_and_parity_digests", E, non_empty=True
    )
    require_unique(
        [pins.family_contract_digest, *secondary, *witnesses], "pinned digests", E
    )
    return pins


def validate_study_decline(record: StudyDeclineRecord) -> StudyDeclineRecord:
    """Check a study decline record; every failed arm names its reasons.

    Parameters
    ----------
    record : StudyDeclineRecord
        The record to check.

    Returns
    -------
    StudyDeclineRecord
        The same record.

    Raises
    ------
    PromotionDecisionError
        If no arm is named, an arm repeats or names no reason, or a field is
        malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(record, StudyDeclineRecord):
        raise E("a StudyDeclineRecord is required")
    require_choice(record.schema, "schema", {"model-study-decline.v1"}, E)
    require_text(record.study_id, "study_id", E)
    require_sha256(record.preregistration_digest, "preregistration_digest")
    require_sha256(record.sealed_look_result_digest, "sealed_look_result_digest")
    arms: list[str] = []
    for pair in require_tuple(record.failed_arm_reasons, "failed_arm_reasons", E):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise E("failed_arm_reasons must hold (arm, reasons) pairs")
        arms.append(require_text(pair[0], "failed arm", E))
        reasons = require_tuple(pair[1], f"reasons of {pair[0]}", E)
        if not reasons:
            raise E(f"failed arm {pair[0]!r} names no reason")
        for reason in reasons:
            require_text(reason, f"reason of {pair[0]}", E)
    if not arms:
        raise E("a decline names every failed arm")
    require_unique(arms, "failed arms", E)
    for name in ("null_control_rejected", "negative_controls_failed_target_gates"):
        require_bool(getattr(record, name), name, E)
    require_utc_timestamp(record.declined_at_utc, "declined_at_utc", E)
    return record
