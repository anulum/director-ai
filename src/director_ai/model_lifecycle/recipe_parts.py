# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Recipe Part Validators

"""Validators of a recipe's seeds, hardware class and matched budget.

The four seeds are pairwise distinct and lie in ``[0, 2**63)``. ``bf16`` is
refused on the capability classes without native bf16. A matched budget's
axes are not negative and its held-equal axes are distinct known axes.
"""

from __future__ import annotations

from ._checks import (
    require_choice,
    require_finite,
    require_int,
    require_tuple,
    require_unique,
)
from .architecture_specs import MAX_SEED
from .errors import RecipeContractError
from .recipe_records import (
    BUDGET_AXES,
    HARDWARE_TIERS,
    NO_BF16_TIERS,
    HardwareCapabilityProfile,
    MatchedBudgetAccountingSpec,
    SeedQuadruple,
)

E = RecipeContractError


def validate_seed_quadruple(seeds: SeedQuadruple) -> SeedQuadruple:
    """Check that a run's four seeds are distinct and lie in ``[0, 2**63)``.

    Parameters
    ----------
    seeds : SeedQuadruple
        The seeds to check.

    Returns
    -------
    SeedQuadruple
        The same seeds.

    Raises
    ------
    RecipeContractError
        If a seed is outside the range, two seeds are equal, or a field is
        malformed.
    """
    if not isinstance(seeds, SeedQuadruple):
        raise E("a SeedQuadruple is required")
    require_choice(seeds.schema, "schema", {"seed-quadruple.v1"}, E)
    values = [
        require_int(getattr(seeds, name), name, E, minimum=0)
        for name in (
            "init_seed",
            "data_order_seed",
            "dropout_seed",
            "dataloader_worker_seed",
        )
    ]
    if max(values) >= MAX_SEED:
        raise E("seeds must lie below 2**63")
    if len(set(values)) != len(values):
        raise E("the four seeds must be pairwise distinct")
    return seeds


def validate_hardware_profile(
    profile: HardwareCapabilityProfile,
) -> HardwareCapabilityProfile:
    """Check a capability class and its precision.

    Parameters
    ----------
    profile : HardwareCapabilityProfile
        The profile to check.

    Returns
    -------
    HardwareCapabilityProfile
        The same profile.

    Raises
    ------
    RecipeContractError
        If ``bf16`` is requested on a class without it, or a field is
        malformed.
    """
    if not isinstance(profile, HardwareCapabilityProfile):
        raise E("a HardwareCapabilityProfile is required")
    require_choice(profile.schema, "schema", {"hardware-capability-profile.v1"}, E)
    tier = require_choice(profile.target_tier, "target_tier", HARDWARE_TIERS, E)
    dtype = require_choice(profile.compute_dtype, "compute_dtype", {"fp32", "bf16"}, E)
    require_choice(
        profile.host_cpu_isa_ceiling,
        "host_cpu_isa_ceiling",
        {"x86-64-v2", "x86-64-v3"},
        E,
    )
    if profile.diloco_inner_steps_h is not None:
        require_int(profile.diloco_inner_steps_h, "diloco_inner_steps_h", E, minimum=1)
    if dtype == "bf16" and tier in NO_BF16_TIERS:
        raise E(f"capability class {tier!r} has no native bf16")
    return profile


def validate_matched_budget(
    budget: MatchedBudgetAccountingSpec,
) -> MatchedBudgetAccountingSpec:
    """Check a matched-budget vector and its held-equal axes.

    Parameters
    ----------
    budget : MatchedBudgetAccountingSpec
        The budget to check.

    Returns
    -------
    MatchedBudgetAccountingSpec
        The same budget.

    Raises
    ------
    RecipeContractError
        If an axis is negative, a held-equal axis is unknown or repeats, or a
        field is malformed.
    """
    if not isinstance(budget, MatchedBudgetAccountingSpec):
        raise E("a MatchedBudgetAccountingSpec is required")
    require_choice(budget.schema, "schema", {"matched-budget-accounting.v1"}, E)
    for name in BUDGET_AXES:
        if name == "accelerator_seconds":
            if require_finite(budget.accelerator_seconds, name, E) < 0.0:
                raise E("accelerator_seconds must not be negative")
        else:
            require_int(getattr(budget, name), name, E, minimum=0)
    axes = require_tuple(budget.held_equal_axes, "held_equal_axes", E)
    for axis in axes:
        require_choice(axis, "held_equal_axes", frozenset(BUDGET_AXES), E)
    require_unique(axes, "held_equal_axes", E)
    return budget
