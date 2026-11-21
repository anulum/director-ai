# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Training Recipe Validators

"""Validators of training recipes, their seeds, hardware, curriculum and budget.

A recipe's four seeds are distinct and below 2**63. ``bf16`` is refused on
the capability classes without native bf16. Curriculum stages are numbered
``0..N-1`` and each stage's mixture weights sum to exactly one in decimal
arithmetic. Training from scratch names an architecture and no base
checkpoint; every other mode starts from at least one base. A
verifiable-reward recipe, and only that one, carries its algorithm spec.
"""

from __future__ import annotations

from decimal import Decimal

from ._checks import (
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
)
from .architecture_contracts import validate_transformer_architecture_spec
from .architecture_specs import POOL_FUNDING_KINDS
from .envelopes import require_active_family
from .errors import RecipeContractError
from .recipe_parts import (
    validate_hardware_profile,
    validate_matched_budget,
    validate_seed_quadruple,
)
from .recipe_records import (
    ADAPTATION_MODES,
    CURRICULUM_PHASES,
    REPRODUCIBILITY_MODES,
    CurriculumStageSpec,
    PretrainingCheckpointState,
    TrainingRecipeManifest,
)
from .rlvr_contracts import validate_rlvr_spec

E = RecipeContractError


def validate_curriculum_stage(stage: CurriculumStageSpec) -> CurriculumStageSpec:
    """Check one curriculum stage; its mixture weights sum to exactly one.

    Parameters
    ----------
    stage : CurriculumStageSpec
        The stage to check.

    Returns
    -------
    CurriculumStageSpec
        The same stage.

    Raises
    ------
    RecipeContractError
        If the weights do not sum to one as decimals, a weight is not above
        zero, a shard repeats, or a field is malformed.
    """
    if not isinstance(stage, CurriculumStageSpec):
        raise E("a CurriculumStageSpec is required")
    require_choice(stage.schema, "schema", {"curriculum-stage.v1"}, E)
    require_int(stage.stage_index, "stage_index", E, minimum=0)
    require_choice(stage.phase, "phase", CURRICULUM_PHASES, E)
    require_int(stage.sequence_length, "sequence_length", E, minimum=1)
    require_int(stage.token_budget, "token_budget", E, minimum=1)
    if require_finite(stage.peak_or_target_lr, "peak_or_target_lr", E) <= 0.0:
        raise E("peak_or_target_lr must be above zero")
    if stage.rope_theta_override is not None and (
        require_finite(stage.rope_theta_override, "rope_theta_override", E) <= 0.0
    ):
        raise E("rope_theta_override must be above zero")
    total = Decimal(0)
    shards: list[str] = []
    for pair in require_tuple(stage.shard_mixture_weights, "shard_mixture_weights", E):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise E("shard_mixture_weights must hold (shard, weight) pairs")
        shards.append(require_text(pair[0], "mixture shard", E))
        weight = require_finite(pair[1], "mixture weight", E)
        if weight <= 0.0:
            raise E("mixture weights must be above zero")
        total += Decimal(str(weight))
    require_unique(shards, "mixture shards", E)
    if total != Decimal(1):
        raise E(f"stage {stage.stage_index} mixture weights sum to {total}, not 1")
    return stage


def _check_mode(recipe: TrainingRecipeManifest) -> None:
    """Check the base checkpoints, architecture and RLVR spec against the mode."""
    mode = recipe.adaptation_mode
    bases = require_tuple(recipe.base_checkpoint_digests, "base_checkpoint_digests", E)
    for digest in bases:
        require_sha256(digest, "base_checkpoint_digests")
    require_unique(bases, "base_checkpoint_digests", E)
    if mode == "pretrained_from_scratch":
        if bases or recipe.architecture_spec is None:
            raise E(
                "training from scratch needs an architecture and no base checkpoint"
            )
    elif not bases:
        raise E(f"adaptation mode {mode!r} needs at least one base checkpoint")
    if recipe.architecture_spec is not None:
        validate_transformer_architecture_spec(
            recipe.architecture_spec, funding_kind=recipe.funding_kind
        )
        if recipe.architecture_spec.family != recipe.family:
            raise E("the architecture belongs to another family")
    if (mode == "rlvr_policy_optimisation") != (recipe.rlvr_spec is not None):
        raise E("an RLVR spec belongs exactly to rlvr_policy_optimisation recipes")
    if recipe.rlvr_spec is not None:
        validate_rlvr_spec(recipe.rlvr_spec)


def validate_recipe_and_curriculum(
    recipe: TrainingRecipeManifest,
) -> TrainingRecipeManifest:
    """Check a training recipe end to end.

    Parameters
    ----------
    recipe : TrainingRecipeManifest
        The recipe to check.

    Returns
    -------
    TrainingRecipeManifest
        The same recipe.

    Raises
    ------
    RecipeContractError
        If the seeds, hardware, curriculum, budget or mode rules fail, or a
        field is malformed.
    ArchitectureSpecError
        If the architecture spec is refused on the recipe's funding pool.
    RlvrContractError
        If the RLVR spec is malformed.
    MalformedDigestError
        If a digest is malformed.
    EnvelopeError
        If the family is not registered.
    """
    if not isinstance(recipe, TrainingRecipeManifest):
        raise E("a TrainingRecipeManifest is required")
    require_choice(recipe.schema, "schema", {"training-recipe-manifest.v1"}, E)
    require_active_family(recipe.family)
    require_choice(recipe.adaptation_mode, "adaptation_mode", ADAPTATION_MODES, E)
    require_choice(recipe.funding_kind, "funding_kind", POOL_FUNDING_KINDS, E)
    require_choice(
        recipe.reproducibility_mode, "reproducibility_mode", REPRODUCIBILITY_MODES, E
    )
    validate_seed_quadruple(recipe.seeds)
    validate_hardware_profile(recipe.hardware_profile)
    require_sha256(recipe.dedup_report_digest, "dedup_report_digest")
    require_sha256(recipe.family_contract_digest, "family_contract_digest")
    require_tuple(recipe.curriculum_stages, "curriculum_stages", E)
    for position, stage in enumerate(recipe.curriculum_stages):
        if validate_curriculum_stage(stage).stage_index != position:
            raise E("curriculum stage indices must run 0..N-1 in order")
    validate_matched_budget(recipe.matched_budget)
    _check_mode(recipe)
    return recipe


def validate_pretraining_checkpoint_state(
    state: PretrainingCheckpointState,
) -> PretrainingCheckpointState:
    """Check the resumption state of a pretraining run.

    Parameters
    ----------
    state : PretrainingCheckpointState
        The state to check.

    Returns
    -------
    PretrainingCheckpointState
        The same state.

    Raises
    ------
    RecipeContractError
        If a count or the gradient-norm average is negative, an average is
        not finite, or a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(state, PretrainingCheckpointState):
        raise E("a PretrainingCheckpointState is required")
    require_choice(state.schema, "schema", {"pretraining-checkpoint-state.v1"}, E)
    for name in (
        "recipe_digest",
        "weights_tensor_set_digest",
        "optimiser_state_sha256",
        "dataloader_shard_digest",
        "rng_state_sha256",
    ):
        require_sha256(getattr(state, name), name)
    for name in ("step", "tokens_consumed", "dataloader_byte_offset"):
        require_int(getattr(state, name), name, E, minimum=0)
    require_finite(state.rolling_loss_ema, "rolling_loss_ema", E)
    if require_finite(state.grad_norm_ema, "grad_norm_ema", E) < 0.0:
        raise E("grad_norm_ema must not be negative")
    return state
