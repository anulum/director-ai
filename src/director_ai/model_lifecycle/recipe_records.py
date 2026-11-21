# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Training Recipe Records

"""Records of a reproducible training or adaptation recipe.

:class:`TrainingRecipeManifest` names everything a run depends on: the family,
the adaptation mode, the funding pool, four distinct seeds, a vendor-neutral
hardware capability class, the split audit, the family contract, the base
checkpoints, the architecture when training from scratch, the curriculum
stages, the verifiable-reward spec and the matched budget.
:class:`PretrainingCheckpointState` is what a preempted run resumes from.
The rules are in :mod:`.recipe_contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .architecture_specs import PoolFundingKind, TransformerArchitectureSpec
from .envelopes import ActiveModelFamily
from .rlvr_records import RlvrAlgorithmSpec

BUDGET_AXES: tuple[str, ...] = (
    "training_tokens",
    "gradient_steps",
    "accelerator_seconds",
    "dev_selection_evaluations",
    "teacher_or_frontier_tokens",
    "human_review_seconds",
)
"""The six matched-budget axes."""
ADAPTATION_MODES: frozenset[str] = frozenset(
    {
        "pretrained_from_scratch",
        "continued_pretraining",
        "full_safetensors",
        "peft_safetensors_adapter",
        "rlvr_policy_optimisation",
        "adapter_merge",
        "deterministic_word_patch",
    }
)
REPRODUCIBILITY_MODES: frozenset[str] = frozenset(
    {
        "bitwise_deterministic_eval_and_init",
        "bitwise_full_train",
        "statistical_ci_train",
    }
)
HARDWARE_TIERS: frozenset[str] = frozenset(
    {"rocm_consumer_gpu", "cuda_sm61_gpu", "cpu_x86_64_v2_avx", "cloud_accelerator"}
)
NO_BF16_TIERS: frozenset[str] = frozenset({"cuda_sm61_gpu", "cpu_x86_64_v2_avx"})
"""Capability classes without native bf16 arithmetic."""
CURRICULUM_PHASES: frozenset[str] = frozenset({"warmup", "stable", "decay"})


@dataclass(frozen=True, slots=True)
class MatchedBudgetAccountingSpec:
    """Six-axis resource accounting vector for matched-budget study arms."""

    schema: Literal["matched-budget-accounting.v1"]
    training_tokens: int
    gradient_steps: int
    accelerator_seconds: float
    dev_selection_evaluations: int
    teacher_or_frontier_tokens: int
    human_review_seconds: int
    held_equal_axes: tuple[
        Literal[
            "training_tokens",
            "gradient_steps",
            "accelerator_seconds",
            "dev_selection_evaluations",
            "teacher_or_frontier_tokens",
            "human_review_seconds",
        ],
        ...,
    ]


@dataclass(frozen=True, slots=True)
class SeedQuadruple:
    """Pairwise-distinct non-negative 63-bit RNG seeds for training runs."""

    schema: Literal["seed-quadruple.v1"]
    init_seed: int
    data_order_seed: int
    dropout_seed: int
    dataloader_worker_seed: int


@dataclass(frozen=True, slots=True)
class HardwareCapabilityProfile:
    """Vendor-neutral capability class, floating-point precision, and CPU ISA floor."""

    schema: Literal["hardware-capability-profile.v1"]
    target_tier: Literal[
        "rocm_consumer_gpu",
        "cuda_sm61_gpu",
        "cpu_x86_64_v2_avx",
        "cloud_accelerator",
    ]
    compute_dtype: Literal["fp32", "bf16"]
    host_cpu_isa_ceiling: Literal["x86-64-v2", "x86-64-v3"]
    diloco_inner_steps_h: int | None


@dataclass(frozen=True, slots=True)
class CurriculumStageSpec:
    """Single warmup-stable-decay curriculum stage with exact mixture weights."""

    schema: Literal["curriculum-stage.v1"]
    stage_index: int
    phase: Literal["warmup", "stable", "decay"]
    sequence_length: int
    token_budget: int
    peak_or_target_lr: float
    shard_mixture_weights: tuple[tuple[str, float], ...]
    rope_theta_override: float | None


@dataclass(frozen=True, slots=True)
class TrainingRecipeManifest:
    """Complete reproducible training or adaptation recipe manifest."""

    schema: Literal["training-recipe-manifest.v1"]
    family: ActiveModelFamily
    adaptation_mode: Literal[
        "pretrained_from_scratch",
        "continued_pretraining",
        "full_safetensors",
        "peft_safetensors_adapter",
        "rlvr_policy_optimisation",
        "adapter_merge",
        "deterministic_word_patch",
    ]
    funding_kind: PoolFundingKind
    reproducibility_mode: Literal[
        "bitwise_deterministic_eval_and_init",
        "bitwise_full_train",
        "statistical_ci_train",
    ]
    seeds: SeedQuadruple
    hardware_profile: HardwareCapabilityProfile
    dedup_report_digest: str
    family_contract_digest: str
    base_checkpoint_digests: tuple[str, ...]
    architecture_spec: TransformerArchitectureSpec | None
    curriculum_stages: tuple[CurriculumStageSpec, ...]
    rlvr_spec: RlvrAlgorithmSpec | None
    matched_budget: MatchedBudgetAccountingSpec


@dataclass(frozen=True, slots=True)
class PretrainingCheckpointState:
    """Exact resumption state for pretraining and preemptible cloud jobs."""

    schema: Literal["pretraining-checkpoint-state.v1"]
    recipe_digest: str
    step: int
    tokens_consumed: int
    weights_tensor_set_digest: str
    optimiser_state_sha256: str
    dataloader_shard_digest: str
    dataloader_byte_offset: int
    rng_state_sha256: str
    rolling_loss_ema: float
    grad_norm_ema: float
