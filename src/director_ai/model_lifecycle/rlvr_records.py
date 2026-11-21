# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Verifiable-Reward Records

"""Records of verifiable-reward policy optimisation.

A deterministic verifier (code, proof, math or hardware design) scores each
rollout. A candidate's own failure, including running out of time, memory or
output, is a failure with reward zero. Only a fault of the verifier's own
infrastructure masks a rollout, and only with an attestation from the outer
harness. :class:`RlvrAlgorithmSpec` bounds how many rollouts a batch may mask.
The rules are in :mod:`.rlvr_contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VerifiableOutcomeStatus = Literal[
    "verified_pass",
    "verified_fail",
    "candidate_resource_exceeded",
    "verifier_infrastructure_fault",
]
OUTCOME_STATUSES: frozenset[str] = frozenset(
    {
        "verified_pass",
        "verified_fail",
        "candidate_resource_exceeded",
        "verifier_infrastructure_fault",
    }
)
FAULT_KINDS: frozenset[str] = frozenset(
    {"host_io_error", "missing_verifier_binary", "licence_daemon_unreachable"}
)
RLVR_ALGORITHMS: frozenset[str] = frozenset(
    {"grpo_unbiased", "rloo", "expert_iteration_sft"}
)
MAX_MASKED_FRACTION_CEILING = 0.05
"""The highest per-batch masked fraction a spec may allow."""
MAX_DRIFT_ALPHA = 0.10
"""The highest significance level of the masking drift test."""


@dataclass(frozen=True, slots=True)
class VerifierFaultAttestation:
    """Attestation emitted solely by the outer verifier harness outside candidate pgrp."""

    schema: Literal["verifier-fault-attestation.v1"]
    fault_kind: Literal[
        "host_io_error",
        "missing_verifier_binary",
        "licence_daemon_unreachable",
    ]
    harness_pid: int
    candidate_caused_excluded: Literal[True]
    observed_at_utc: str


@dataclass(frozen=True, slots=True)
class VerifiableRewardOutcome:
    """Single rollout outcome from a deterministic code, proof, math, or RTL verifier."""

    schema: Literal["verifiable-reward-outcome.v1"]
    prompt_id: str
    rollout_index: int
    status: VerifiableOutcomeStatus
    reward: float | None
    fault_attestation_digest: str | None
    response_token_length: int
    wall_ms: int


@dataclass(frozen=True, slots=True)
class RlvrAlgorithmSpec:
    """Group-relative / RLOO verifiable-reward policy optimisation contract."""

    schema: Literal["rlvr-algorithm-spec.v1"]
    algorithm: Literal["grpo_unbiased", "rloo", "expert_iteration_sft"]
    group_size_g: int
    min_valid_group_rollouts: int
    loss_length_normaliser: Literal["fixed_max_response_tokens"]
    clip_eps_low: float
    clip_eps_high: float
    kl_anchor_beta: float
    reference_checkpoint_digest: str
    drop_zero_variance_groups: bool
    conciseness_bonus_on_pass_only: float
    max_masked_rollout_fraction_per_batch: float
    max_masked_fraction_drift_vs_reference: float
    masked_drift_alpha: float
