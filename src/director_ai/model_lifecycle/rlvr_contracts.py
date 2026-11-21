# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Verifiable-Reward Validators

"""Validators of verifiable-reward outcomes and of a batch's masking rate.

A pass earns a reward in ``(0, 1 + bonus]``; a failure, or a candidate that
exceeded its time, memory or output limits, earns exactly zero. A verifier
infrastructure fault carries no reward and must name the outer harness's
attestation. The batch masking rules are in :mod:`.rlvr_mask_rate`.
"""

from __future__ import annotations

from ._checks import (
    require_bool,
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
    require_utc_timestamp,
)
from .digests import record_digest
from .errors import (
    RlvrContractError,
    UnattestedVerifierFaultMaskError,
)
from .rlvr_records import (
    FAULT_KINDS,
    MAX_DRIFT_ALPHA,
    MAX_MASKED_FRACTION_CEILING,
    OUTCOME_STATUSES,
    RLVR_ALGORITHMS,
    RlvrAlgorithmSpec,
    VerifiableRewardOutcome,
    VerifierFaultAttestation,
)

E = RlvrContractError
FAULT = "verifier_infrastructure_fault"


def _non_negative(value: object, field: str) -> float:
    """Return a finite float of at least zero."""
    number = require_finite(value, field, E)
    if number < 0.0:
        raise E(f"{field} must not be negative")
    return number


def validate_rlvr_spec(spec: RlvrAlgorithmSpec) -> RlvrAlgorithmSpec:
    """Check a verifiable-reward algorithm spec.

    Parameters
    ----------
    spec : RlvrAlgorithmSpec
        The spec to check.

    Returns
    -------
    RlvrAlgorithmSpec
        The same spec.

    Raises
    ------
    RlvrContractError
        If the group is below 4, the valid minimum is outside ``[2, g]``, a
        clip bound is not above zero or the high one is below the low one,
        the masking ceiling exceeds 0.05, the drift test's alpha is outside
        ``(0, 0.10]``, or a field is malformed.
    MalformedDigestError
        If the reference checkpoint digest is malformed.
    """
    if not isinstance(spec, RlvrAlgorithmSpec):
        raise E("an RlvrAlgorithmSpec is required")
    require_choice(spec.schema, "schema", {"rlvr-algorithm-spec.v1"}, E)
    require_choice(spec.algorithm, "algorithm", RLVR_ALGORITHMS, E)
    group = require_int(spec.group_size_g, "group_size_g", E, minimum=4)
    valid = require_int(
        spec.min_valid_group_rollouts, "min_valid_group_rollouts", E, minimum=2
    )
    if valid > group:
        raise E("min_valid_group_rollouts must not exceed group_size_g")
    require_choice(
        spec.loss_length_normaliser,
        "loss_length_normaliser",
        {"fixed_max_response_tokens"},
        E,
    )
    low = require_finite(spec.clip_eps_low, "clip_eps_low", E)
    high = require_finite(spec.clip_eps_high, "clip_eps_high", E)
    if low <= 0.0 or high < low:
        raise E("clip bounds need 0 < clip_eps_low <= clip_eps_high")
    _non_negative(spec.kl_anchor_beta, "kl_anchor_beta")
    require_sha256(spec.reference_checkpoint_digest, "reference_checkpoint_digest")
    require_bool(spec.drop_zero_variance_groups, "drop_zero_variance_groups", E)
    _non_negative(spec.conciseness_bonus_on_pass_only, "conciseness_bonus_on_pass_only")
    ceiling = _non_negative(
        spec.max_masked_rollout_fraction_per_batch,
        "max_masked_rollout_fraction_per_batch",
    )
    if ceiling > MAX_MASKED_FRACTION_CEILING:
        raise E("max_masked_rollout_fraction_per_batch must not exceed 0.05")
    _non_negative(
        spec.max_masked_fraction_drift_vs_reference,
        "max_masked_fraction_drift_vs_reference",
    )
    alpha = require_finite(spec.masked_drift_alpha, "masked_drift_alpha", E)
    if not 0.0 < alpha <= MAX_DRIFT_ALPHA:
        raise E("masked_drift_alpha must lie in (0, 0.10]")
    return spec


def validate_verifier_fault_attestation(
    attestation: VerifierFaultAttestation,
) -> VerifierFaultAttestation:
    """Check an outer harness's attestation of a verifier infrastructure fault.

    Parameters
    ----------
    attestation : VerifierFaultAttestation
        The attestation to check.

    Returns
    -------
    VerifierFaultAttestation
        The same attestation.

    Raises
    ------
    RlvrContractError
        If a field is malformed or the candidate is not excluded as the cause.
    """
    if not isinstance(attestation, VerifierFaultAttestation):
        raise E("a VerifierFaultAttestation is required")
    require_choice(attestation.schema, "schema", {"verifier-fault-attestation.v1"}, E)
    require_choice(attestation.fault_kind, "fault_kind", FAULT_KINDS, E)
    require_int(attestation.harness_pid, "harness_pid", E, minimum=1)
    if attestation.candidate_caused_excluded is not True:
        raise E("an attestation must exclude the candidate as the cause")
    require_utc_timestamp(attestation.observed_at_utc, "observed_at_utc", E)
    return attestation


def validate_verifiable_reward_outcome(
    outcome: VerifiableRewardOutcome,
    *,
    spec: RlvrAlgorithmSpec,
    attestation: VerifierFaultAttestation | None = None,
) -> VerifiableRewardOutcome:
    """Check one rollout outcome's reward against its status (TR-1).

    Parameters
    ----------
    outcome : VerifiableRewardOutcome
        The outcome to check.
    spec : RlvrAlgorithmSpec
        The algorithm spec, for the conciseness bonus.
    attestation : VerifierFaultAttestation or None, optional
        The fault attestation, when the caller holds it; it must then be the
        one the outcome names.

    Returns
    -------
    VerifiableRewardOutcome
        The same outcome.

    Raises
    ------
    UnattestedVerifierFaultMaskError
        If a fault has a reward or no attestation digest, or the given
        attestation is not the one it names.
    RlvrContractError
        If a pass's reward is outside ``(0, 1 + bonus]``, a failure's reward is
        not exactly zero, a non-fault names an attestation, or the outcome or
        the spec is malformed.
    """
    validate_rlvr_spec(spec)
    if not isinstance(outcome, VerifiableRewardOutcome):
        raise E("a VerifiableRewardOutcome is required")
    require_choice(outcome.schema, "schema", {"verifiable-reward-outcome.v1"}, E)
    require_text(outcome.prompt_id, "prompt_id", E)
    for name in ("rollout_index", "response_token_length", "wall_ms"):
        require_int(getattr(outcome, name), name, E, minimum=0)
    status = require_choice(outcome.status, "status", OUTCOME_STATUSES, E)
    if status == FAULT:
        if outcome.reward is not None or outcome.fault_attestation_digest is None:
            raise UnattestedVerifierFaultMaskError(
                "a verifier fault carries no reward and names its attestation"
            )
        require_sha256(outcome.fault_attestation_digest, "fault_attestation_digest")
        if attestation is not None and (
            record_digest(validate_verifier_fault_attestation(attestation))
            != outcome.fault_attestation_digest
        ):
            raise UnattestedVerifierFaultMaskError(
                "the attestation is not the one named"
            )
        return outcome
    if outcome.fault_attestation_digest is not None:
        raise E(f"a {status} outcome names no fault attestation")
    reward = require_finite(outcome.reward, "reward", E)
    if status == "verified_pass":
        if not 0.0 < reward <= 1.0 + spec.conciseness_bonus_on_pass_only:
            raise E("a pass earns a reward in (0, 1 + conciseness bonus]")
    elif reward != 0.0:
        raise E(f"a {status} outcome earns exactly zero")
    return outcome
