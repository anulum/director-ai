# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Study Term Validators

"""Validators of a build-2 study's terms and its arms' test-time compute caps.

The terms fix the study's seeds, flip-rate ceiling, non-rate noise floors and
margins, baseline arm, and each arm's matched budget and inference envelope.
The dynamics evaluation contract's rules are in :mod:`.dynamics_contracts`.
"""

from __future__ import annotations

from ._checks import (
    require_bool,
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
)
from .envelopes import ACTIVE_MODEL_FAMILIES
from .errors import EvaluationContractError
from .recipe_parts import validate_matched_budget
from .study_records import (
    InferenceComputeEnvelope,
    PreregistrationBuild2Terms,
)

E = EvaluationContractError
BPS_SCALE = 10_000
"""Basis points in a whole."""


def validate_inference_envelope(
    envelope: InferenceComputeEnvelope,
) -> InferenceComputeEnvelope:
    """Check an arm's per-item test-time compute caps.

    Parameters
    ----------
    envelope : InferenceComputeEnvelope
        The caps to check.

    Returns
    -------
    InferenceComputeEnvelope
        The same caps.

    Raises
    ------
    EvaluationContractError
        If a cap is below its minimum or a field is malformed.
    """
    if not isinstance(envelope, InferenceComputeEnvelope):
        raise E("an InferenceComputeEnvelope is required")
    require_choice(envelope.schema, "schema", {"inference-compute-envelope.v1"}, E)
    require_int(
        envelope.max_generated_tokens_per_item,
        "max_generated_tokens_per_item",
        E,
        minimum=1,
    )
    require_int(envelope.max_rollouts_n, "max_rollouts_n", E, minimum=1)
    require_int(
        envelope.max_verifier_calls_per_item,
        "max_verifier_calls_per_item",
        E,
        minimum=0,
    )
    require_int(envelope.max_wall_ms_per_item, "max_wall_ms_per_item", E, minimum=1)
    return envelope


def named_table(value: object, field: str) -> list[tuple[str, object]]:
    """Return ``(name, entry)`` pairs with distinct, non-empty names, else refuse."""
    pairs: list[tuple[str, object]] = []
    for pair in require_tuple(value, field, E):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise E(f"{field} must hold (name, value) pairs")
        pairs.append((require_text(pair[0], f"{field} name", E), pair[1]))
    require_unique([name for name, _ in pairs], f"{field} names", E)
    return pairs


def validate_build2_terms(
    terms: PreregistrationBuild2Terms,
) -> PreregistrationBuild2Terms:
    """Check a build-2 study's terms.

    Parameters
    ----------
    terms : PreregistrationBuild2Terms
        The terms to check.

    Returns
    -------
    PreregistrationBuild2Terms
        The same terms.

    Raises
    ------
    EvaluationContractError
        If the flip-rate ceiling is outside ``[0, 10000]`` basis points, a
        noise floor or non-rate margin is negative, the margins and noise
        floors name different metrics, the two arm tables name different
        arms, the baseline is not one of them, the family's evaluation
        contract is missing or stray, or a field is malformed.
    RecipeContractError
        If an arm's matched budget is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(terms, PreregistrationBuild2Terms):
        raise E("a PreregistrationBuild2Terms is required")
    require_choice(terms.schema, "schema", {"preregistration-build2-terms.v1"}, E)
    require_sha256(terms.search_spaces_digest, "search_spaces_digest")
    family = require_choice(terms.family, "family", ACTIVE_MODEL_FAMILIES, E)
    require_int(terms.min_training_seeds, "min_training_seeds", E, minimum=1)
    require_choice(
        terms.seed_aggregation_rule,
        "seed_aggregation_rule",
        {"every_seed_must_pass", "median_seed"},
        E,
    )
    ceiling = require_int(
        terms.test_retest_flip_rate_ceiling_bps,
        "test_retest_flip_rate_ceiling_bps",
        E,
        minimum=0,
    )
    if ceiling > BPS_SCALE:
        raise E("test_retest_flip_rate_ceiling_bps must not exceed 10000")
    floors = named_table(
        terms.non_rate_metric_noise_floors, "non_rate_metric_noise_floors"
    )
    margins = named_table(terms.non_rate_metric_margins, "non_rate_metric_margins")
    for _, value in [*floors, *margins]:
        if require_finite(value, "non-rate floor or margin", E) < 0.0:
            raise E("a non-rate noise floor or margin must not be negative")
    if {metric for metric, _ in floors} != {metric for metric, _ in margins}:
        raise E("every non-rate metric needs both a noise floor and a margin")
    budgets = named_table(terms.arm_matched_budgets, "arm_matched_budgets")
    envelopes = named_table(terms.arm_inference_envelopes, "arm_inference_envelopes")
    if not budgets:
        raise E("a study needs at least one arm")
    if {arm for arm, _ in budgets} != {arm for arm, _ in envelopes}:
        raise E("every arm needs both a matched budget and an inference envelope")
    if require_text(terms.baseline_arm, "baseline_arm", E) not in dict(budgets):
        raise E("the baseline arm is not one of the study's arms")
    for _, budget in terms.arm_matched_budgets:
        validate_matched_budget(budget)
    for _, envelope in terms.arm_inference_envelopes:
        validate_inference_envelope(envelope)
    require_bool(terms.pareto_multi_budget_study, "pareto_multi_budget_study", E)
    require_text(
        terms.predeclared_promotion_eligible_arm_rule,
        "predeclared_promotion_eligible_arm_rule",
        E,
    )
    for name in ("dynamics_eval_contract_digest", "quantum_acceptance_spec_digest"):
        if getattr(terms, name) is not None:
            require_sha256(getattr(terms, name), name)
    if (family == "dynamics") != (terms.dynamics_eval_contract_digest is not None):
        raise E("a dynamics study, and only one, names its evaluation contract")
    if (family == "quantum_variational") != (
        terms.quantum_acceptance_spec_digest is not None
    ):
        raise E("a variational quantum study, and only one, names its acceptance spec")
    return terms
