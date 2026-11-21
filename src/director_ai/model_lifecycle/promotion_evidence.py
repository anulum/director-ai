# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Promotion Evidence and Study Gates

"""Which artefact each arm is, and the study gates E-4, E-5 and M-J4-12.

A preregistration names its arms and, separately, the artefacts its looks
freeze, but does not join them. :class:`PromotionEvidence` supplies the
join, and the promoted arm for a promotion. The join is a caller-supplied
trust, like the completeness of a custody journal: the kernel checks that
the arms are exactly the declared arms and that the digests are distinct
and frozen, not that a given arm produced a given artefact. A later
preregistration version should freeze the join with the candidates. The
baseline arm is not a trust: the build-2 terms fix it before any look.

E-4 requires at least three training seeds and noise floors well below
every margin. Three test-retest flip-rate ceilings are at most each rate
metric's margin in basis points. Each non-rate metric's native-unit margin
is at least three noise floors, and a non-rate metric never has a
basis-point margin. Non-rate values are compared as the decimals they were
declared as, so ``0.3`` is three times ``0.1``. The ceiling, the floors and
the seed count are declared, not measured: they are trusts, and a ceiling
or floor of 0 claims that the evaluation is deterministic for that metric.

E-5 requires every arm to hold the same axes equal, at equal declared
values, and a matched-budget study to hold at least one training-compute
axis, so that an arm's effect is not its training compute. A Pareto study
may hold none; its promotion is a Pareto claim, not a matched-budget one.
M-J4-12 requires the promoted arm's test-time compute to stay within the
baseline's on all four caps, unless the study compares several budgets on
a Pareto front.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from ._checks import (
    require_choice,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
)
from .errors import PromotionDecisionError
from .evaluation_contracts import PreregistrationRecord
from .study_records import PreregistrationBuild2Terms

E = PromotionDecisionError
MIN_TRAINING_SEEDS = 3
"""E-4: the fewest training seeds a promotion's study may use."""
NOISE_FLOOR_MULTIPLE = 3
"""E-4: how many noise floors or flip-rate ceilings a margin must span."""
TRAINING_COMPUTE_AXES: frozenset[str] = frozenset(
    {"training_tokens", "gradient_steps", "accelerator_seconds"}
)
"""E-5: a matched-budget study holds at least one of these equal."""
ENVELOPE_CAPS = (
    "max_generated_tokens_per_item",
    "max_rollouts_n",
    "max_verifier_calls_per_item",
    "max_wall_ms_per_item",
)


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    """The join of a study's arms to their artefacts.

    Attributes
    ----------
    schema : Literal["promotion-evidence.v1"]
        Record schema.
    promoted_arm : str or None
        The arm a promotion promotes; ``None`` for a decline.
    arm_artefact_digests : tuple[tuple[str, str], ...]
        ``(arm, artefact digest)`` for every declared arm, a caller-supplied
        trust.
    """

    schema: Literal["promotion-evidence.v1"]
    promoted_arm: str | None
    arm_artefact_digests: tuple[tuple[str, str], ...]


def validate_promotion_evidence(evidence: PromotionEvidence) -> PromotionEvidence:
    """Check the evidence record's shape.

    Parameters
    ----------
    evidence : PromotionEvidence
        The evidence to check.

    Returns
    -------
    PromotionEvidence
        The same evidence.

    Raises
    ------
    PromotionDecisionError
        If an arm or digest repeats, or a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(evidence, PromotionEvidence):
        raise E("a PromotionEvidence is required")
    require_choice(evidence.schema, "schema", {"promotion-evidence.v1"}, E)
    arms: list[str] = []
    digests: list[str] = []
    for pair in require_tuple(evidence.arm_artefact_digests, "arm_artefact_digests", E):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise E("arm_artefact_digests must hold (arm, digest) pairs")
        arms.append(require_text(pair[0], "arm", E))
        digests.append(require_sha256(pair[1], "arm artefact digest"))
    require_unique(arms, "arms", E)
    require_unique(digests, "arm artefact digests", E)
    if evidence.promoted_arm is not None:
        require_text(evidence.promoted_arm, "promoted_arm", E)
    return evidence


def evidence_arm_digests(
    evidence: PromotionEvidence, preregistration: PreregistrationRecord
) -> Mapping[str, str]:
    """Return each arm's artefact digest after checking the join against the plan.

    Parameters
    ----------
    evidence : PromotionEvidence
        The evidence.
    preregistration : PreregistrationRecord
        The study's validated preregistration.

    Returns
    -------
    Mapping[str, str]
        Arm to artefact digest.

    Raises
    ------
    PromotionDecisionError
        If the evidence is malformed, does not name exactly the declared
        arms, names an artefact the looks did not freeze, or names a
        promoted arm that is not declared.
    MalformedDigestError
        If a digest is malformed.
    """
    validate_promotion_evidence(evidence)
    arms = dict(evidence.arm_artefact_digests)
    if set(arms) != set(preregistration.declared_arms):
        raise E("the evidence must name exactly the declared arms")
    if not set(arms.values()) <= set(preregistration.frozen_look_candidate_digests):
        raise E("an arm's artefact is not among the looks' frozen candidates")
    if evidence.promoted_arm is not None and evidence.promoted_arm not in arms:
        raise E("the promoted arm is not a declared arm")
    return arms


def _declared(value: float) -> Decimal:
    """Return a float as the decimal it was declared and serialised as."""
    return Decimal(repr(value))


def spans_noise_floors(margin: float, floor: float) -> bool:
    """Return whether ``margin`` is at least ``NOISE_FLOOR_MULTIPLE`` floors.

    Parameters
    ----------
    margin : float
        A margin in the metric's native unit.
    floor : float
        The metric's noise floor in the same unit.

    Returns
    -------
    bool
        Whether the margin spans the floors, compared as declared decimals.
    """
    return _declared(margin) >= NOISE_FLOOR_MULTIPLE * _declared(floor)


def check_noise_floors(
    terms: PreregistrationBuild2Terms, preregistration: PreregistrationRecord
) -> None:
    """Refuse a study whose seeds or noise floors break E-4.

    The seed count, the flip-rate ceiling and the noise floors are declared
    trusts, compared with the declared margins; nothing here measures them.
    A ceiling or floor of 0 claims a deterministic evaluation.

    Parameters
    ----------
    terms : PreregistrationBuild2Terms
        The study's validated build-2 terms.
    preregistration : PreregistrationRecord
        The study's validated preregistration.

    Raises
    ------
    PromotionDecisionError
        If the study has fewer than three seeds, a non-rate metric has a
        basis-point margin, or a margin spans fewer than three flip-rate
        ceilings or noise floors.
    """
    if terms.min_training_seeds < MIN_TRAINING_SEEDS:
        raise E(f"E-4: a promotion needs at least {MIN_TRAINING_SEEDS} training seeds")
    floors = dict(terms.non_rate_metric_noise_floors)
    rate_margins = dict(preregistration.non_inferiority_margins_bps)
    if set(floors) & set(rate_margins):
        raise E("E-4: a non-rate metric must not have a basis-point margin")
    ceiling = terms.test_retest_flip_rate_ceiling_bps
    for metric, margin_bps in rate_margins.items():
        if NOISE_FLOOR_MULTIPLE * ceiling > margin_bps:
            raise E(
                f"E-4: the margin of {metric} spans fewer than 3 flip-rate ceilings"
            )
    margins = dict(terms.non_rate_metric_margins)
    for metric, floor in floors.items():
        if not spans_noise_floors(margins[metric], floor):
            raise E(f"E-4: the margin of {metric} spans fewer than 3 noise floors")


def check_held_equal_axes(terms: PreregistrationBuild2Terms) -> None:
    """Refuse a study whose arms do not hold the same axes equal (E-5).

    Parameters
    ----------
    terms : PreregistrationBuild2Terms
        The study's validated build-2 terms.

    Raises
    ------
    PromotionDecisionError
        If two arms hold different axes equal, differ on a held axis, or a
        matched-budget study holds no training-compute axis.
    """
    budgets = [budget for _, budget in terms.arm_matched_budgets]
    axes = frozenset(budgets[0].held_equal_axes)
    for budget in budgets[1:]:
        if frozenset(budget.held_equal_axes) != axes:
            raise E("E-5: every arm must hold the same axes equal")
    if not terms.pareto_multi_budget_study and not axes & TRAINING_COMPUTE_AXES:
        raise E("E-5: a matched-budget study holds a training-compute axis equal")
    for axis in sorted(axes):
        if len({getattr(budget, axis) for budget in budgets}) != 1:
            raise E(f"E-5: the arms differ on the held-equal axis {axis}")


def check_inference_envelope(
    terms: PreregistrationBuild2Terms, *, promoted_arm: str
) -> None:
    """Refuse a promoted arm whose test-time compute exceeds the terms' baseline's.

    Parameters
    ----------
    terms : PreregistrationBuild2Terms
        The study's validated build-2 terms.
    promoted_arm : str
        The promoted arm.

    Raises
    ------
    PromotionDecisionError
        If the study is not a Pareto multi-budget study and a cap of the
        promoted arm exceeds the baseline's (M-J4-12).
    """
    if terms.pareto_multi_budget_study:
        return
    envelopes = dict(terms.arm_inference_envelopes)
    promoted, baseline = envelopes[promoted_arm], envelopes[terms.baseline_arm]
    for cap in ENVELOPE_CAPS:
        if getattr(promoted, cap) > getattr(baseline, cap):
            raise E(f"M-J4-12: the promoted arm's {cap} exceeds the baseline's")
