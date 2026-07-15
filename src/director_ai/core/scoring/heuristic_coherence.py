# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Heuristic Coherence Orchestration
"""Route and combine heuristic-coherence component scores."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

DIVERGENCE_NEUTRAL: float = 0.5


class HeuristicCoherenceRoute(Enum):
    """Execution routes inside the no-model coherence orchestrator."""

    DIALOGUE = "dialogue"
    SUMMARISATION = "summarisation"
    FACTUAL_ONLY = "factual_only"
    PARALLEL_COMPONENTS = "parallel_components"


@dataclass(frozen=True)
class HeuristicCoherenceInputs:
    """Inputs that determine the heuristic-coherence execution route.

    Parameters
    ----------
    auto_dialogue_profile:
        Whether dialogue and summarisation auto-routing is enabled.
    use_prompt_as_premise:
        Whether the source prompt should be treated as the factual premise.
    nli_available:
        Whether a model-backed NLI scorer is loaded and usable.
    task_type:
        Task label returned by the scorer task detector.
    w_logic:
        Logical-divergence weight used by the scorer.
    """

    auto_dialogue_profile: bool
    use_prompt_as_premise: bool
    nli_available: bool
    task_type: str
    w_logic: float


def select_heuristic_coherence_route(
    inputs: HeuristicCoherenceInputs,
) -> HeuristicCoherenceRoute:
    """Select the scorer route without computing any component score.

    Dialogue is intentionally evaluated before summarisation because dialogue
    prompts can also be configured with prompt-as-premise scoring. The
    zero-logical-weight route remains available without NLI so summarisation
    profiles can skip the logical component in heuristic-only deployments.

    Parameters
    ----------
    inputs:
        Current scorer flags and task classification.

    Returns
    -------
    HeuristicCoherenceRoute
        Route that should compute the logical and factual components.
    """
    if (
        inputs.auto_dialogue_profile
        and not inputs.use_prompt_as_premise
        and inputs.nli_available
        and inputs.task_type == "dialogue"
    ):
        return HeuristicCoherenceRoute.DIALOGUE

    if inputs.nli_available and (
        (inputs.use_prompt_as_premise and inputs.w_logic < 1e-9)
        or (inputs.task_type == "summarization" and inputs.auto_dialogue_profile)
    ):
        return HeuristicCoherenceRoute.SUMMARISATION

    if inputs.w_logic < 1e-9:
        return HeuristicCoherenceRoute.FACTUAL_ONLY

    return HeuristicCoherenceRoute.PARALLEL_COMPONENTS


def combine_weighted_coherence(
    *,
    h_logic: float,
    h_factual: float,
    w_logic: float,
    w_fact: float,
    nli_available: bool,
    evidence_present: bool,
    dialogue_route: bool,
    raw_support_route: bool = False,
) -> float:
    """Combine component divergences into a calibrated coherence score.

    Parameters
    ----------
    h_logic:
        Logical divergence in the inclusive range [0, 1].
    h_factual:
        Factual divergence in the inclusive range [0, 1].
    w_logic:
        Weight assigned to logical divergence.
    w_fact:
        Weight assigned to factual divergence.
    nli_available:
        Whether the component scores came from an NLI-capable scorer.
    evidence_present:
        Whether factual retrieval returned evidence. Absence of evidence and a
        neutral factual score activates the no-KB calibration.
    dialogue_route:
        Whether the dialogue route produced the score. Dialogue calibration is
        already handled by the route-specific factual scorer.
    raw_support_route:
        Whether a raw-support route (WCS-2a) produced the factual score.
        Coherence is then the raw weakest-link support ``1 − h_factual``
        with no component weighting — the review gate compares it against
        a matched-FPR support operating point, so any reweighting here
        would silently move that calibrated operating point.

    Returns
    -------
    float
        Composite coherence score in the inclusive range [0, 1].
    """
    if raw_support_route:
        return max(0.0, min(1.0, 1.0 - h_factual))

    total_divergence = w_logic * h_logic + w_fact * h_factual
    coherence = 1.0 - total_divergence

    fact_is_neutral = abs(h_factual - DIVERGENCE_NEUTRAL) < 1e-9
    if (
        nli_available
        and fact_is_neutral
        and not evidence_present
        and not dialogue_route
    ):
        lo = 1.0 - w_logic - w_fact * DIVERGENCE_NEUTRAL
        hi = 1.0 - w_fact * DIVERGENCE_NEUTRAL
        span = hi - lo
        if span > 1e-9:
            coherence = (coherence - lo) / span

    return max(0.0, min(1.0, coherence))
