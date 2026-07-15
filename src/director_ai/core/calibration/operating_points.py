# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Matched-FPR Operating Points (WCS-2a)

"""Pick matched-FPR operating points for the raw-support task routes.

The WCS-1 evidence campaign (``benchmarks/BENCHMARK_REPORT.md`` §16)
showed that evidence improvements do not change decisions while the
operating point stays a fixed composite-coherence cut: the calibration
layer absorbs everything the checker sees. The raw-support routes gate
the weakest-link claim support directly, and this module picks that
gate — the largest support threshold whose false-positive rate on
labelled GOOD responses does not exceed a deployment target.

Calibration flow per deployment::

    scorer = config.build_scorer()          # production checker + config
    points = calibrate_from_samples(
        scorer,
        samples,                            # (prompt, response, is_hallucinated)
        target_fpr_by_task={"dialogue": 0.045, "summarization": 0.025},
    )
    print(format_env_overlay(points))       # DIRECTOR_NLI_* lines

Supports are collected through
:meth:`~director_ai.core.scoring.scorer.CoherenceScorer.raw_task_support`,
which scores through the SAME premise composition and checker the
raw-support routes use in production — so the resulting threshold is
self-consistent with the decision path by construction.

Calibrate on samples whose prompts are composed exactly like the
production traffic that the gate will score. The WCS-2a proof run
(``benchmarks/BENCHMARK_REPORT.md`` §17) measured what happens
otherwise: benchmark-pair calibration realised roughly double the
target false-positive rate end-to-end, because thresholds this deep in
the support distribution's lower tail double their tail mass under a
small distribution shift. Prefer real traffic samples, and re-fit when
the premise composition, checker, or traffic mix changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "OperatingPoint",
    "calibrate_from_samples",
    "calibrate_operating_point",
    "config_overlay",
    "format_env_overlay",
    "matched_fpr_support_threshold",
]

#: Matched-FPR targets from the tracked 200-sample E2E baseline
#: (``benchmarks/results/judge_bench_nli_only_200.json``): the raw-support
#: routes hold the per-task false-positive rate the shipped defaults
#: already produce, so catch-rate comparisons stay like-for-like.
DEFAULT_TARGET_FPR: dict[str, float] = {
    "dialogue": 0.045,
    "summarization": 0.025,
}

#: Config fields the overlay writes per task route.
_TASK_FIELDS: dict[str, tuple[str, str, str]] = {
    "dialogue": (
        "nli_dialogue_support_threshold",
        "nli_dialogue_scoring",
        "raw_support",
    ),
    "summarization": (
        "nli_summarization_support_threshold",
        "nli_summarization_aggregation",
        "weakest_link",
    ),
}


@dataclass(frozen=True)
class OperatingPoint:
    """One calibrated matched-FPR operating point for a task route.

    Parameters
    ----------
    task:
        Task route the threshold applies to (``"dialogue"`` or
        ``"summarization"``).
    support_threshold:
        Largest support threshold whose false-positive rate on the good
        samples does not exceed *target_fpr*; a response is flagged when
        its weakest-link support falls strictly below it.
    target_fpr:
        Requested false-positive-rate ceiling.
    actual_fpr:
        Realised false-positive rate of the threshold on the good
        samples (at most *target_fpr* by construction).
    catch_rate:
        Fraction of hallucinated samples flagged at the threshold, or
        ``None`` when no hallucinated samples were supplied.
    n_good / n_bad:
        Sample counts behind the estimate — thresholds from thin
        samples are seeds, not calibrations.
    """

    task: str
    support_threshold: float
    target_fpr: float
    actual_fpr: float
    catch_rate: float | None
    n_good: int
    n_bad: int


def matched_fpr_support_threshold(
    good_supports: Sequence[float],
    target_fpr: float,
) -> float:
    """Largest threshold flagging at most *target_fpr* of good supports.

    A response is flagged hallucinated iff ``support < threshold``, so
    the threshold is the (⌊target·n⌋)-th smallest good support — every
    good support strictly below it is a false positive, and there are at
    most ⌊target·n⌋ of those by construction. This is the same order
    statistic the WCS-1 sweep used, so thresholds calibrated here read
    on the same scale as the tracked sweep artefacts.
    """
    if not good_supports:
        raise ValueError("matched_fpr_support_threshold() needs good supports")
    if not (0.0 <= target_fpr < 1.0):
        raise ValueError(f"target_fpr must be in [0, 1), got {target_fpr}")
    allowed = int(target_fpr * len(good_supports))
    return sorted(good_supports)[allowed]


def calibrate_operating_point(
    task: str,
    good_supports: Sequence[float],
    bad_supports: Sequence[float] = (),
    *,
    target_fpr: float,
) -> OperatingPoint:
    """Calibrate one task route from labelled weakest-link supports."""
    threshold = matched_fpr_support_threshold(good_supports, target_fpr)
    actual_fpr = sum(1 for s in good_supports if s < threshold) / len(good_supports)
    catch_rate = (
        sum(1 for s in bad_supports if s < threshold) / len(bad_supports)
        if bad_supports
        else None
    )
    return OperatingPoint(
        task=task,
        support_threshold=float(threshold),
        target_fpr=target_fpr,
        actual_fpr=actual_fpr,
        catch_rate=catch_rate,
        n_good=len(good_supports),
        n_bad=len(bad_supports),
    )


def calibrate_from_samples(
    scorer: Any,
    samples: Iterable[tuple[str, str, bool]],
    *,
    target_fpr_by_task: dict[str, float] | None = None,
) -> list[OperatingPoint]:
    """Calibrate every raw-support task route present in *samples*.

    Each sample is ``(prompt, response, is_hallucinated)``. Supports are
    collected through ``scorer.raw_task_support`` and bucketed by the
    task type the production router detects — samples that land on
    routes without a raw-support operating point (qa, rag, fact_check,
    default) are ignored. Tasks whose good-sample bucket is empty are
    skipped: a threshold needs good supports to bound the FPR.
    """
    targets = dict(DEFAULT_TARGET_FPR)
    if target_fpr_by_task:
        targets.update(target_fpr_by_task)

    buckets: dict[str, dict[bool, list[float]]] = {}
    for prompt, response, is_hallucinated in samples:
        task, support = scorer.raw_task_support(prompt, response)
        if task not in _TASK_FIELDS:
            continue
        buckets.setdefault(task, {False: [], True: []})[bool(is_hallucinated)].append(
            support
        )

    points: list[OperatingPoint] = []
    for task, by_label in sorted(buckets.items()):
        if not by_label[False]:
            continue
        points.append(
            calibrate_operating_point(
                task,
                by_label[False],
                by_label[True],
                target_fpr=targets[task],
            )
        )
    return points


def config_overlay(points: Sequence[OperatingPoint]) -> dict[str, float | str]:
    """Config field overlay enabling the calibrated raw-support routes."""
    overlay: dict[str, float | str] = {}
    for point in points:
        threshold_field, mode_field, mode_value = _TASK_FIELDS[point.task]
        overlay[threshold_field] = round(point.support_threshold, 6)
        overlay[mode_field] = mode_value
    return overlay


def format_env_overlay(points: Sequence[OperatingPoint]) -> str:
    """Render the overlay as ``DIRECTOR_*`` environment-variable lines."""
    lines = [
        f"DIRECTOR_{field.upper()}={value}"
        for field, value in config_overlay(points).items()
    ]
    return "\n".join(lines)
