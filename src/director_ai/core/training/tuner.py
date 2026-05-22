# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Threshold Tuner

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, cast

try:  # pragma: no cover - optional acceleration
    from backfire_kernel import rust_sum_i64

    _RUST_TUNER = True
except ImportError:  # pragma: no cover - mandatory accelerator guard
    _RUST_TUNER = True

    def rust_sum_i64(_values: list[int]) -> int:
        raise RuntimeError("backfire_kernel rust_sum_i64 is unavailable")


from ..mandatory import mandatory_execution
from ..scoring.scorer import CoherenceScorer

__all__ = [
    "BoundaryExample",
    "ThresholdCandidate",
    "TuneResult",
    "format_confidence_report",
    "format_profile_overlay",
    "tune",
]

DEFAULT_THRESHOLDS = [round(0.30 + i * 0.05, 2) for i in range(13)]  # 0.30..0.90
DEFAULT_WEIGHT_PAIRS = [(0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]
_REPORT_CANDIDATE_LIMIT = 5
_REPORT_EXAMPLE_LIMIT = 3


class _TuneResultLike(Protocol):
    """Protocol for objects that can render tuning reports."""

    threshold: float
    w_logic: float
    w_fact: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    samples: int


@dataclass(frozen=True)
class ThresholdCandidate:
    """Metrics for one evaluated threshold and weight pair."""

    threshold: float
    w_logic: float
    w_fact: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class BoundaryExample:
    """Sample closest to the selected decision boundary."""

    sample_index: int
    score: float
    label: bool
    predicted: bool
    margin: float
    counterfactual_threshold: float
    prompt_excerpt: str
    response_excerpt: str


@dataclass(frozen=True)
class _ScoredSample:
    """One labelled sample after scoring by a candidate scorer."""

    sample_index: int
    score: float
    label: bool
    prompt: str
    response: str


@dataclass
class TuneResult:
    """Selected threshold and diagnostic evidence from a tuning run."""

    threshold: float
    w_logic: float
    w_fact: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    samples: int
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    positive_samples: int = 0
    negative_samples: int = 0
    selection_margin: float = 0.0
    confidence_level: str = "low"
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    tradeoff_summary: str = ""
    evaluated_candidates: tuple[ThresholdCandidate, ...] = ()
    boundary_examples: tuple[BoundaryExample, ...] = ()

    def to_profile_overlay(
        self,
        *,
        profile: str = "tuned",
        base_profile: str = "",
    ) -> dict[str, object]:
        """Return a ready-to-use config overlay for this tuning result."""
        hard_limit = max(0.0, round(self.threshold - 0.10, 4))
        soft_limit = min(1.0, round(self.threshold + 0.10, 4))
        extra = {
            "tune_samples": str(self.samples),
            "tune_balanced_accuracy": f"{self.balanced_accuracy:.4f}",
            "tune_precision": f"{self.precision:.4f}",
            "tune_recall": f"{self.recall:.4f}",
            "tune_f1": f"{self.f1:.4f}",
            "tune_confidence_level": self.confidence_level,
            "tune_confidence_intervals": _format_confidence_intervals(
                self.confidence_intervals
            ),
            "tune_selection_margin": f"{self.selection_margin:.4f}",
            "tune_confusion_matrix": (
                f"tp={self.tp},fp={self.fp},tn={self.tn},fn={self.fn}"
            ),
            "tune_class_balance": (
                f"positive={self.positive_samples},negative={self.negative_samples}"
            ),
            "tune_tradeoff_summary": self.tradeoff_summary,
        }
        if self.boundary_examples:
            extra["tune_boundary_examples"] = "; ".join(
                _format_boundary_example(example)
                for example in self.boundary_examples[:_REPORT_EXAMPLE_LIMIT]
            )
        if base_profile:
            extra["tuned_from_profile"] = base_profile
        return {
            "profile": profile,
            "coherence_threshold": self.threshold,
            "hard_limit": hard_limit,
            "soft_limit": soft_limit,
            "w_logic": self.w_logic,
            "w_fact": self.w_fact,
            "extra": extra,
        }


def format_profile_overlay(
    result: object,
    *,
    profile: str = "tuned",
    base_profile: str = "",
) -> str:
    """Render a deterministic YAML profile overlay for CLI output."""
    overlay = _to_profile_overlay(
        result,
        profile=profile,
        base_profile=base_profile,
    )
    extra = overlay.pop("extra")
    result_like = cast("_TuneResultLike", result)
    samples = int(result_like.samples)
    balanced_accuracy = float(result_like.balanced_accuracy)
    precision = float(result_like.precision)
    recall = float(result_like.recall)
    f1 = float(result_like.f1)
    lines = [
        "# Director-AI tuned profile overlay",
        "# Load with DirectorConfig.from_yaml(...) or merge over a base profile.",
    ]
    if base_profile:
        lines.append(f"# Base profile: {base_profile}")
    report = format_confidence_report(result)
    if report:
        lines.extend(f"# {line}" if line else "#" for line in report.splitlines())
    lines.extend(
        [
            f"# Samples: {samples}",
            f"# Balanced accuracy: {balanced_accuracy:.4f}",
            f"# Precision: {precision:.4f}",
            f"# Recall: {recall:.4f}",
            f"# F1: {f1:.4f}",
        ],
    )
    for key, value in overlay.items():
        lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("extra:")
    if not isinstance(extra, dict):
        raise TypeError("training report extra metadata must be a mapping")
    for key, value in extra.items():
        lines.append(f"  {key}: {_yaml_scalar(value)}")
    lines.append("")
    return "\n".join(lines)


def format_confidence_report(result: object) -> str:
    """Render selected-threshold diagnostics for CLI and YAML comments."""
    result_like = cast("_TuneResultLike", result)
    confidence_level = str(getattr(result, "confidence_level", "low"))
    selection_margin = float(getattr(result, "selection_margin", 0.0))
    tradeoff_summary = str(getattr(result, "tradeoff_summary", ""))
    confidence_intervals = cast(
        "dict[str, tuple[float, float]]",
        getattr(result, "confidence_intervals", {}),
    )
    tp = int(getattr(result, "tp", 0))
    fp = int(getattr(result, "fp", 0))
    tn = int(getattr(result, "tn", 0))
    fn = int(getattr(result, "fn", 0))
    positive = int(getattr(result, "positive_samples", 0))
    negative = int(getattr(result, "negative_samples", 0))
    candidates = tuple(
        cast(
            "tuple[ThresholdCandidate, ...]",
            getattr(result, "evaluated_candidates", ()),
        )
    )
    examples = tuple(
        cast("tuple[BoundaryExample, ...]", getattr(result, "boundary_examples", ()))
    )

    lines = [
        "Confidence report:",
        (
            f"- Selected threshold {result_like.threshold:.4f} with "
            f"w_logic={result_like.w_logic:.2f}, w_fact={result_like.w_fact:.2f}."
        ),
        (
            f"- Confidence level: {confidence_level} "
            f"(selection margin {selection_margin:.4f} BA)."
        ),
        f"- Class balance: {positive} approved / {negative} rejected labels.",
        f"- Confusion matrix at selection: TP={tp}, FP={fp}, TN={tn}, FN={fn}.",
    ]
    if tradeoff_summary:
        lines.append(f"- Trade-off: {tradeoff_summary}")
    if confidence_intervals:
        lines.append(
            "- 95% Wilson intervals: "
            f"{_format_confidence_intervals(confidence_intervals)}"
        )
    if candidates:
        lines.append("- Top candidate thresholds:")
        for candidate in candidates[:_REPORT_CANDIDATE_LIMIT]:
            lines.append(
                "  "
                f"t={candidate.threshold:.4f}, "
                f"w=({candidate.w_logic:.2f},{candidate.w_fact:.2f}), "
                f"BA={candidate.balanced_accuracy:.4f}, "
                f"P={candidate.precision:.4f}, "
                f"R={candidate.recall:.4f}, "
                f"FPR={candidate.false_positive_rate:.4f}, "
                f"FNR={candidate.false_negative_rate:.4f}"
            )
    if examples:
        lines.append("- Boundary examples:")
        for example in examples[:_REPORT_EXAMPLE_LIMIT]:
            lines.append(f"  {_format_boundary_example(example)}")
    return "\n".join(lines)


def _to_profile_overlay(
    result: object,
    *,
    profile: str,
    base_profile: str,
) -> dict[str, object]:
    """Return a profile overlay for modern or legacy tuning results."""
    to_overlay = getattr(result, "to_profile_overlay", None)
    if callable(to_overlay):
        return cast(
            "dict[str, object]",
            to_overlay(profile=profile, base_profile=base_profile),
        )

    result_like = cast("_TuneResultLike", result)
    threshold = float(result_like.threshold)
    balanced_accuracy = float(result_like.balanced_accuracy)
    precision = float(result_like.precision)
    recall = float(result_like.recall)
    f1 = float(result_like.f1)
    samples = int(result_like.samples)
    extra = {
        "tune_samples": str(samples),
        "tune_balanced_accuracy": f"{balanced_accuracy:.4f}",
        "tune_precision": f"{precision:.4f}",
        "tune_recall": f"{recall:.4f}",
        "tune_f1": f"{f1:.4f}",
        "tune_confidence_level": str(getattr(result, "confidence_level", "low")),
        "tune_confidence_intervals": _format_confidence_intervals(
            cast(
                "dict[str, tuple[float, float]]",
                getattr(result, "confidence_intervals", {}),
            )
        ),
        "tune_selection_margin": f"{float(getattr(result, 'selection_margin', 0.0)):.4f}",
    }
    tradeoff_summary = str(getattr(result, "tradeoff_summary", ""))
    if tradeoff_summary:
        extra["tune_tradeoff_summary"] = tradeoff_summary
    if base_profile:
        extra["tuned_from_profile"] = base_profile
    return {
        "profile": profile,
        "coherence_threshold": threshold,
        "hard_limit": max(0.0, round(threshold - 0.10, 4)),
        "soft_limit": min(1.0, round(threshold + 0.10, 4)),
        "w_logic": float(result_like.w_logic),
        "w_fact": float(result_like.w_fact),
        "extra": extra,
    }


def tune(
    samples: list[dict],
    thresholds: list[float] | None = None,
    weight_pairs: list[tuple[float, float]] | None = None,
    use_nli: bool = False,
    ground_truth_store=None,
) -> TuneResult:
    """Grid-search over thresholds and weight pairs, maximize balanced accuracy.

    Each sample: ``{"prompt": str, "response": str, "label": bool}``
    where ``label=True`` means the response is correct (should be approved).
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    thresholds = thresholds or DEFAULT_THRESHOLDS
    weight_pairs = weight_pairs or DEFAULT_WEIGHT_PAIRS

    best: TuneResult | None = None
    best_samples: list[_ScoredSample] = []
    evaluated: list[ThresholdCandidate] = []

    for wl, wf in weight_pairs:
        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=use_nli,
            ground_truth_store=ground_truth_store,
            w_logic=wl,
            w_fact=wf,
        )
        scores: list[_ScoredSample] = []
        for sample_index, s in enumerate(samples):
            _, cs = scorer.review(s["prompt"], s["response"])
            scores.append(
                _ScoredSample(
                    sample_index=sample_index,
                    score=cs.score,
                    label=bool(s["label"]),
                    prompt=str(s["prompt"]),
                    response=str(s["response"]),
                ),
            )

        for thr in thresholds:
            candidate = _evaluate_candidate(thr, wl, wf, scores)
            evaluated.append(candidate)

            result = TuneResult(
                threshold=thr,
                w_logic=wl,
                w_fact=wf,
                balanced_accuracy=candidate.balanced_accuracy,
                precision=candidate.precision,
                recall=candidate.recall,
                f1=candidate.f1,
                samples=len(samples),
                tp=candidate.tp,
                fp=candidate.fp,
                tn=candidate.tn,
                fn=candidate.fn,
            )
            if best is None or candidate.balanced_accuracy > best.balanced_accuracy:
                best = result
                best_samples = scores

    if best is None:
        raise ValueError("No valid tuning result — check that samples is non-empty")

    positives = _sum_int([1 if s.label else 0 for s in best_samples])
    negatives = len(best_samples) - positives
    ranked_candidates = tuple(_rank_candidates(evaluated))
    selection_margin = _selection_margin(best, ranked_candidates)
    confidence_level = _confidence_level(
        samples=len(samples),
        positives=positives,
        negatives=negatives,
        selection_margin=selection_margin,
    )
    return replace(
        best,
        positive_samples=positives,
        negative_samples=negatives,
        selection_margin=selection_margin,
        confidence_level=confidence_level,
        confidence_intervals=_confidence_intervals(best),
        tradeoff_summary=_tradeoff_summary(best),
        evaluated_candidates=ranked_candidates,
        boundary_examples=tuple(_boundary_examples(best.threshold, best_samples)),
    )


def _evaluate_candidate(
    threshold: float,
    w_logic: float,
    w_fact: float,
    scores: list[_ScoredSample],
) -> ThresholdCandidate:
    """Evaluate one threshold and weight pair against scored samples."""
    tp = fp = tn = fn = 0
    for scored in scores:
        predicted = scored.score >= threshold
        if scored.label and predicted:
            tp += 1
        elif scored.label and not predicted:
            fn += 1
        elif not scored.label and predicted:
            fp += 1
        else:
            tn += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return ThresholdCandidate(
        threshold=threshold,
        w_logic=w_logic,
        w_fact=w_fact,
        balanced_accuracy=round((tpr + tnr) / 2.0, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_positive_rate=round(false_positive_rate, 4),
        false_negative_rate=round(false_negative_rate, 4),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def _rank_candidates(
    candidates: list[ThresholdCandidate],
) -> list[ThresholdCandidate]:
    """Return candidates ordered by quality and tie-breaker stability."""
    return sorted(
        candidates,
        key=lambda c: (
            c.balanced_accuracy,
            c.f1,
            -abs(c.false_positive_rate - c.false_negative_rate),
            -c.threshold,
        ),
        reverse=True,
    )


def _selection_margin(
    best: TuneResult,
    ranked_candidates: tuple[ThresholdCandidate, ...],
) -> float:
    """Return the balanced-accuracy margin over the next-best candidate."""
    alternatives = [
        c
        for c in ranked_candidates
        if not (
            c.threshold == best.threshold
            and c.w_logic == best.w_logic
            and c.w_fact == best.w_fact
        )
    ]
    if not alternatives:
        return best.balanced_accuracy
    return round(best.balanced_accuracy - alternatives[0].balanced_accuracy, 4)


def _confidence_level(
    *,
    samples: int,
    positives: int,
    negatives: int,
    selection_margin: float,
) -> str:
    """Return a coarse confidence level for the selected threshold."""
    minority_ratio = min(positives, negatives) / samples if samples else 0.0
    if samples >= 50 and minority_ratio >= 0.2 and selection_margin >= 0.05:
        return "high"
    if samples >= 20 and minority_ratio >= 0.15 and selection_margin >= 0.02:
        return "medium"
    return "low"


def _confidence_intervals(result: TuneResult) -> dict[str, tuple[float, float]]:
    """Approximate 95% intervals for selected classification metrics.

    Recall and specificity are binomial rates, so Wilson score intervals are a
    deterministic small-sample choice. Balanced accuracy averages the recall and
    specificity bounds. F1 uses the monotonic transform of precision and recall
    bounds, giving an interpretable conservative interval without stochastic
    bootstrap variance in CLI output.
    """
    recall_ci = _wilson_interval(result.tp, result.tp + result.fn)
    specificity_ci = _wilson_interval(result.tn, result.tn + result.fp)
    precision_ci = _wilson_interval(result.tp, result.tp + result.fp)
    f1_ci = (
        _f1_from_precision_recall(precision_ci[0], recall_ci[0]),
        _f1_from_precision_recall(precision_ci[1], recall_ci[1]),
    )
    return {
        "balanced_accuracy": (
            round((recall_ci[0] + specificity_ci[0]) / 2.0, 4),
            round((recall_ci[1] + specificity_ci[1]) / 2.0, 4),
        ),
        "precision": precision_ci,
        "recall": recall_ci,
        "f1": (round(f1_ci[0], 4), round(f1_ci[1], 4)),
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the deterministic 95% Wilson score interval."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95% normal quantile
    p = successes / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * ((p * (1.0 - p) + z * z / (4.0 * total)) / total) ** 0.5
    return (
        round(max(0.0, (centre - margin) / denom), 4),
        round(min(1.0, (centre + margin) / denom), 4),
    )


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    """Return F1 from precision and recall."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _format_confidence_intervals(
    intervals: dict[str, tuple[float, float]],
) -> str:
    """Render confidence intervals as deterministic metadata text."""
    if not intervals:
        return ""
    return ", ".join(
        f"{name}=[{lo:.4f},{hi:.4f}]" for name, (lo, hi) in intervals.items()
    )


def _tradeoff_summary(result: TuneResult) -> str:
    """Describe the selected threshold's FP/FN trade-off."""
    false_positive_rate = (
        result.fp / (result.fp + result.tn) if (result.fp + result.tn) > 0 else 0.0
    )
    false_negative_rate = (
        result.fn / (result.fn + result.tp) if (result.fn + result.tp) > 0 else 0.0
    )
    if false_positive_rate > false_negative_rate + 0.05:
        return "selected threshold favours catching misses but may reject more clean responses"
    if false_negative_rate > false_positive_rate + 0.05:
        return "selected threshold favours fewer clean-response rejections but may miss more bad responses"
    return (
        "selected threshold balances clean-response rejections and missed bad responses"
    )


def _boundary_examples(
    threshold: float,
    scores: list[_ScoredSample],
) -> list[BoundaryExample]:
    """Return samples closest to the selected threshold boundary."""
    examples = []
    for scored in sorted(scores, key=lambda s: abs(s.score - threshold)):
        margin = round(scored.score - threshold, 4)
        examples.append(
            BoundaryExample(
                sample_index=scored.sample_index,
                score=round(scored.score, 4),
                label=scored.label,
                predicted=scored.score >= threshold,
                margin=margin,
                counterfactual_threshold=round(scored.score, 4),
                prompt_excerpt=_excerpt(scored.prompt),
                response_excerpt=_excerpt(scored.response),
            )
        )
        if len(examples) >= _REPORT_EXAMPLE_LIMIT:
            break
    return examples


def _excerpt(text: str, limit: int = 80) -> str:
    """Return a single-line excerpt bounded by character count."""
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def _format_boundary_example(example: BoundaryExample) -> str:
    """Render one boundary example for reports and YAML comments."""
    verdict = "approved" if example.predicted else "rejected"
    expected = "approved" if example.label else "rejected"
    return (
        f"sample {example.sample_index}: score={example.score:.4f}, "
        f"margin={example.margin:.4f}, predicted={verdict}, expected={expected}, "
        f"flip_threshold={example.counterfactual_threshold:.4f}, "
        f"prompt={example.prompt_excerpt!r}, response={example.response_excerpt!r}"
    )


def _yaml_scalar(value: object) -> str:
    """Render a primitive value as a conservative YAML scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _sum_int(values: list[int]) -> int:
    if _RUST_TUNER:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return int(rust_sum_i64(values))
    return sum(values)
