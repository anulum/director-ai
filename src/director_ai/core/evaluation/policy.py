# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Policy Evaluation Harness

"""Controlled comparison of profiles, thresholds, and scorer policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..mandatory import mandatory_execution

try:  # pragma: no cover - optional acceleration
    from backfire_kernel import rust_sum_i64

    _RUST_POLICY_EVAL = True
except ImportError:  # pragma: no cover - mandatory accelerator guard
    _RUST_POLICY_EVAL = True

    def rust_sum_i64(_values: list[int]) -> int:
        raise RuntimeError("backfire_kernel rust_sum_i64 is unavailable")


__all__ = [
    "LabelledPolicySample",
    "PolicyComparisonReport",
    "PolicyEvaluationReport",
    "PolicyVariant",
    "PolicyVariantResult",
    "compare_policy_variants",
    "evaluate_policy_variants",
]

ScoreFunction = Callable[["LabelledPolicySample", "PolicyVariant"], float]


@dataclass(frozen=True)
class LabelledPolicySample:
    """One labelled prompt/response sample used for policy evaluation."""

    prompt: str
    response: str
    label: bool
    score: float | None = None
    dataset: str = ""
    synthetic: bool = False
    benchmark_evidence: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if not self.response.strip():
            raise ValueError("response is required")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        object.__setattr__(
            self,
            "metadata",
            {str(key): str(value) for key, value in self.metadata.items()},
        )


@dataclass(frozen=True)
class PolicyVariant:
    """Profile or threshold policy evaluated against the same labelled samples."""

    name: str
    threshold: float
    profile: str = ""
    w_logic: float = 0.6
    w_fact: float = 0.4
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("variant name is required")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0.0 <= self.w_logic <= 1.0:
            raise ValueError("w_logic must be in [0, 1]")
        if not 0.0 <= self.w_fact <= 1.0:
            raise ValueError("w_fact must be in [0, 1]")
        if abs(self.w_logic + self.w_fact - 1.0) > 1e-9:
            raise ValueError("w_logic + w_fact must equal 1.0")
        object.__setattr__(
            self,
            "metadata",
            {str(key): str(value) for key, value in self.metadata.items()},
        )


@dataclass(frozen=True)
class PolicyVariantResult:
    """Confusion-matrix metrics for one policy variant."""

    name: str
    threshold: float
    profile: str
    sample_count: int
    balanced_accuracy: float
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "profile": self.profile,
            "sample_count": self.sample_count,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


@dataclass(frozen=True)
class PolicyEvaluationReport:
    """Multi-variant policy evaluation report with provenance guardrails."""

    results: tuple[PolicyVariantResult, ...]
    sample_count: int
    datasets: tuple[str, ...]
    provenance_counts: Mapping[str, int]
    public_benchmark_eligible: bool
    public_claim_reason: str

    def variant(self, name: str) -> PolicyVariantResult:
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "datasets": list(self.datasets),
            "provenance_counts": dict(self.provenance_counts),
            "public_benchmark_eligible": self.public_benchmark_eligible,
            "public_claim_reason": self.public_claim_reason,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class PolicyComparisonReport:
    """Two-arm A/B comparison extracted from a policy evaluation report."""

    baseline: PolicyVariantResult
    candidate: PolicyVariantResult
    delta_balanced_accuracy: float
    delta_false_positive_rate: float
    delta_false_negative_rate: float
    winner: str
    evaluation: PolicyEvaluationReport


def evaluate_policy_variants(
    samples: Sequence[LabelledPolicySample],
    *,
    variants: Sequence[PolicyVariant],
    score_fn: ScoreFunction | None = None,
) -> PolicyEvaluationReport:
    """Evaluate all variants on the same labelled sample set."""
    sample_tuple = tuple(samples)
    variant_tuple = tuple(variants)
    if not sample_tuple:
        raise ValueError("samples must be non-empty")
    if not variant_tuple:
        raise ValueError("variants must be non-empty")
    _reject_duplicate_variants(variant_tuple)

    results = tuple(
        _evaluate_variant(sample_tuple, variant, score_fn) for variant in variant_tuple
    )
    provenance_counts = _provenance_counts(sample_tuple)
    eligible, reason = _public_claim_status(sample_tuple, provenance_counts)
    datasets = tuple(
        sorted({sample.dataset for sample in sample_tuple if sample.dataset})
    )
    return PolicyEvaluationReport(
        results=results,
        sample_count=len(sample_tuple),
        datasets=datasets,
        provenance_counts=provenance_counts,
        public_benchmark_eligible=eligible,
        public_claim_reason=reason,
    )


def compare_policy_variants(
    samples: Sequence[LabelledPolicySample],
    *,
    baseline: PolicyVariant,
    candidate: PolicyVariant,
    score_fn: ScoreFunction | None = None,
) -> PolicyComparisonReport:
    """Run a controlled two-arm policy comparison."""
    evaluation = evaluate_policy_variants(
        samples,
        variants=(baseline, candidate),
        score_fn=score_fn,
    )
    baseline_result = evaluation.variant(baseline.name)
    candidate_result = evaluation.variant(candidate.name)
    delta_ba = round(
        candidate_result.balanced_accuracy - baseline_result.balanced_accuracy,
        10,
    )
    delta_fpr = round(
        candidate_result.false_positive_rate - baseline_result.false_positive_rate,
        10,
    )
    delta_fnr = round(
        candidate_result.false_negative_rate - baseline_result.false_negative_rate,
        10,
    )
    if delta_ba > 0:
        winner = candidate.name
    elif delta_ba < 0:
        winner = baseline.name
    else:
        winner = "tie"
    return PolicyComparisonReport(
        baseline=baseline_result,
        candidate=candidate_result,
        delta_balanced_accuracy=delta_ba,
        delta_false_positive_rate=delta_fpr,
        delta_false_negative_rate=delta_fnr,
        winner=winner,
        evaluation=evaluation,
    )


def _evaluate_variant(
    samples: Sequence[LabelledPolicySample],
    variant: PolicyVariant,
    score_fn: ScoreFunction | None,
) -> PolicyVariantResult:
    tp = fp = tn = fn = 0
    for sample in samples:
        score = _score_sample(sample, variant, score_fn)
        predicted = score >= variant.threshold
        if predicted and sample.label:
            tp += 1
        elif predicted and not sample.label:
            fp += 1
        elif not predicted and sample.label:
            fn += 1
        else:
            tn += 1
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    return PolicyVariantResult(
        name=variant.name,
        threshold=variant.threshold,
        profile=variant.profile,
        sample_count=len(samples),
        balanced_accuracy=(tpr + tnr) / 2.0,
        precision=precision,
        recall=tpr,
        false_positive_rate=_safe_div(fp, fp + tn),
        false_negative_rate=_safe_div(fn, fn + tp),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def _score_sample(
    sample: LabelledPolicySample,
    variant: PolicyVariant,
    score_fn: ScoreFunction | None,
) -> float:
    if sample.score is not None:
        return sample.score
    if score_fn is None:
        raise ValueError("score_fn is required when samples do not carry scores")
    score = float(score_fn(sample, variant))
    if not 0.0 <= score <= 1.0:
        raise ValueError("score_fn must return scores in [0, 1]")
    return score


def _reject_duplicate_variants(variants: Sequence[PolicyVariant]) -> None:
    seen: set[str] = set()
    for variant in variants:
        if variant.name in seen:
            raise ValueError(f"duplicate policy variant name {variant.name!r}")
        seen.add(variant.name)


def _provenance_counts(samples: Sequence[LabelledPolicySample]) -> dict[str, int]:
    synthetic = _sum_int([1 if sample.synthetic else 0 for sample in samples])
    benchmark = _sum_int(
        [
            1 if sample.benchmark_evidence and not sample.synthetic else 0
            for sample in samples
        ]
    )
    internal = len(samples) - synthetic - benchmark
    return {"benchmark": benchmark, "internal": internal, "synthetic": synthetic}


def _public_claim_status(
    samples: Sequence[LabelledPolicySample],
    provenance_counts: Mapping[str, int],
) -> tuple[bool, str]:
    if provenance_counts["synthetic"]:
        return False, "synthetic samples are present; report is internal-only"
    if provenance_counts["internal"]:
        return False, "internal non-benchmark samples are present"
    datasets = {sample.dataset for sample in samples if sample.dataset}
    if len(datasets) != 1:
        return False, "public benchmark claims require exactly one named dataset"
    return True, "all samples are real benchmark evidence from one dataset"


def _safe_div(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _sum_int(values: list[int]) -> int:
    if _RUST_POLICY_EVAL:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return int(rust_sum_i64(values))
    return sum(values)
