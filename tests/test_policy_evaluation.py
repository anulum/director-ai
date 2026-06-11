# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Policy Evaluation Harness Tests
"""Tests for profile and threshold policy comparison on labelled data."""

from __future__ import annotations

import pytest

from director_ai.core.evaluation import policy as policy_module
from director_ai.core.evaluation.policy import (
    LabelledPolicySample,
    PolicyVariant,
    compare_policy_variants,
    evaluate_policy_variants,
)


def _samples() -> list[LabelledPolicySample]:
    return [
        LabelledPolicySample(
            prompt="q1",
            response="supported",
            label=True,
            score=0.91,
            dataset="regression",
            benchmark_evidence=True,
        ),
        LabelledPolicySample(
            prompt="q2",
            response="borderline supported",
            label=True,
            score=0.58,
            dataset="regression",
            benchmark_evidence=True,
        ),
        LabelledPolicySample(
            prompt="q3",
            response="unsupported",
            label=False,
            score=0.42,
            dataset="regression",
            benchmark_evidence=True,
        ),
        LabelledPolicySample(
            prompt="q4",
            response="bad",
            label=False,
            score=0.12,
            dataset="regression",
            benchmark_evidence=True,
        ),
    ]


def test_policy_evaluation_compares_threshold_variants_on_same_samples():
    report = evaluate_policy_variants(
        _samples(),
        variants=[
            PolicyVariant(name="balanced", threshold=0.5),
            PolicyVariant(name="strict", threshold=0.7),
        ],
    )

    balanced = report.variant("balanced")
    strict = report.variant("strict")

    assert report.sample_count == 4
    assert report.public_benchmark_eligible is True
    assert balanced.balanced_accuracy == 1.0
    assert (balanced.tp, balanced.fp, balanced.tn, balanced.fn) == (2, 0, 2, 0)
    assert strict.false_negative_rate == 0.5
    assert strict.balanced_accuracy == 0.75


def test_policy_comparison_reports_delta_and_winner():
    report = compare_policy_variants(
        _samples(),
        baseline=PolicyVariant(name="baseline", threshold=0.7),
        candidate=PolicyVariant(name="candidate", threshold=0.5),
    )

    assert report.baseline.name == "baseline"
    assert report.candidate.name == "candidate"
    assert report.delta_balanced_accuracy == 0.25
    assert report.winner == "candidate"


def test_policy_evaluation_flags_mixed_or_synthetic_evidence_as_internal_only():
    samples = [
        *_samples(),
        LabelledPolicySample(
            prompt="synthetic prompt",
            response="synthetic response",
            label=False,
            score=0.2,
            dataset="synthetic",
            synthetic=True,
            benchmark_evidence=False,
        ),
    ]

    report = evaluate_policy_variants(
        samples,
        variants=[PolicyVariant(name="balanced", threshold=0.5)],
    )

    assert report.public_benchmark_eligible is False
    assert report.provenance_counts == {
        "benchmark": 4,
        "internal": 0,
        "synthetic": 1,
    }
    assert "synthetic" in report.public_claim_reason


def test_policy_evaluation_uses_score_function_when_scores_are_not_cached():
    calls = []

    def score_fn(sample: LabelledPolicySample, variant: PolicyVariant) -> float:
        calls.append((sample.prompt, variant.name))
        return 0.9 if sample.label else 0.1

    samples = [
        LabelledPolicySample("q1", "a1", True, dataset="internal"),
        LabelledPolicySample("q2", "a2", False, dataset="internal"),
    ]
    report = evaluate_policy_variants(
        samples,
        variants=[PolicyVariant(name="profile-fast", threshold=0.5, profile="fast")],
        score_fn=score_fn,
    )

    assert calls == [("q1", "profile-fast"), ("q2", "profile-fast")]
    assert report.variant("profile-fast").balanced_accuracy == 1.0


def test_policy_evaluation_rejects_ambiguous_inputs():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_policy_variants(
            _samples(),
            variants=[
                PolicyVariant(name="same", threshold=0.5),
                PolicyVariant(name="same", threshold=0.6),
            ],
        )

    with pytest.raises(ValueError, match="score_fn"):
        evaluate_policy_variants(
            [LabelledPolicySample("q", "a", True)],
            variants=[PolicyVariant(name="v", threshold=0.5)],
        )


def test_policy_evaluation_public_exports():
    from director_ai import PolicyVariant as TopLevelPolicyVariant
    from director_ai import compare_policy_variants as top_level_compare
    from director_ai.core import PolicyEvaluationReport

    assert TopLevelPolicyVariant is PolicyVariant
    assert top_level_compare is compare_policy_variants
    assert PolicyEvaluationReport.__name__ == "PolicyEvaluationReport"


def test_labelled_samples_validate_required_fields_and_score_range():
    with pytest.raises(ValueError, match="prompt is required"):
        LabelledPolicySample("", "response", True)

    with pytest.raises(ValueError, match="response is required"):
        LabelledPolicySample("prompt", " ", True)

    with pytest.raises(ValueError, match="score must be in"):
        LabelledPolicySample("prompt", "response", True, score=1.1)

    sample = LabelledPolicySample(
        "prompt",
        "response",
        True,
        score=0.5,
        metadata={"numeric": 7},
    )
    assert sample.metadata == {"numeric": "7"}


def test_policy_variants_validate_threshold_weights_and_metadata():
    invalid_variants = [
        ({"name": " ", "threshold": 0.5}, "variant name"),
        ({"name": "bad-threshold", "threshold": 1.1}, "threshold"),
        ({"name": "bad-logic", "threshold": 0.5, "w_logic": -0.1}, "w_logic"),
        ({"name": "bad-fact", "threshold": 0.5, "w_fact": 1.1}, "w_fact"),
        (
            {"name": "bad-sum", "threshold": 0.5, "w_logic": 0.2, "w_fact": 0.2},
            "must equal 1.0",
        ),
    ]

    for kwargs, expected in invalid_variants:
        with pytest.raises(ValueError, match=expected):
            PolicyVariant(**kwargs)

    variant = PolicyVariant(
        name="stringify",
        threshold=0.5,
        metadata={"numeric": 7},
    )
    assert variant.metadata == {"numeric": "7"}


def test_report_exports_and_missing_variant_lookup():
    report = evaluate_policy_variants(
        _samples(),
        variants=[PolicyVariant(name="balanced", threshold=0.5)],
    )

    exported = report.to_dict()

    assert exported["results"][0]["name"] == "balanced"
    assert exported["results"][0]["tp"] == 2
    with pytest.raises(KeyError, match="missing"):
        report.variant("missing")


def test_policy_evaluation_rejects_empty_samples_and_empty_variants():
    with pytest.raises(ValueError, match="samples must be non-empty"):
        evaluate_policy_variants([], variants=[PolicyVariant(name="v", threshold=0.5)])

    with pytest.raises(ValueError, match="variants must be non-empty"):
        evaluate_policy_variants(_samples(), variants=[])


def test_policy_comparison_reports_baseline_winner_and_tie():
    baseline_wins = compare_policy_variants(
        _samples(),
        baseline=PolicyVariant(name="baseline", threshold=0.5),
        candidate=PolicyVariant(name="candidate", threshold=0.7),
    )
    tied = compare_policy_variants(
        _samples(),
        baseline=PolicyVariant(name="baseline", threshold=0.5),
        candidate=PolicyVariant(name="candidate", threshold=0.5),
    )

    assert baseline_wins.winner == "baseline"
    assert tied.winner == "tie"


def test_policy_evaluation_counts_false_positives_and_rejects_bad_score_fn():
    samples = [
        LabelledPolicySample("q1", "a1", True, score=0.2),
        LabelledPolicySample("q2", "a2", False, score=0.9),
    ]
    report = evaluate_policy_variants(
        samples,
        variants=[PolicyVariant(name="too-open", threshold=0.5)],
    )

    assert report.variant("too-open").fp == 1
    assert report.variant("too-open").precision == 0.0

    with pytest.raises(ValueError, match="scores in"):
        evaluate_policy_variants(
            [LabelledPolicySample("q", "a", True)],
            variants=[PolicyVariant(name="bad-score", threshold=0.5)],
            score_fn=lambda _sample, _variant: 2.0,
        )


def test_policy_evaluation_rejects_public_claims_for_mixed_benchmark_datasets():
    samples = [
        LabelledPolicySample(
            "q1",
            "a1",
            True,
            score=0.8,
            dataset="benchmark-a",
            benchmark_evidence=True,
        ),
        LabelledPolicySample(
            "q2",
            "a2",
            False,
            score=0.2,
            dataset="benchmark-b",
            benchmark_evidence=True,
        ),
    ]

    report = evaluate_policy_variants(
        samples,
        variants=[PolicyVariant(name="balanced", threshold=0.5)],
    )

    assert report.public_benchmark_eligible is False
    assert "exactly one named dataset" in report.public_claim_reason


def test_policy_evaluation_uses_python_sum_fallback_when_accelerator_disabled(
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "_RUST_POLICY_EVAL", False)
    samples = _samples()

    report = evaluate_policy_variants(
        samples,
        variants=[PolicyVariant(name="balanced", threshold=0.5)],
    )

    assert report.provenance_counts == {
        "benchmark": 4,
        "internal": 0,
        "synthetic": 0,
    }
