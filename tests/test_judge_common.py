# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Tests for benchmarks/_judge_common.py
"""Multi-angle tests for the shared AggreFact judge utilities.

Covers:

* ``parse_response`` — every documented input class (SUPPORTED,
  NOT_SUPPORTED, NOT SUPPORTED, NOT-SUPPORTED, Yes/No, True/False,
  mixed case, prefix match, gibberish, empty, whitespace, embedded
  substring edge cases)
* ``compute_balanced_accuracy`` — all-correct, all-wrong, half-right,
  empty, single-class (positive-only, negative-only), unknown
  filtering (-1), all-unknown, mismatched lengths
* ``DATASET_TO_FAMILY`` — all 11 AggreFact datasets mapped, no
  unmapped datasets, exactly 3 families
* ``PROMPTS`` — every family has a non-empty prompt, prompts contain
  ``{premise}`` and ``{hypothesis}`` placeholders
* ``AGGREFACT_DATASETS`` — length 11, sorted, matches
  DATASET_TO_FAMILY keys
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from _judge_common import (  # noqa: E402
    AGGREFACT_DATASETS,
    DATASET_TO_FAMILY,
    PROMPTS,
    compute_balanced_accuracy,
    parse_response,
)

# ── parse_response ───────────────────────────────────────────────────────


class TestParseResponse:
    """Multi-angle tests for LLM verdict parsing."""

    # ── Positive verdicts → 1 ────────────────────────────────────────

    def test_supported_uppercase(self):
        assert parse_response("SUPPORTED") == 1

    def test_supported_lowercase(self):
        assert parse_response("supported") == 1

    def test_supported_mixed_case(self):
        assert parse_response("Supported") == 1

    def test_supported_with_whitespace(self):
        assert parse_response("  SUPPORTED  \n") == 1

    def test_supported_embedded_in_sentence(self):
        assert parse_response("The claim is SUPPORTED by the context.") == 1

    def test_yes_prefix(self):
        assert parse_response("Yes") == 1

    def test_yes_with_explanation(self):
        assert parse_response("Yes, the claim is fully grounded.") == 1

    def test_true_prefix(self):
        assert parse_response("True") == 1

    def test_true_uppercase(self):
        assert parse_response("TRUE") == 1

    # ── Negative verdicts → 0 ────────────────────────────────────────

    def test_not_supported_underscore(self):
        assert parse_response("NOT_SUPPORTED") == 0

    def test_not_supported_space(self):
        assert parse_response("NOT SUPPORTED") == 0

    def test_not_supported_hyphen(self):
        assert parse_response("NOT-SUPPORTED") == 0

    def test_not_supported_lowercase(self):
        assert parse_response("not_supported") == 0

    def test_not_supported_mixed_case(self):
        assert parse_response("Not Supported") == 0

    def test_not_supported_in_sentence(self):
        assert parse_response("The claim is NOT_SUPPORTED.") == 0

    def test_no_prefix(self):
        assert parse_response("No") == 0

    def test_no_with_explanation(self):
        assert parse_response("No, the claim contradicts the source.") == 0

    def test_false_prefix(self):
        assert parse_response("False") == 0

    def test_false_uppercase(self):
        assert parse_response("FALSE") == 0

    # ── NOT_SUPPORTED takes priority over SUPPORTED ──────────────────

    def test_not_supported_contains_supported_substring(self):
        """NOT_SUPPORTED contains 'SUPPORTED' as substring. The NOT
        variant must match first."""
        assert parse_response("NOT_SUPPORTED") == 0

    def test_not_supported_space_contains_supported(self):
        assert parse_response("NOT SUPPORTED by the evidence") == 0

    # ── Unknown → -1 ─────────────────────────────────────────────────

    def test_empty_string(self):
        assert parse_response("") == -1

    def test_whitespace_only(self):
        assert parse_response("   \n\t  ") == -1

    def test_gibberish(self):
        assert parse_response("Lorem ipsum dolor sit amet") == -1

    def test_error_string(self):
        assert parse_response("ERROR") == -1

    def test_maybe(self):
        assert parse_response("Maybe") == -1

    def test_partially(self):
        assert parse_response("Partially supported") == 1  # contains "SUPPORTED"

    def test_number(self):
        assert parse_response("42") == -1


# ── compute_balanced_accuracy ────────────────────────────────────────────


class TestComputeBalancedAccuracy:
    """Multi-angle tests for two-class balanced accuracy."""

    def test_all_correct(self):
        assert compute_balanced_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_all_wrong(self):
        assert compute_balanced_accuracy([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0

    def test_half_right(self):
        # tp=1, fn=1, tn=1, fp=1 → recalls 0.5 / 0.5 → BA 0.5
        assert compute_balanced_accuracy([1, 0, 0, 1], [1, 0, 1, 0]) == 0.5

    def test_empty_returns_zero(self):
        assert compute_balanced_accuracy([], []) == 0.0

    def test_unknowns_filtered(self):
        # -1 predictions dropped; remaining two are perfect.
        assert compute_balanced_accuracy([1, -1, -1, 0], [1, 0, 1, 0]) == 1.0

    def test_all_unknown_returns_zero(self):
        assert compute_balanced_accuracy([-1, -1], [1, 0]) == 0.0

    def test_only_positive_class(self):
        # No negatives → pos==0 check returns 0.0
        assert compute_balanced_accuracy([1, 1], [1, 1]) == 0.0

    def test_only_negative_class(self):
        assert compute_balanced_accuracy([0, 0], [0, 0]) == 0.0

    def test_unbalanced_dataset(self):
        # 3 positives (2 correct), 1 negative (1 correct)
        # recall_pos = 2/3, recall_neg = 1/1 → BA = (2/3 + 1)/2 ≈ 0.8333
        ba = compute_balanced_accuracy([1, 1, 0, 0], [1, 1, 1, 0])
        assert abs(ba - 5 / 6) < 1e-9

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_balanced_accuracy([1, 0], [1])

    def test_large_realistic(self):
        """1000-sample synthetic dataset with 60% positive label and
        a correlated predictor that flips 20% of the time."""
        import random

        rng = random.Random(42)
        labels = [1] * 600 + [0] * 400
        # Correlated predictor: correct 80% of the time, flip 20%
        preds = [lab if rng.random() > 0.2 else (1 - lab) for lab in labels]
        ba = compute_balanced_accuracy(preds, labels)
        # 80% recall on both classes → BA ≈ 0.80 ± sampling noise
        assert 0.75 < ba < 0.85

    def test_symmetry(self):
        """BA should give equal weight to both classes regardless of
        class imbalance."""
        # 1 positive (correct), 99 negatives (all correct)
        # recall_pos = 1/1 = 1.0, recall_neg = 99/99 = 1.0 → BA = 1.0
        preds = [1] + [0] * 99
        labels = [1] + [0] * 99
        assert compute_balanced_accuracy(preds, labels) == 1.0

    def test_single_sample_per_class(self):
        assert compute_balanced_accuracy([1, 0], [1, 0]) == 1.0
        assert compute_balanced_accuracy([0, 1], [1, 0]) == 0.0


# ── DATASET_TO_FAMILY ────────────────────────────────────────────────────


class TestDatasetToFamily:
    """Completeness and consistency checks on the dataset mapping."""

    def test_exactly_11_datasets(self):
        assert len(DATASET_TO_FAMILY) == 11

    def test_exactly_3_families(self):
        families = set(DATASET_TO_FAMILY.values())
        assert families == {"summ", "rag", "claim"}

    def test_summ_family_datasets(self):
        summ = {k for k, v in DATASET_TO_FAMILY.items() if v == "summ"}
        assert summ == {
            "AggreFact-CNN",
            "AggreFact-XSum",
            "TofuEval-MediaS",
            "TofuEval-MeetB",
        }

    def test_rag_family_datasets(self):
        rag = {k for k, v in DATASET_TO_FAMILY.items() if v == "rag"}
        assert rag == {"RAGTruth", "ClaimVerify", "FactCheck-GPT", "ExpertQA"}

    def test_claim_family_datasets(self):
        claim = {k for k, v in DATASET_TO_FAMILY.items() if v == "claim"}
        assert claim == {"Reveal", "Lfqa", "Wice"}

    def test_all_values_are_strings(self):
        for k, v in DATASET_TO_FAMILY.items():
            assert isinstance(k, str), f"key {k!r} is not str"
            assert isinstance(v, str), f"value {v!r} for {k!r} is not str"


# ── PROMPTS ──────────────────────────────────────────────────────────────


class TestPrompts:
    """Sanity checks on the three routed prompt templates."""

    def test_exactly_3_prompts(self):
        assert set(PROMPTS) == {"summ", "rag", "claim"}

    @pytest.mark.parametrize("family", ["summ", "rag", "claim"])
    def test_prompt_has_premise_placeholder(self, family):
        assert "{premise}" in PROMPTS[family]

    @pytest.mark.parametrize("family", ["summ", "rag", "claim"])
    def test_prompt_has_hypothesis_placeholder(self, family):
        assert "{hypothesis}" in PROMPTS[family]

    @pytest.mark.parametrize("family", ["summ", "rag", "claim"])
    def test_prompt_non_empty(self, family):
        assert len(PROMPTS[family]) > 50

    @pytest.mark.parametrize("family", ["summ", "rag", "claim"])
    def test_prompt_ends_with_instruction(self, family):
        assert PROMPTS[family].rstrip().endswith("NOT_SUPPORTED.")

    @pytest.mark.parametrize("family", ["summ", "rag", "claim"])
    def test_prompt_formats_without_error(self, family):
        """Formatting with real-ish values should not raise."""
        result = PROMPTS[family].format(
            premise="The sky is blue.", hypothesis="The sky is blue."
        )
        assert "The sky is blue." in result


# ── AGGREFACT_DATASETS ───────────────────────────────────────────────────


class TestAggrefactDatasets:
    def test_length_11(self):
        assert len(AGGREFACT_DATASETS) == 11

    def test_sorted(self):
        assert tuple(sorted(AGGREFACT_DATASETS)) == AGGREFACT_DATASETS

    def test_matches_dataset_to_family_keys(self):
        assert set(AGGREFACT_DATASETS) == set(DATASET_TO_FAMILY)
