# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Build Judge Dataset Tests
"""Multi-angle tests for training/build_judge_dataset.py.

Covers: label remapping, stratified subsampling, borderline filtering,
input formatting, multi-GPU shard logic, CLI argument handling,
edge cases, and pipeline performance documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.build_judge_dataset import (  # noqa: E402
    BORDERLINE_HIGH,
    BORDERLINE_LOW,
    LABEL_APPROVE,
    LABEL_REJECT,
    filter_and_balance,
    format_judge_input,
    remap_labels,
    stratified_subsample,
)

# ── Label remapping ─────────────────────────────────────────────────


class TestRemapLabels:
    """Test 3-class → binary label remapping."""

    def _make_dataset(self, labels):
        return datasets.Dataset.from_dict(
            {
                "premise": ["P"] * len(labels),
                "hypothesis": ["H"] * len(labels),
                "label": labels,
            }
        )

    def test_entailment_maps_to_approve(self):
        ds = self._make_dataset([0])
        result = remap_labels(ds)
        assert result[0]["label"] == LABEL_APPROVE

    @pytest.mark.parametrize("label", [1, 2])
    def test_neutral_and_contradiction_map_to_reject(self, label):
        ds = self._make_dataset([label])
        result = remap_labels(ds)
        assert result[0]["label"] == LABEL_REJECT

    def test_mixed_labels_correctly_remapped(self):
        ds = self._make_dataset([0, 1, 2, 0, 2])
        result = remap_labels(ds)
        expected = [
            LABEL_APPROVE,
            LABEL_REJECT,
            LABEL_REJECT,
            LABEL_APPROVE,
            LABEL_REJECT,
        ]
        assert [r["label"] for r in result] == expected

    def test_preserves_other_columns(self):
        ds = self._make_dataset([0, 1])
        result = remap_labels(ds)
        assert result[0]["premise"] == "P"
        assert result[1]["hypothesis"] == "H"


# ── Stratified subsampling ──────────────────────────────────────────


class TestStratifiedSubsample:
    """Test stratified subsampling maintains label balance."""

    def _make_binary_dataset(self, n_approve, n_reject):
        labels = [LABEL_APPROVE] * n_approve + [LABEL_REJECT] * n_reject
        return datasets.Dataset.from_dict(
            {
                "premise": [f"P{i}" for i in range(len(labels))],
                "hypothesis": [f"H{i}" for i in range(len(labels))],
                "label": labels,
            }
        )

    def test_subsample_reduces_size(self):
        ds = self._make_binary_dataset(500, 500)
        sub = stratified_subsample(ds, 200)
        assert len(sub) == 200

    def test_maintains_approximate_balance(self):
        ds = self._make_binary_dataset(500, 500)
        sub = stratified_subsample(ds, 200)
        labels = np.array(sub["label"])
        approve_ratio = (labels == LABEL_APPROVE).mean()
        assert 0.4 <= approve_ratio <= 0.6

    def test_subsample_larger_than_dataset_uses_all(self):
        ds = self._make_binary_dataset(10, 10)
        sub = stratified_subsample(ds, 1000)
        assert len(sub) <= 20

    def test_deterministic_with_same_seed(self):
        ds = self._make_binary_dataset(100, 100)
        sub1 = stratified_subsample(ds, 50, seed=42)
        sub2 = stratified_subsample(ds, 50, seed=42)
        assert sub1["premise"] == sub2["premise"]

    def test_different_seeds_differ(self):
        ds = self._make_binary_dataset(100, 100)
        sub1 = stratified_subsample(ds, 50, seed=42)
        sub2 = stratified_subsample(ds, 50, seed=99)
        assert sub1["premise"] != sub2["premise"]


# ── Borderline filtering ───────────────────────────────────────────


class TestFilterAndBalance:
    """Test borderline zone filtering and confident sample selection."""

    def _make_scored_dataset(self, divergences, labels=None):
        n = len(divergences)
        if labels is None:
            labels = [LABEL_APPROVE] * (n // 2) + [LABEL_REJECT] * (n - n // 2)
        return datasets.Dataset.from_dict(
            {
                "premise": [f"P{i}" for i in range(n)],
                "hypothesis": [f"H{i}" for i in range(n)],
                "label": labels[:n],
                "nli_divergence": divergences,
            }
        )

    def test_borderline_zone_boundaries(self):
        assert BORDERLINE_LOW == 0.2
        assert BORDERLINE_HIGH == 0.8

    def test_borderline_samples_kept(self):
        divs = [0.1, 0.3, 0.5, 0.7, 0.9]
        ds = self._make_scored_dataset(divs)
        result = filter_and_balance(ds, borderline_keep=100, confident_keep=100)
        result_divs = result["nli_divergence"]
        borderline = [d for d in result_divs if BORDERLINE_LOW <= d <= BORDERLINE_HIGH]
        assert len(borderline) == 3  # 0.3, 0.5, 0.7

    def test_confident_samples_included(self):
        divs = [0.05, 0.1, 0.5, 0.85, 0.95]
        ds = self._make_scored_dataset(divs)
        result = filter_and_balance(ds, borderline_keep=100, confident_keep=100)
        assert len(result) == 5  # all kept when limits > count

    def test_borderline_keep_zero_keeps_all(self):
        divs = [0.3, 0.4, 0.5, 0.6, 0.7]
        ds = self._make_scored_dataset(divs)
        result = filter_and_balance(ds, borderline_keep=0, confident_keep=0)
        assert len(result) == 5

    def test_borderline_keep_limits_count(self):
        divs = [0.3, 0.4, 0.5, 0.6, 0.7]
        ds = self._make_scored_dataset(divs)
        result = filter_and_balance(ds, borderline_keep=2, confident_keep=0)
        borderline = [
            d
            for d in result["nli_divergence"]
            if BORDERLINE_LOW <= d <= BORDERLINE_HIGH
        ]
        assert len(borderline) == 2

    def test_confident_keep_limits_count(self):
        divs = [0.05, 0.1, 0.15, 0.5]
        ds = self._make_scored_dataset(divs)
        result = filter_and_balance(ds, borderline_keep=100, confident_keep=1)
        confident = [
            d
            for d in result["nli_divergence"]
            if d < BORDERLINE_LOW or d > BORDERLINE_HIGH
        ]
        assert len(confident) == 1

    @pytest.mark.parametrize("n_borderline", [0, 1, 10, 100])
    def test_various_borderline_counts(self, n_borderline):
        divs = ([0.5] * n_borderline) + ([0.05] * 5)
        ds = self._make_scored_dataset(divs)
        result = filter_and_balance(ds, borderline_keep=0, confident_keep=0)
        assert len(result) == n_borderline + 5


# ── Input formatting ───────────────────────────────────────────────


class TestFormatJudgeInput:
    """Test judge input text formatting."""

    def test_format_includes_divergence(self):
        example = {
            "premise": "Context",
            "hypothesis": "Response",
            "nli_divergence": 0.45,
        }
        result = format_judge_input(dict(example))
        assert "NLI divergence: 0.45" in result["text"]

    def test_format_includes_context(self):
        example = {
            "premise": "The sky is blue",
            "hypothesis": "Response",
            "nli_divergence": 0.5,
        }
        result = format_judge_input(dict(example))
        assert "Context: The sky is blue" in result["text"]

    def test_format_includes_response(self):
        example = {
            "premise": "P",
            "hypothesis": "The answer is 42",
            "nli_divergence": 0.5,
        }
        result = format_judge_input(dict(example))
        assert "Response: The answer is 42" in result["text"]

    def test_premise_truncated_at_400(self):
        example = {"premise": "X" * 1000, "hypothesis": "H", "nli_divergence": 0.5}
        result = format_judge_input(dict(example))
        context_line = [
            ln for ln in result["text"].split("\n") if ln.startswith("Context:")
        ][0]
        assert len(context_line) <= 410  # "Context: " + 400 chars

    def test_hypothesis_truncated_at_400(self):
        example = {"premise": "P", "hypothesis": "Y" * 1000, "nli_divergence": 0.5}
        result = format_judge_input(dict(example))
        response_line = [
            ln for ln in result["text"].split("\n") if ln.startswith("Response:")
        ][0]
        assert len(response_line) <= 410

    @pytest.mark.parametrize("divergence", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_divergence_formatting_two_decimals(self, divergence):
        example = {"premise": "P", "hypothesis": "H", "nli_divergence": divergence}
        result = format_judge_input(dict(example))
        assert f"NLI divergence: {divergence:.2f}" in result["text"]


# ── Pipeline performance documentation ──────────────────────────────


class TestPipelinePerformance:
    """Verify build pipeline documents performance metrics."""

    def test_borderline_zone_is_documented(self):
        """Borderline zone constants must match documented values."""
        assert BORDERLINE_LOW == 0.2
        assert BORDERLINE_HIGH == 0.8

    def test_label_constants_documented(self):
        """Binary label constants must be 0 (approve) and 1 (reject)."""
        assert LABEL_APPROVE == 0
        assert LABEL_REJECT == 1

    def test_format_produces_three_line_input(self):
        """Judge input must have exactly 3 lines: divergence, context, response."""
        example = {"premise": "P", "hypothesis": "H", "nli_divergence": 0.5}
        result = format_judge_input(dict(example))
        lines = result["text"].strip().split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("NLI divergence:")
        assert lines[1].startswith("Context:")
        assert lines[2].startswith("Response:")
