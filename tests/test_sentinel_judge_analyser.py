# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``benchmarks.sentinel_judge_analyser``.

Covers:

* ``load_judge()`` — schema normalisation, v1 legacy scores, length mismatch.
* ``align_judges()`` — matching judges, label mismatch, dataset mismatch.
* ``balanced_accuracy()`` — edge cases (empty, all-unknown, single-class).
* ``per_dataset_ba()`` — correct grouping and per-dataset metrics.
* ``voting_ensemble()`` — majority vote, ties, all-abstain.
* ``routed_ensemble()`` — per-dataset routing with 50/50 split.
* ``lr_fusion_ensemble()`` — 5-fold CV on synthetic data, output length.
* ``oracle_upper_bound()`` — ceiling computation.
* ``main()`` — full CLI with synthetic JSON files, output schema verification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from sentinel_judge_analyser import (  # noqa: E402
    align_judges,
    balanced_accuracy,
    load_judge,
    lr_fusion_ensemble,
    oracle_upper_bound,
    per_dataset_ba,
    routed_ensemble,
    voting_ensemble,
)

# ── Fixtures ────────────────────────────────────────────────────────────


def _write_judge_json(path: Path, preds, labels, datasets, scores=None, model="test"):
    data = {
        "model": model,
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets,
    }
    if scores is not None:
        data["scores"] = scores
    path.write_text(json.dumps(data))


# ── load_judge ──────────────────────────────────────────────────────────


class TestLoadJudge:
    def test_basic_load(self, tmp_path):
        p = tmp_path / "judge_a.json"
        _write_judge_json(p, [1, 0], [1, 0], ["ds_a", "ds_a"], scores=[0.9, 0.1])
        j = load_judge(str(p))
        assert j["name"] == "judge_a"
        assert j["preds"] == [1, 0]
        assert j["labels"] == [1, 0]
        assert j["scores"] == [0.9, 0.1]

    def test_no_scores_gives_none(self, tmp_path):
        p = tmp_path / "judge_b.json"
        _write_judge_json(p, [1, 0], [1, 0], ["ds_a", "ds_a"])
        j = load_judge(str(p))
        assert j["scores"] is None

    def test_length_mismatch_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {
                    "model": "x",
                    "predictions": [1, 0, 1],
                    "labels": [1, 0],
                    "datasets_per_sample": ["a", "b"],
                }
            )
        )
        with pytest.raises(ValueError, match="inconsistent"):
            load_judge(str(p))

    def test_name_from_stem(self, tmp_path):
        p = tmp_path / "my_judge_name.json"
        _write_judge_json(p, [1], [1], ["a"])
        j = load_judge(str(p))
        assert j["name"] == "my_judge_name"


# ── align_judges ────────────────────────────────────────────────────────


class TestAlignJudges:
    def test_matching_judges(self):
        j1 = {
            "name": "a",
            "preds": [1, 0],
            "scores": [0.9, 0.1],
            "labels": [1, 0],
            "datasets": ["d1", "d1"],
        }
        j2 = {
            "name": "b",
            "preds": [0, 1],
            "scores": [0.2, 0.8],
            "labels": [1, 0],
            "datasets": ["d1", "d1"],
        }
        labels, datasets, preds_m, scores_m = align_judges([j1, j2])
        assert labels == [1, 0]
        assert len(preds_m) == 2

    def test_label_mismatch_raises(self):
        j1 = {
            "name": "a",
            "preds": [1],
            "scores": None,
            "labels": [1],
            "datasets": ["d"],
        }
        j2 = {
            "name": "b",
            "preds": [0],
            "scores": None,
            "labels": [0],
            "datasets": ["d"],
        }
        with pytest.raises(ValueError, match="label mismatch"):
            align_judges([j1, j2])

    def test_dataset_mismatch_raises(self):
        j1 = {
            "name": "a",
            "preds": [1],
            "scores": None,
            "labels": [1],
            "datasets": ["d1"],
        }
        j2 = {
            "name": "b",
            "preds": [0],
            "scores": None,
            "labels": [1],
            "datasets": ["d2"],
        }
        with pytest.raises(ValueError, match="dataset mismatch"):
            align_judges([j1, j2])


# ── balanced_accuracy ───────────────────────────────────────────────────


class TestBalancedAccuracy:
    def test_perfect(self):
        assert balanced_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_all_wrong(self):
        assert balanced_accuracy([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0

    def test_unknowns_filtered(self):
        assert balanced_accuracy([1, -1, -1, 0], [1, 0, 1, 0]) == 1.0

    def test_empty(self):
        assert balanced_accuracy([], []) == 0.0

    def test_single_class_returns_zero(self):
        assert balanced_accuracy([1, 1], [1, 1]) == 0.0


# ── per_dataset_ba ──────────────────────────────────────────────────────


class TestPerDatasetBa:
    def test_groups_correctly(self):
        preds = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        datasets = ["a", "a", "b", "b"]
        result = per_dataset_ba(preds, labels, datasets)
        assert "a" in result
        assert "b" in result
        assert result["a"]["balanced_accuracy"] == 1.0
        assert result["b"]["balanced_accuracy"] == 1.0

    def test_samples_count(self):
        result = per_dataset_ba([1, 0, 1], [1, 0, 1], ["a", "a", "b"])
        assert result["a"]["samples"] == 2
        assert result["b"]["samples"] == 1


# ── voting_ensemble ─────────────────────────────────────────────────────


class TestVotingEnsemble:
    def test_unanimous(self):
        preds = [[1, 0], [1, 0], [1, 0]]
        assert voting_ensemble(preds) == [1, 0]

    def test_majority(self):
        preds = [[1, 0], [1, 1], [1, 0]]
        assert voting_ensemble(preds) == [1, 0]

    def test_tie_breaks_to_first_judge(self):
        preds = [[1, 0], [0, 1]]
        result = voting_ensemble(preds)
        # Tie → judge #0 wins
        assert result == [1, 0]

    def test_all_abstain(self):
        preds = [[-1, -1], [-1, -1]]
        assert voting_ensemble(preds) == [-1, -1]

    def test_partial_abstain(self):
        preds = [[-1, 1], [0, -1]]
        result = voting_ensemble(preds)
        assert result == [0, 1]


# ── routed_ensemble ─────────────────────────────────────────────────────


class TestRoutedEnsemble:
    def test_picks_best_per_dataset(self):
        """Judge 0 is perfect on ds_a, judge 1 is perfect on ds_b."""
        # 8 samples: 4 in ds_a, 4 in ds_b
        labels = [1, 0, 1, 0, 1, 0, 1, 0]
        datasets = ["a", "a", "a", "a", "b", "b", "b", "b"]
        # Judge 0: perfect on a, all-wrong on b
        # Judge 1: all-wrong on a, perfect on b
        preds_m = [
            [1, 0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 0],
        ]
        result, routing = routed_ensemble(preds_m, labels, datasets, ["j0", "j1"])
        # Routing should pick j0 for "a" and j1 for "b"
        assert routing["a"] == "j0"
        assert routing["b"] == "j1"

    def test_returns_correct_length(self):
        labels = [1, 0]
        datasets = ["a", "a"]
        preds_m = [[1, 0], [0, 1]]
        result, routing = routed_ensemble(preds_m, labels, datasets, ["j0", "j1"])
        assert len(result) == 2


# ── lr_fusion_ensemble ──────────────────────────────────────────────────


class TestLrFusionEnsemble:
    def test_output_length(self):
        rng = np.random.default_rng(42)
        n = 100
        scores_m = [rng.random(n).tolist(), rng.random(n).tolist()]
        labels = (rng.random(n) > 0.5).astype(int).tolist()
        datasets = [f"ds_{i % 3}" for i in range(n)]
        result = lr_fusion_ensemble(scores_m, labels, datasets)
        assert len(result) == n

    def test_predictions_are_binary(self):
        rng = np.random.default_rng(42)
        n = 100
        scores_m = [rng.random(n).tolist()]
        labels = (rng.random(n) > 0.5).astype(int).tolist()
        datasets = ["ds_0"] * n
        result = lr_fusion_ensemble(scores_m, labels, datasets)
        assert all(p in (0, 1) for p in result)


# ── oracle_upper_bound ──────────────────────────────────────────────────


class TestOracleUpperBound:
    def test_perfect_if_any_judge_correct(self):
        preds_m = [[1, 0, 0], [0, 1, 0]]
        labels = [1, 1, 0]
        result = oracle_upper_bound(preds_m, labels)
        # Sample 0: judge 0 correct; Sample 1: judge 1 correct; Sample 2: both correct
        assert result == [1, 1, 0]

    def test_wrong_if_no_judge_correct(self):
        preds_m = [[0], [0]]
        labels = [1]
        result = oracle_upper_bound(preds_m, labels)
        assert result == [0]  # 1 - target = 0

    def test_ba_with_oracle(self):
        preds_m = [[1, 0, 0, 1], [0, 1, 1, 0]]
        labels = [1, 0, 1, 0]
        oracle = oracle_upper_bound(preds_m, labels)
        ba = balanced_accuracy(oracle, labels)
        assert ba == 1.0  # Oracle should get everything right


# ── main() CLI integration ──────────────────────────────────────────────


class TestMainCli:
    def _make_judge_file(self, tmp_path, name, preds, labels, datasets, scores=None):
        p = tmp_path / f"{name}.json"
        _write_judge_json(p, preds, labels, datasets, scores=scores, model=name)
        return str(p)

    def test_full_run(self, tmp_path):
        # Need at least 5 samples per class for 5-fold StratifiedKFold
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        datasets = ["a"] * 6 + ["b"] * 6
        j1 = self._make_judge_file(
            tmp_path,
            "judge_1",
            [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            labels,
            datasets,
            scores=[0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.8, 0.6, 0.3, 0.4, 0.7, 0.2],
        )
        j2 = self._make_judge_file(
            tmp_path,
            "judge_2",
            [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
            labels,
            datasets,
            scores=[0.8, 0.6, 0.3, 0.2, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.6],
        )
        out = tmp_path / "report.json"
        with patch(
            "sys.argv",
            [
                "prog",
                "--judges",
                j1,
                j2,
                "--output",
                str(out),
            ],
        ):
            from sentinel_judge_analyser import main

            main()
        assert out.exists()
        report = json.loads(out.read_text())

        # Schema checks
        assert "judges" in report
        assert "individual" in report
        assert "voting" in report
        assert "routed" in report
        assert "oracle_upper_bound" in report
        assert report["samples"] == 12
        assert len(report["judges"]) == 2

    def test_lr_fusion_skipped_without_scores(self, tmp_path):
        labels = [1, 0, 1, 0]
        datasets = ["a", "a", "b", "b"]
        j1 = self._make_judge_file(tmp_path, "j1", [1, 0, 1, 0], labels, datasets)
        j2 = self._make_judge_file(tmp_path, "j2", [0, 1, 0, 1], labels, datasets)
        out = tmp_path / "report_noscores.json"
        with patch(
            "sys.argv",
            [
                "prog",
                "--judges",
                j1,
                j2,
                "--output",
                str(out),
            ],
        ):
            from sentinel_judge_analyser import main

            main()
        report = json.loads(out.read_text())
        assert report["lr_fusion"] is None

    def test_lr_fusion_present_with_scores(self, tmp_path):
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        datasets = ["a"] * 6 + ["b"] * 6
        scores1 = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.6, 0.4]
        scores2 = [0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
        preds1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        preds2 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        j1 = self._make_judge_file(
            tmp_path, "j1", preds1, labels, datasets, scores=scores1
        )
        j2 = self._make_judge_file(
            tmp_path, "j2", preds2, labels, datasets, scores=scores2
        )
        out = tmp_path / "report_scores.json"
        with patch(
            "sys.argv",
            [
                "prog",
                "--judges",
                j1,
                j2,
                "--output",
                str(out),
            ],
        ):
            from sentinel_judge_analyser import main

            main()
        report = json.loads(out.read_text())
        assert report["lr_fusion"] is not None
        assert "global_balanced_accuracy" in report["lr_fusion"]

    def test_voting_ba_in_report(self, tmp_path):
        labels = [1, 0, 1, 0]
        datasets = ["a", "a", "b", "b"]
        j1 = self._make_judge_file(tmp_path, "j1", [1, 0, 1, 0], labels, datasets)
        j2 = self._make_judge_file(tmp_path, "j2", [1, 0, 1, 0], labels, datasets)
        out = tmp_path / "report_vote.json"
        with patch(
            "sys.argv",
            [
                "prog",
                "--judges",
                j1,
                j2,
                "--output",
                str(out),
            ],
        ):
            from sentinel_judge_analyser import main

            main()
        report = json.loads(out.read_text())
        assert report["voting"]["global_balanced_accuracy"] == 1.0
