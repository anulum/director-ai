# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Unit guard coverage for ``benchmarks.sentinel_judge_analyser``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy", reason="numpy required for sentinel_judge_analyser")
pytest.importorskip("sklearn", reason="sklearn required for sentinel_judge_analyser")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from sentinel_judge_analyser import (  # noqa: E402
    JudgeRecord,
    align_judges,
    balanced_accuracy,
    load_judge,
    lr_fusion_ensemble,
    main,
    oracle_upper_bound,
    per_dataset_ba,
    routed_ensemble,
    voting_ensemble,
)


def _write_judge_json(
    path: Path,
    preds: list[int],
    labels: list[int],
    datasets: list[str],
    scores: list[float] | None = None,
    model: str = "test",
) -> None:
    """Write one judge-result JSON fixture."""
    data: dict[str, object] = {
        "model": model,
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets,
    }
    if scores is not None:
        data["scores"] = scores
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_report(path: Path) -> dict[str, Any]:
    """Read a generated report as a typed mapping."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


class TestLoadJudge:
    """Unit guard coverage for judge JSON loading."""

    def test_basic_load(self, tmp_path: Path) -> None:
        """Numeric score lists should load unchanged."""
        p = tmp_path / "judge_a.json"
        _write_judge_json(p, [1, 0], [1, 0], ["ds_a", "ds_a"], scores=[0.9, 0.1])
        j = load_judge(str(p))

        assert j["name"] == "judge_a"
        assert j["preds"] == [1, 0]
        assert j["labels"] == [1, 0]
        assert j["scores"] == [0.9, 0.1]

    def test_no_scores_gives_none(self, tmp_path: Path) -> None:
        """Missing scores should remain unavailable rather than fake-filled."""
        p = tmp_path / "judge_b.json"
        _write_judge_json(p, [1, 0], [1, 0], ["ds_a", "ds_a"])

        assert load_judge(str(p))["scores"] is None

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        """Prediction, label, and dataset length mismatches should fail."""
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {
                    "model": "x",
                    "predictions": [1, 0, 1],
                    "labels": [1, 0],
                    "datasets_per_sample": ["a", "b"],
                },
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="inconsistent"):
            load_judge(str(p))

    def test_name_from_stem(self, tmp_path: Path) -> None:
        """The stable judge name should come from the filename stem."""
        p = tmp_path / "my_judge_name.json"
        _write_judge_json(p, [1], [1], ["a"])

        assert load_judge(str(p))["name"] == "my_judge_name"


class TestAlignJudges:
    """Unit guard coverage for judge alignment."""

    def test_matching_judges(self) -> None:
        """Aligned judges should produce shared labels and prediction matrices."""
        j1: JudgeRecord = {
            "name": "a",
            "model": "a",
            "preds": [1, 0],
            "scores": [0.9, 0.1],
            "labels": [1, 0],
            "datasets": ["d1", "d1"],
        }
        j2: JudgeRecord = {
            "name": "b",
            "model": "b",
            "preds": [0, 1],
            "scores": [0.2, 0.8],
            "labels": [1, 0],
            "datasets": ["d1", "d1"],
        }

        labels, _datasets, preds_m, _scores_m = align_judges([j1, j2])

        assert labels == [1, 0]
        assert len(preds_m) == 2

    def test_label_mismatch_raises(self) -> None:
        """Label-order mismatches should be rejected."""
        j1: JudgeRecord = {
            "name": "a",
            "model": "a",
            "preds": [1],
            "scores": None,
            "labels": [1],
            "datasets": ["d"],
        }
        j2: JudgeRecord = {
            "name": "b",
            "model": "b",
            "preds": [0],
            "scores": None,
            "labels": [0],
            "datasets": ["d"],
        }

        with pytest.raises(ValueError, match="label mismatch"):
            align_judges([j1, j2])

    def test_dataset_mismatch_raises(self) -> None:
        """Dataset-order mismatches should be rejected."""
        j1: JudgeRecord = {
            "name": "a",
            "model": "a",
            "preds": [1],
            "scores": None,
            "labels": [1],
            "datasets": ["d1"],
        }
        j2: JudgeRecord = {
            "name": "b",
            "model": "b",
            "preds": [0],
            "scores": None,
            "labels": [1],
            "datasets": ["d2"],
        }

        with pytest.raises(ValueError, match="dataset mismatch"):
            align_judges([j1, j2])


class TestBalancedAccuracy:
    """Unit guard coverage for balanced accuracy."""

    def test_perfect(self) -> None:
        """Perfect predictions should score 1.0."""
        assert balanced_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_all_wrong(self) -> None:
        """Fully inverted predictions should score 0.0."""
        assert balanced_accuracy([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0

    def test_unknowns_filtered(self) -> None:
        """Abstentions should be filtered from metric calculation."""
        assert balanced_accuracy([1, -1, -1, 0], [1, 0, 1, 0]) == 1.0

    def test_empty(self) -> None:
        """Empty inputs should not divide by zero."""
        assert balanced_accuracy([], []) == 0.0

    def test_single_class_returns_zero(self) -> None:
        """Single-class inputs should report no balanced-accuracy signal."""
        assert balanced_accuracy([1, 1], [1, 1]) == 0.0


class TestPerDatasetBa:
    """Unit guard coverage for dataset grouping."""

    def test_groups_correctly(self) -> None:
        """Dataset metrics should be grouped under each dataset key."""
        result = per_dataset_ba(
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            ["a", "a", "b", "b"],
        )

        assert result["a"]["balanced_accuracy"] == 1.0
        assert result["b"]["balanced_accuracy"] == 1.0

    def test_samples_count(self) -> None:
        """Per-dataset metrics should include sample counts."""
        result = per_dataset_ba([1, 0, 1], [1, 0, 1], ["a", "a", "b"])

        assert result["a"]["samples"] == 2
        assert result["b"]["samples"] == 1


class TestVotingEnsemble:
    """Unit guard coverage for majority voting."""

    def test_unanimous(self) -> None:
        """Unanimous predictions should pass through unchanged."""
        assert voting_ensemble([[1, 0], [1, 0], [1, 0]]) == [1, 0]

    def test_majority(self) -> None:
        """A strict majority should select the majority class."""
        assert voting_ensemble([[1, 0], [1, 1], [1, 0]]) == [1, 0]

    def test_tie_breaks_to_first_judge(self) -> None:
        """Ties should break toward the first non-abstaining judge."""
        assert voting_ensemble([[1, 0], [0, 1]]) == [1, 0]

    def test_all_abstain(self) -> None:
        """All-abstain samples should remain abstentions."""
        assert voting_ensemble([[-1, -1], [-1, -1]]) == [-1, -1]

    def test_partial_abstain(self) -> None:
        """Single non-abstaining votes should be used."""
        assert voting_ensemble([[-1, 1], [0, -1]]) == [0, 1]


class TestRoutedEnsemble:
    """Unit guard coverage for per-dataset routing."""

    def test_picks_best_per_dataset(self) -> None:
        """The router should pick each dataset's best train-half judge."""
        labels = [1, 0, 1, 0, 1, 0, 1, 0]
        datasets = ["a", "a", "a", "a", "b", "b", "b", "b"]
        preds_m = [
            [1, 0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 0],
        ]

        result, routing = routed_ensemble(preds_m, labels, datasets, ["j0", "j1"])

        assert len(result) == 8
        assert routing["a"] == "j0"
        assert routing["b"] == "j1"

    def test_returns_correct_length(self) -> None:
        """Routed predictions should preserve sample count."""
        result, _routing = routed_ensemble(
            [[1, 0], [0, 1]], [1, 0], ["a", "a"], ["j0", "j1"]
        )

        assert len(result) == 2


class TestLrFusionEnsemble:
    """Unit guard coverage for logistic-regression fusion."""

    def test_output_length(self) -> None:
        """Out-of-fold fusion should return one prediction per input sample."""
        rng = np.random.default_rng(42)
        n = 100
        scores_m = cast(
            list[list[float]],
            [rng.random(n).tolist(), rng.random(n).tolist()],
        )
        labels = cast(list[int], (rng.random(n) > 0.5).astype(int).tolist())
        datasets = [f"ds_{i % 3}" for i in range(n)]

        assert len(lr_fusion_ensemble(scores_m, labels, datasets)) == n

    def test_predictions_are_binary(self) -> None:
        """Fusion predictions should be binary labels."""
        rng = np.random.default_rng(42)
        n = 100
        scores_m = cast(list[list[float]], [rng.random(n).tolist()])
        labels = cast(list[int], (rng.random(n) > 0.5).astype(int).tolist())
        datasets = ["ds_0"] * n

        assert all(
            pred in (0, 1) for pred in lr_fusion_ensemble(scores_m, labels, datasets)
        )


class TestOracleUpperBound:
    """Unit guard coverage for the oracle upper-bound metric."""

    def test_perfect_if_any_judge_correct(self) -> None:
        """Oracle output should pick a correct judge when any judge is correct."""
        assert oracle_upper_bound([[1, 0, 0], [0, 1, 0]], [1, 1, 0]) == [1, 1, 0]

    def test_wrong_if_no_judge_correct(self) -> None:
        """Oracle output should invert when no judge is correct."""
        assert oracle_upper_bound([[0], [0]], [1]) == [0]

    def test_ba_with_oracle(self) -> None:
        """Oracle predictions should report perfect BA when coverage exists."""
        oracle = oracle_upper_bound([[1, 0, 0, 1], [0, 1, 1, 0]], [1, 0, 1, 0])

        assert balanced_accuracy(oracle, [1, 0, 1, 0]) == 1.0


class TestMainCli:
    """Unit guard coverage for in-process CLI invocation."""

    def _make_judge_file(
        self,
        tmp_path: Path,
        name: str,
        preds: list[int],
        labels: list[int],
        datasets: list[str],
        scores: list[float] | None = None,
    ) -> str:
        """Write one judge fixture and return its path."""
        p = tmp_path / f"{name}.json"
        _write_judge_json(p, preds, labels, datasets, scores=scores, model=name)
        return str(p)

    def test_full_run(self, tmp_path: Path) -> None:
        """The in-process CLI should write the expected report schema."""
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

        with patch("sys.argv", ["prog", "--judges", j1, j2, "--output", str(out)]):
            assert main() == 0

        report = _read_report(out)
        assert "judges" in report
        assert "individual" in report
        assert "voting" in report
        assert "routed" in report
        assert "oracle_upper_bound" in report
        assert report["samples"] == 12
        assert len(report["judges"]) == 2

    def test_lr_fusion_skipped_without_scores(self, tmp_path: Path) -> None:
        """The report should omit LR fusion when any judge lacks scores."""
        labels = [1, 0, 1, 0]
        datasets = ["a", "a", "b", "b"]
        j1 = self._make_judge_file(tmp_path, "j1", [1, 0, 1, 0], labels, datasets)
        j2 = self._make_judge_file(tmp_path, "j2", [0, 1, 0, 1], labels, datasets)
        out = tmp_path / "report_noscores.json"

        with patch("sys.argv", ["prog", "--judges", j1, j2, "--output", str(out)]):
            assert main() == 0

        assert _read_report(out)["lr_fusion"] is None

    def test_lr_fusion_present_with_scores(self, tmp_path: Path) -> None:
        """The report should include LR fusion when all judges provide scores."""
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        datasets = ["a"] * 6 + ["b"] * 6
        scores1 = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.6, 0.4]
        scores2 = [0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
        preds1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        preds2 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        j1 = self._make_judge_file(
            tmp_path,
            "j1",
            preds1,
            labels,
            datasets,
            scores=scores1,
        )
        j2 = self._make_judge_file(
            tmp_path,
            "j2",
            preds2,
            labels,
            datasets,
            scores=scores2,
        )
        out = tmp_path / "report_scores.json"

        with patch("sys.argv", ["prog", "--judges", j1, j2, "--output", str(out)]):
            assert main() == 0

        lr_fusion = _read_report(out)["lr_fusion"]
        assert lr_fusion is not None
        assert "global_balanced_accuracy" in lr_fusion

    def test_voting_ba_in_report(self, tmp_path: Path) -> None:
        """The voting report should include balanced accuracy."""
        labels = [1, 0, 1, 0]
        datasets = ["a", "a", "b", "b"]
        j1 = self._make_judge_file(tmp_path, "j1", [1, 0, 1, 0], labels, datasets)
        j2 = self._make_judge_file(tmp_path, "j2", [1, 0, 1, 0], labels, datasets)
        out = tmp_path / "report_vote.json"

        with patch("sys.argv", ["prog", "--judges", j1, j2, "--output", str(out)]):
            assert main() == 0

        assert _read_report(out)["voting"]["global_balanced_accuracy"] == 1.0
