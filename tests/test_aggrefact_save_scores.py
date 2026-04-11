# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for ``benchmarks.aggrefact_eval.score_and_save`` (STRONG).

Covers the v2 ensemble-compatible JSON schema introduced 2026-04-11:

- v2 schema completeness (every documented field present)
- semantic correctness of ``global_balanced_accuracy`` (sample-pooled BA)
  vs ``per_dataset_avg_balanced_accuracy`` (mean of per-dataset BAs)
- ``load_cached_scores`` round-trip on v2
- ``load_cached_scores`` backward compat with v1 (legacy) files
- ``train_dataset_classifier.load_cached_scores`` schema-aware path

No HuggingFace dataset access — uses an in-process monkeypatch on
``_load_aggrefact`` and a stub for ``_BinaryNLIPredictor`` so the test
runs without HF_TOKEN, network, or models loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make benchmarks/ importable as a package even when run via plain pytest.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from benchmarks import aggrefact_eval as ae  # noqa: E402

# ── Fixtures ─────────────────────────────────────────────────────────────


def _synthetic_rows() -> list[dict]:
    """24 deterministic samples across 3 datasets with hand-tuned scores.

    The score-to-label correlation differs per dataset so per-dataset
    thresholds and the global threshold give measurably different BAs.
    """
    return [
        # easy_dataset: clean separation, optimal t ~ 0.5
        {"dataset": "easy", "doc": "d", "claim": "c", "label": 1, "_score": 0.95},
        {"dataset": "easy", "doc": "d", "claim": "c", "label": 1, "_score": 0.92},
        {"dataset": "easy", "doc": "d", "claim": "c", "label": 1, "_score": 0.88},
        {"dataset": "easy", "doc": "d", "claim": "c", "label": 0, "_score": 0.10},
        {"dataset": "easy", "doc": "d", "claim": "c", "label": 0, "_score": 0.15},
        {"dataset": "easy", "doc": "d", "claim": "c", "label": 0, "_score": 0.05},
        # mid_dataset: optimum t ~ 0.7
        {"dataset": "mid", "doc": "d", "claim": "c", "label": 1, "_score": 0.85},
        {"dataset": "mid", "doc": "d", "claim": "c", "label": 1, "_score": 0.78},
        {"dataset": "mid", "doc": "d", "claim": "c", "label": 1, "_score": 0.72},
        {"dataset": "mid", "doc": "d", "claim": "c", "label": 0, "_score": 0.62},
        {"dataset": "mid", "doc": "d", "claim": "c", "label": 0, "_score": 0.55},
        {"dataset": "mid", "doc": "d", "claim": "c", "label": 0, "_score": 0.45},
        # hard_dataset: optimum t ~ 0.3, scores compressed
        {"dataset": "hard", "doc": "d", "claim": "c", "label": 1, "_score": 0.45},
        {"dataset": "hard", "doc": "d", "claim": "c", "label": 1, "_score": 0.38},
        {"dataset": "hard", "doc": "d", "claim": "c", "label": 1, "_score": 0.32},
        {"dataset": "hard", "doc": "d", "claim": "c", "label": 0, "_score": 0.28},
        {"dataset": "hard", "doc": "d", "claim": "c", "label": 0, "_score": 0.20},
        {"dataset": "hard", "doc": "d", "claim": "c", "label": 0, "_score": 0.12},
        # 6 extra samples to keep BAs non-degenerate
        {"dataset": "easy", "doc": "d", "claim": "c", "label": 1, "_score": 0.99},
        {"dataset": "easy", "doc": "d", "claim": "c", "label": 0, "_score": 0.02},
        {"dataset": "mid", "doc": "d", "claim": "c", "label": 1, "_score": 0.90},
        {"dataset": "mid", "doc": "d", "claim": "c", "label": 0, "_score": 0.40},
        {"dataset": "hard", "doc": "d", "claim": "c", "label": 1, "_score": 0.50},
        {"dataset": "hard", "doc": "d", "claim": "c", "label": 0, "_score": 0.08},
    ]


class _StubPredictor:
    """Returns the per-row ``_score`` field set in the synthetic fixture."""

    def __init__(self, model_name=None):
        self.model_name = model_name
        self._iter = None

    def score(self, doc: str, claim: str) -> float:  # noqa: ARG002
        # Pull next score from the queue installed by the test.
        return next(self._iter)


@pytest.fixture()
def synthetic_run(tmp_path, monkeypatch):
    """Run ``score_and_save`` against the synthetic dataset.

    Returns a tuple ``(json_path, parsed_json, expected_pooled_ba)`` so
    individual tests can assert on whatever slice they care about.
    """
    rows = _synthetic_rows()
    score_iter = iter([r["_score"] for r in rows])

    stub = _StubPredictor()
    stub._iter = score_iter

    monkeypatch.setattr(ae, "_load_aggrefact", lambda max_samples=None: rows)
    monkeypatch.setattr(ae, "_BinaryNLIPredictor", lambda model_name=None: stub)

    out = tmp_path / "factcg_synthetic.json"
    ae.score_and_save(out, max_samples=None, model_name="stub-model")
    parsed = json.loads(out.read_text())

    # Reference: pooled BA at the threshold the loader found globally.
    pooled_threshold = parsed["global_threshold"]
    y_true = [r["label"] for r in rows]
    y_pred = [1 if r["_score"] >= pooled_threshold else 0 for r in rows]
    expected_pooled_ba = balanced_accuracy_score(y_true, y_pred)

    return out, parsed, expected_pooled_ba


# ── Schema completeness ──────────────────────────────────────────────────


REQUIRED_KEYS = {
    "schema_version",
    "model",
    "backend",
    "samples",
    "global_balanced_accuracy",
    "global_threshold",
    "per_dataset_avg_balanced_accuracy",
    "per_dataset",
    "per_dataset_thresholds",
    "scores",
    "predictions",
    "labels",
    "datasets_per_sample",
    "unknown_predictions",
    "total_time_seconds",
    "mean_latency_ms",
    "p50_latency_ms",
    "p99_latency_ms",
}


class TestSchemaCompleteness:
    def test_all_required_keys_present(self, synthetic_run):
        _, parsed, _ = synthetic_run
        missing = REQUIRED_KEYS - set(parsed.keys())
        assert not missing, f"Missing top-level keys: {sorted(missing)}"

    def test_schema_version_is_v2(self, synthetic_run):
        _, parsed, _ = synthetic_run
        assert parsed["schema_version"] == 2

    def test_parallel_lists_have_equal_length(self, synthetic_run):
        _, parsed, _ = synthetic_run
        n = parsed["samples"]
        assert len(parsed["scores"]) == n
        assert len(parsed["labels"]) == n
        assert len(parsed["predictions"]) == n
        assert len(parsed["datasets_per_sample"]) == n

    def test_predictions_are_binary(self, synthetic_run):
        _, parsed, _ = synthetic_run
        assert set(parsed["predictions"]) <= {0, 1}

    def test_labels_are_binary(self, synthetic_run):
        _, parsed, _ = synthetic_run
        assert set(parsed["labels"]) <= {0, 1}

    def test_per_dataset_block_has_three_datasets(self, synthetic_run):
        _, parsed, _ = synthetic_run
        assert set(parsed["per_dataset"].keys()) == {"easy", "mid", "hard"}

    def test_per_dataset_thresholds_match_per_dataset_block(self, synthetic_run):
        _, parsed, _ = synthetic_run
        assert set(parsed["per_dataset_thresholds"]) == set(parsed["per_dataset"])


# ── Semantic correctness — the bug Gemini introduced ─────────────────────


class TestBalancedAccuracySemantics:
    def test_global_ba_is_sample_pooled_not_per_dataset_average(self, synthetic_run):
        _, parsed, expected_pooled_ba = synthetic_run
        # global_balanced_accuracy must equal the BA computed at a single
        # global threshold across the pooled samples — NOT the unweighted
        # mean of per-dataset BAs.
        assert parsed["global_balanced_accuracy"] == pytest.approx(
            expected_pooled_ba, abs=1e-6
        )

    def test_per_dataset_avg_ba_differs_from_global_ba(self, synthetic_run):
        _, parsed, _ = synthetic_run
        # On heterogeneous data with per-dataset thresholds the per-dataset
        # average is strictly higher than (or equal to) the single-threshold
        # global BA. We require strict inequality on this fixture to prove
        # the two metrics are not the same field.
        assert (
            parsed["per_dataset_avg_balanced_accuracy"]
            >= parsed["global_balanced_accuracy"]
        )
        assert (
            parsed["per_dataset_avg_balanced_accuracy"]
            > parsed["global_balanced_accuracy"]
        ), (
            "fixture should expose semantic difference; if this fires, the "
            "fixture became degenerate, not the implementation"
        )

    def test_global_threshold_is_in_sweep_range(self, synthetic_run):
        _, parsed, _ = synthetic_run
        # sweep_on_cached scans 0.10..0.90 in steps of 0.01.
        assert 0.10 <= parsed["global_threshold"] <= 0.90


# ── load_cached_scores round-trip on v2 ──────────────────────────────────


class TestLoadCachedScoresV2:
    def test_round_trip_returns_per_dataset_dict(self, synthetic_run):
        out, _, _ = synthetic_run
        loaded = ae.load_cached_scores(out)
        assert set(loaded.keys()) == {"easy", "mid", "hard"}
        for pairs in loaded.values():
            for pair in pairs:
                assert isinstance(pair, tuple) and len(pair) == 2
                lbl, scr = pair
                assert lbl in (0, 1)
                assert 0.0 <= scr <= 1.0

    def test_total_sample_count_matches_input(self, synthetic_run):
        out, parsed, _ = synthetic_run
        loaded = ae.load_cached_scores(out)
        total = sum(len(v) for v in loaded.values())
        assert total == parsed["samples"]


# ── load_cached_scores backward compat with v1 (legacy) ──────────────────


class TestLoadCachedScoresV1Backcompat:
    def test_legacy_v1_layout_still_loads(self, tmp_path):
        legacy = {
            "model": "legacy-model",
            "total": 4,
            "scores": [
                {"dataset": "ds_a", "label": 1, "score": 0.9, "latency_ms": 1.0},
                {"dataset": "ds_a", "label": 0, "score": 0.1, "latency_ms": 1.0},
                {"dataset": "ds_b", "label": 1, "score": 0.8, "latency_ms": 1.0},
                {"dataset": "ds_b", "label": 0, "score": 0.2, "latency_ms": 1.0},
            ],
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy))

        loaded = ae.load_cached_scores(path)
        assert set(loaded.keys()) == {"ds_a", "ds_b"}
        assert (1, 0.9) in loaded["ds_a"]
        assert (0, 0.2) in loaded["ds_b"]


# ── train_dataset_classifier external loader ─────────────────────────────


class TestTrainDatasetClassifierLoader:
    def test_v2_layout_loads_into_index_keyed_dict(self, synthetic_run):
        # Ensure the external loader in tools/ also handles v2.
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        try:
            with patch.dict(sys.modules):
                import importlib

                tdc = importlib.import_module("train_dataset_classifier")
                out, parsed, _ = synthetic_run
                cached = tdc.load_cached_scores(str(out))
                assert len(cached) == parsed["samples"]
                # Spot-check first entry has the dict shape downstream code expects.
                assert {"dataset", "label", "score"} <= set(cached[0].keys())
        finally:
            sys.path.remove(str(PROJECT_ROOT / "tools"))

    def test_v1_layout_still_loads_via_train_dataset_classifier(self, tmp_path):
        legacy = {
            "model": "legacy",
            "total": 2,
            "scores": [
                {"dataset": "x", "label": 1, "score": 0.7, "latency_ms": 1.0},
                {"dataset": "x", "label": 0, "score": 0.3, "latency_ms": 1.0},
            ],
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy))

        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        try:
            import importlib

            tdc = importlib.import_module("train_dataset_classifier")
            cached = tdc.load_cached_scores(str(path))
            assert len(cached) == 2
            assert cached[0]["dataset"] == "x"
        finally:
            sys.path.remove(str(PROJECT_ROOT / "tools"))


# ── Inconsistent schema is rejected (defensive) ──────────────────────────


class TestSchemaValidation:
    def test_v2_with_mismatched_lengths_raises(self, tmp_path):
        bad = {
            "schema_version": 2,
            "scores": [0.1, 0.2, 0.3],
            "labels": [1, 0],  # length mismatch
            "datasets_per_sample": ["a", "b", "c"],
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="inconsistent"):
            ae.load_cached_scores(path)
