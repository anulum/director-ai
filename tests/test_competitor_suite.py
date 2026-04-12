# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for ``benchmarks.competitor_aggrefact_suite``.

Covers:

* ``balanced_accuracy()`` semantics across edge cases (empty, all-correct,
  all-wrong, single-class, threshold-edge, unknown filtering).
* ``MockBackend`` constructor validation.
* ``run_suite()`` driver: result schema completeness, threshold cut,
  per-sample latencies present, label/dataset preservation, max_samples
  truncation, exception path -> -1 prediction, output file write.
* Stub-backend safety: every commercial backend raises
  ``NotImplementedError`` from ``score()``. This is the regression
  guard that prevents the suite from ever silently emitting fabricated
  competitor numbers.
* CLI registry sanity: every key in ``BACKENDS`` resolves to a
  ``BaseBackend`` subclass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from competitor_aggrefact_suite import (  # noqa: E402
    BACKENDS,
    AzureBackend,
    BaseBackend,
    GalileoBackend,
    GuardrailsBackend,
    LakeraBackend,
    MockBackend,
    balanced_accuracy,
    run_suite,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


def _toy_dataset() -> list[dict]:
    """Two datasets, mixed labels, deterministic."""
    return [
        {"doc": "ctx", "claim": "c1", "label": 1, "dataset": "ds_a"},
        {"doc": "ctx", "claim": "c2", "label": 0, "dataset": "ds_a"},
        {"doc": "ctx", "claim": "c3", "label": 1, "dataset": "ds_b"},
        {"doc": "ctx", "claim": "c4", "label": 0, "dataset": "ds_b"},
    ]


# ── balanced_accuracy ────────────────────────────────────────────────────


class TestBalancedAccuracy:
    def test_all_correct(self):
        assert balanced_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_all_wrong(self):
        assert balanced_accuracy([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0

    def test_half_right(self):
        # tp=1, fn=1, tn=1, fp=1 -> recalls 0.5 / 0.5 -> BA 0.5
        assert balanced_accuracy([1, 0, 0, 1], [1, 0, 1, 0]) == 0.5

    def test_empty_returns_zero(self):
        assert balanced_accuracy([], []) == 0.0

    def test_unknowns_filtered(self):
        # The -1s drop both samples; the remaining two are perfect.
        assert balanced_accuracy([1, -1, -1, 0], [1, 0, 1, 0]) == 1.0

    def test_only_positive_class(self):
        # No negatives -> BA degrades to recall on positives.
        assert balanced_accuracy([1, 1], [1, 1]) == 1.0
        assert balanced_accuracy([1, 0], [1, 1]) == 0.5

    def test_only_negative_class(self):
        assert balanced_accuracy([0, 0], [0, 0]) == 1.0
        assert balanced_accuracy([0, 1], [0, 0]) == 0.5

    def test_all_unknown_returns_zero(self):
        assert balanced_accuracy([-1, -1], [1, 0]) == 0.0


# ── MockBackend ──────────────────────────────────────────────────────────


class TestMockBackend:
    def test_returns_fixed_score(self):
        b = MockBackend(fixed_score=0.42)
        assert b.score("p", "h") == 0.42

    def test_default_score(self):
        assert MockBackend().score("p", "h") == 0.8

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0, -1.0])
    def test_constructor_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError):
            MockBackend(fixed_score=bad)

    def test_constructor_accepts_boundaries(self):
        assert MockBackend(0.0).fixed_score == 0.0
        assert MockBackend(1.0).fixed_score == 1.0


# ── Stub-backend regression guard ────────────────────────────────────────


class TestStubBackendsRaise:
    """The four commercial backends MUST raise NotImplementedError.

    This is the test that prevents anyone from accidentally publishing a
    competitor leaderboard built on hardcoded stub return values.
    """

    @pytest.mark.parametrize(
        "cls",
        [LakeraBackend, GalileoBackend, AzureBackend, GuardrailsBackend],
    )
    def test_score_raises(self, cls):
        with pytest.raises(NotImplementedError):
            cls().score("premise", "hypothesis")

    def test_run_suite_propagates_stub_refusal(self):
        # Stub backends must propagate NotImplementedError out of
        # run_suite() — never get caught and turned into -1 predictions.
        with pytest.raises(NotImplementedError):
            run_suite(LakeraBackend(), dataset=_toy_dataset())


# ── run_suite driver ─────────────────────────────────────────────────────


class TestRunSuite:
    def test_schema_completeness(self):
        results = run_suite(MockBackend(0.9), dataset=_toy_dataset())
        for key in (
            "schema_version",
            "model",
            "backend",
            "samples",
            "global_balanced_accuracy",
            "scores",
            "predictions",
            "labels",
            "datasets_per_sample",
            "latencies_per_sample",
            "total_time_seconds",
        ):
            assert key in results, f"missing key {key!r}"
        assert results["schema_version"] == 2
        assert results["samples"] == 4
        assert len(results["scores"]) == 4
        assert len(results["predictions"]) == 4
        assert len(results["labels"]) == 4
        assert len(results["datasets_per_sample"]) == 4
        assert len(results["latencies_per_sample"]) == 4

    def test_threshold_cut_above(self):
        # fixed score 0.6, threshold 0.5 -> all predicted 1; labels split
        # 1/0/1/0 -> recall_pos 1.0, recall_neg 0.0 -> BA 0.5.
        results = run_suite(MockBackend(0.6), dataset=_toy_dataset(), threshold=0.5)
        assert results["predictions"] == [1, 1, 1, 1]
        assert results["global_balanced_accuracy"] == 0.5

    def test_threshold_cut_below(self):
        # fixed score 0.4, threshold 0.5 -> all predicted 0 -> BA 0.5.
        results = run_suite(MockBackend(0.4), dataset=_toy_dataset(), threshold=0.5)
        assert results["predictions"] == [0, 0, 0, 0]
        assert results["global_balanced_accuracy"] == 0.5

    def test_threshold_at_score_predicts_one(self):
        # score >= threshold => 1. Score == threshold is included.
        results = run_suite(MockBackend(0.5), dataset=_toy_dataset(), threshold=0.5)
        assert results["predictions"] == [1, 1, 1, 1]

    def test_max_samples_truncates(self):
        results = run_suite(MockBackend(), dataset=_toy_dataset(), max_samples=2)
        assert results["samples"] == 2
        assert results["labels"] == [1, 0]

    def test_labels_and_datasets_preserved(self):
        results = run_suite(MockBackend(), dataset=_toy_dataset())
        assert results["labels"] == [1, 0, 1, 0]
        assert results["datasets_per_sample"] == [
            "ds_a",
            "ds_a",
            "ds_b",
            "ds_b",
        ]

    def test_latencies_are_non_negative(self):
        results = run_suite(MockBackend(), dataset=_toy_dataset())
        for lat in results["latencies_per_sample"]:
            assert lat >= 0.0

    def test_runtime_exception_becomes_minus_one(self):
        # A backend that raises a *non-NotImplementedError* should be
        # caught and the sample marked unknown (-1).
        class FlakyBackend(BaseBackend):
            name = "flaky"

            def score(self, premise: str, hypothesis: str) -> float:
                raise RuntimeError("transient")

        results = run_suite(FlakyBackend(), dataset=_toy_dataset())
        assert results["predictions"] == [-1, -1, -1, -1]
        assert all(s == -1.0 for s in results["scores"])

    def test_writes_output_file(self, tmp_path):
        out = tmp_path / "mock.json"
        run_suite(MockBackend(), dataset=_toy_dataset(), output=out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["schema_version"] == 2
        assert loaded["model"] == "mock"

    def test_string_backend_name(self):
        results = run_suite("mock", dataset=_toy_dataset())
        assert results["model"] == "mock"

    def test_unknown_string_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            run_suite("not-a-real-backend", dataset=_toy_dataset())


# ── Registry sanity ──────────────────────────────────────────────────────


class TestBackendRegistry:
    def test_every_backend_is_subclass(self):
        for name, cls in BACKENDS.items():
            assert issubclass(cls, BaseBackend), f"{name} not a BaseBackend"

    def test_registry_contains_expected_keys(self):
        assert set(BACKENDS) == {
            "lakera",
            "galileo",
            "azure",
            "guardrails",
            "mock",
        }

    def test_base_backend_score_raises(self):
        with pytest.raises(NotImplementedError):
            BaseBackend().score("p", "h")
