# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Online Calibrator Tests
"""Multi-angle tests for online threshold calibrator.

Covers: empty store, perfect guardrail, all false positives, mixed errors,
optimal threshold sweep, insufficient corrections, domain filtering,
parametrised min_corrections, pipeline integration, and performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest

import director_ai.core.calibration.online_calibrator as calibrator_mod
from director_ai.core.calibration.feedback_store import FeedbackStore
from director_ai.core.calibration.online_calibrator import OnlineCalibrator


class TestCalibrationReport:
    """Check error-rate summaries and native-versus-Python calibration kernels."""

    def test_wilson_ci_empty_total_is_max_uncertainty(self) -> None:
        """Return maximum interval uncertainty when there are no observations."""
        assert calibrator_mod._wilson_ci(0, 0) == 1.0

    def test_wilson_ci_non_default_z_uses_python_formula(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Use the Python Wilson formula for a nondefault z despite native availability.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Enable the native feature flag and reject any native Wilson dispatch.
        """
        monkeypatch.setattr(calibrator_mod, "_RUST_ONLINE_CALIBRATOR", True)

        def _wilson_should_not_run(*_args: object) -> NoReturn:
            """Reject use of the fixed-confidence native kernel in the nondefault-z test.

            Parameters
            ----------
            *_args : object
                Native kernel arguments, which must never be consumed here.

            Raises
            ------
            AssertionError
                If the implementation incorrectly dispatches to the native kernel.
            """
            raise AssertionError("non-default z must not use Rust 95 percent kernel")

        monkeypatch.setattr(
            calibrator_mod,
            "rust_wilson_score_interval",
            _wilson_should_not_run,
            raising=True,
        )

        spread = calibrator_mod._wilson_ci(5, 10, z=1.0)

        assert 0.0 < spread < 0.5

    def test_rust_confusion_kernel_is_used_in_threshold_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dispatch a threshold sweep through the enabled native confusion-count binding.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Enable native confusion counting and install a call-counting implementation.
        """
        monkeypatch.setattr(calibrator_mod, "_RUST_ONLINE_CALIBRATOR", True)
        called = {"count": 0}

        def _conf(
            scores: list[float], labels: list[bool], threshold: float
        ) -> tuple[int, int, int, int]:
            """Record native dispatch and count outcomes for the supplied score threshold.

            Parameters
            ----------
            scores : list of float
                Candidate scores paired with the provided human labels.
            labels : list of bool
                Human approval flags in the same order as the scores.
            threshold : float
                Inclusive lower score bound for a predicted approval.

            Returns
            -------
            tuple of four int
                True positives, true negatives, false positives and false negatives.
            """
            called["count"] += 1
            tp = sum(
                1 for s, h in zip(scores, labels, strict=True) if s >= threshold and h
            )
            tn = sum(
                1
                for s, h in zip(scores, labels, strict=True)
                if s < threshold and not h
            )
            fp = sum(
                1
                for s, h in zip(scores, labels, strict=True)
                if s >= threshold and not h
            )
            fn = sum(
                1 for s, h in zip(scores, labels, strict=True) if s < threshold and h
            )
            return tp, tn, fp, fn

        monkeypatch.setattr(
            calibrator_mod,
            "rust_confusion_counts_threshold",
            _conf,
            raising=True,
        )
        scored = [(0.9, True), (0.8, True), (0.2, False), (0.3, False)]
        best = OnlineCalibrator._sweep_threshold(scored)
        assert best is not None
        assert called["count"] > 0

    def test_rust_confusion_type_error_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Find a threshold through Python when the native confusion binding raises TypeError.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Enable native confusion counting and inject a native signature error.
        """
        monkeypatch.setattr(calibrator_mod, "_RUST_ONLINE_CALIBRATOR", True)
        monkeypatch.setattr(
            calibrator_mod,
            "rust_confusion_counts_threshold",
            lambda *_args: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        scored = [(0.9, True), (0.8, True), (0.2, False), (0.3, False)]
        best = OnlineCalibrator._sweep_threshold(scored)
        assert best is not None

    def test_python_confusion_path_without_rust(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With the Rust kernel disabled the sweep uses the pure-Python confusion
        # counts directly (no try/except), still finding the separating threshold.
        """Find a separating threshold directly through Python with native kernels disabled.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Disable native calibration kernels for the threshold sweep.
        """
        monkeypatch.setattr(calibrator_mod, "_RUST_ONLINE_CALIBRATOR", False)
        scored = [(0.9, True), (0.8, True), (0.2, False), (0.3, False)]
        best = OnlineCalibrator._sweep_threshold(scored)
        assert best is not None
        assert 0.3 < best <= 0.8

    def test_rust_wilson_kernel_is_used_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Use one native Wilson call and derive half-width from its returned endpoints.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Enable native intervals and install fixed call-counted endpoints.
        """
        monkeypatch.setattr(calibrator_mod, "_RUST_ONLINE_CALIBRATOR", True)
        called = {"count": 0}

        def _wilson(_p_hat: float, _n: int, _confidence: float) -> tuple[float, float]:
            """Count the native call and return the fixed endpoints used by the dispatch test.

            Parameters
            ----------
            _p_hat : float
                Observed success proportion passed by the interval helper.
            _n : int
                Observation count passed by the interval helper.
            _confidence : float
                Confidence level passed by the interval helper.

            Returns
            -------
            tuple of float
                Fixed lower and upper endpoints, 0.2 and 0.8.
            """
            called["count"] += 1
            return (0.2, 0.8)

        monkeypatch.setattr(
            calibrator_mod,
            "rust_wilson_score_interval",
            _wilson,
            raising=True,
        )
        spread = calibrator_mod._wilson_ci(20, 40)
        assert spread == pytest.approx(0.3)
        assert called["count"] == 1

    def test_rust_wilson_type_error_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Return a bounded Wilson half-width after a native signature TypeError.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Enable native intervals and inject a native signature error.
        """
        monkeypatch.setattr(calibrator_mod, "_RUST_ONLINE_CALIBRATOR", True)
        monkeypatch.setattr(
            calibrator_mod,
            "rust_wilson_score_interval",
            lambda _p_hat, _n, _confidence: (_ for _ in ()).throw(
                TypeError("ffi signature mismatch")
            ),
            raising=True,
        )
        spread = calibrator_mod._wilson_ci(20, 40)
        assert 0.0 <= spread <= 0.5

    def test_sum_int_uses_python_when_rust_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sum integer flags through Python when native calibration kernels are disabled.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Disable native calibration kernels for integer summation.
        """
        monkeypatch.setattr(calibrator_mod, "_RUST_ONLINE_CALIBRATOR", False)

        assert calibrator_mod._sum_int([1, 0, 1]) == 2

    def test_empty_store(self, tmp_path: Path) -> None:
        """Report zero corrections and no recommended threshold for an empty feedback store.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        store = FeedbackStore(tmp_path / "test.db")
        cal = OnlineCalibrator(store)
        report = cal.calibrate()
        assert report.correction_count == 0
        assert report.optimal_threshold is None
        store.close()

    def test_perfect_guardrail(self, tmp_path: Path) -> None:
        """Report perfect accuracy and zero error rates for thirty approved predictions.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(30):
            store.report(f"q{i}", f"a{i}", True, True, guardrail_score=0.8)
        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.correction_count == 30
        assert report.current_accuracy == 1.0
        assert report.fpr == 0.0
        assert report.fnr == 0.0
        assert report.tpr == 1.0
        store.close()

    def test_all_false_positives(self, tmp_path: Path) -> None:
        """Report unit false-positive rate and zero true-positive rate for rejected predictions.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(30):
            store.report(f"q{i}", f"a{i}", True, False, guardrail_score=0.6)
        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.fpr == 1.0
        assert report.tpr == 0.0
        store.close()

    def test_mixed_errors(self, tmp_path: Path) -> None:
        """Report both nonzero error rates and confidence widths for mixed confusion outcomes.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        store = FeedbackStore(tmp_path / "test.db")
        # 10 true positives
        for i in range(10):
            store.report(f"tp{i}", f"atp{i}", True, True, guardrail_score=0.8)
        # 5 false positives
        for i in range(5):
            store.report(f"fp{i}", f"afp{i}", True, False, guardrail_score=0.55)
        # 10 true negatives
        for i in range(10):
            store.report(f"tn{i}", f"atn{i}", False, False, guardrail_score=0.3)
        # 5 false negatives
        for i in range(5):
            store.report(f"fn{i}", f"afn{i}", False, True, guardrail_score=0.45)

        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.correction_count == 30
        assert 0.6 < report.current_accuracy < 0.7
        assert report.fpr > 0
        assert report.fnr > 0
        assert report.fpr_ci > 0
        assert report.fnr_ci > 0
        store.close()


class TestThresholdSweep:
    """Check threshold recommendations against label support, sample count and zero scores."""

    def test_threshold_sweep_returns_none_for_one_class_samples(self) -> None:
        """Withhold a separating threshold when every label belongs to the same class."""
        assert OnlineCalibrator._sweep_threshold([(0.8, True), (0.9, True)]) is None
        assert OnlineCalibrator._sweep_threshold([(0.1, False), (0.2, False)]) is None

    def test_optimal_threshold_separable(self, tmp_path: Path) -> None:
        """Recommend a threshold between separated good and bad score populations.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        store = FeedbackStore(tmp_path / "test.db")
        # Good responses score high, bad responses score low
        for i in range(25):
            store.report(
                f"good{i}", f"a{i}", True, True, guardrail_score=0.8 + i * 0.005
            )
        for i in range(25):
            store.report(
                f"bad{i}", f"b{i}", False, False, guardrail_score=0.2 + i * 0.005
            )

        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.optimal_threshold is not None
        assert 0.3 < report.optimal_threshold < 0.8
        store.close()

    def test_insufficient_corrections(self, tmp_path: Path) -> None:
        """Withhold a threshold recommendation before reaching the required sample count.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(5):
            store.report(f"q{i}", f"a{i}", True, True, guardrail_score=0.8)
        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.optimal_threshold is None
        store.close()

    def test_zero_scores_are_not_used_for_threshold_sweep(self, tmp_path: Path) -> None:
        """Count zero-score corrections without using them to recommend a threshold.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        store = FeedbackStore(tmp_path / "zero_scores.db")
        for i in range(20):
            store.report(f"good{i}", f"a{i}", True, True, guardrail_score=0.0)
            store.report(f"bad{i}", f"b{i}", False, False, guardrail_score=0.0)

        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()

        assert report.correction_count == 40
        assert report.current_accuracy == 1.0
        assert report.optimal_threshold is None
        store.close()


class TestDomainFilter:
    """Check independent calibration reports for differently labelled domain populations."""

    def test_calibrate_by_domain(self, tmp_path: Path) -> None:
        """Keep medical accuracy and finance false-positive summaries separate.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(25):
            store.report(
                f"med{i}", f"a{i}", True, True, guardrail_score=0.8, domain="medical"
            )
        for i in range(25):
            store.report(
                f"fin{i}", f"b{i}", True, False, guardrail_score=0.6, domain="finance"
            )

        cal = OnlineCalibrator(store, min_corrections=20)
        med_report = cal.calibrate(domain="medical")
        fin_report = cal.calibrate(domain="finance")
        assert med_report.current_accuracy == 1.0
        assert fin_report.fpr == 1.0
        store.close()


class TestCalibratorParametrised:
    """Check sample-count boundaries and domain-specific calibration across parameters."""

    @pytest.mark.parametrize("min_corrections", [5, 10, 20, 50])
    def test_min_corrections_threshold(
        self, tmp_path: Path, min_corrections: int
    ) -> None:
        """Withhold a recommendation with one fewer correction than the configured minimum.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        min_corrections : int
            Required sample count tested just below its boundary.
        """
        store = FeedbackStore(tmp_path / f"test_{min_corrections}.db")
        for i in range(min_corrections - 1):
            store.report(f"q{i}", f"a{i}", True, True, guardrail_score=0.8)
        cal = OnlineCalibrator(store, min_corrections=min_corrections)
        report = cal.calibrate()
        assert report.optimal_threshold is None  # not enough
        store.close()

    @pytest.mark.parametrize("domain", ["medical", "finance", "legal"])
    def test_domain_isolation(self, tmp_path: Path, domain: str) -> None:
        """Report perfect accuracy for the requested domain after storing its corrections.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        domain : str
            Domain used for both stored corrections and the requested report.
        """
        store = FeedbackStore(tmp_path / f"test_{domain}.db")
        for i in range(25):
            store.report(
                f"q{i}", f"a{i}", True, True, guardrail_score=0.8, domain=domain
            )
        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate(domain=domain)
        assert report.current_accuracy == 1.0
        store.close()


class TestCalibratorPerformanceDoc:
    """Check existing calibration latency bound and report-field availability."""

    def test_calibrate_fast(self, tmp_path: Path) -> None:
        """Calibrate one hundred stored corrections within the existing one-second bound.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        import time

        store = FeedbackStore(tmp_path / "perf.db")
        for i in range(100):
            store.report(f"q{i}", f"a{i}", True, True, guardrail_score=0.8)

        cal = OnlineCalibrator(store, min_corrections=20)
        t0 = time.perf_counter()
        cal.calibrate()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 1000, f"Calibration took {elapsed_ms:.0f}ms (expected <1s)"
        store.close()

    def test_report_has_all_fields(self, tmp_path: Path) -> None:
        """Expose correction, accuracy, error-rate and threshold fields on the report.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Isolated directory for the real SQLite feedback-store fixture.
        """
        store = FeedbackStore(tmp_path / "fields.db")
        for i in range(30):
            store.report(f"q{i}", f"a{i}", True, True, guardrail_score=0.8)
        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert hasattr(report, "correction_count")
        assert hasattr(report, "current_accuracy")
        assert hasattr(report, "fpr")
        assert hasattr(report, "fnr")
        assert hasattr(report, "optimal_threshold")
        store.close()


def test_calibrate_returns_the_calibration_report_contract(tmp_path: Path) -> None:
    """Return the public report type with bounded rates and nonnegative confidence widths.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the real SQLite feedback-store fixture.
    """
    from director_ai.core.calibration.online_calibrator import CalibrationReport

    store = FeedbackStore(tmp_path / "contract.db")
    try:
        report = OnlineCalibrator(store).calibrate()
    finally:
        store.close()

    assert isinstance(report, CalibrationReport)
    assert report.correction_count == 0
    assert report.optimal_threshold is None
    for rate in (report.tpr, report.tnr, report.fpr, report.fnr):
        assert 0.0 <= rate <= 1.0
    assert report.fpr_ci >= 0.0
    assert report.fnr_ci >= 0.0
