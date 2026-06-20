# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — meta guard tests

"""Multi-angle coverage: ScoringDecision validation, DecisionLog
append + FIFO + windowed views + thread-safety, MetaAnalyzer
Page-Hinkley change-point detection, Brier calibration drift,
action-rate divergence, ThresholdAdjuster hysteresis, centre
squeeze on Brier alarm, and the MetaGuard orchestrator
end-to-end."""

from __future__ import annotations

import threading
from typing import Any, cast

import pytest

import director_ai.core.meta_guard.analyzer as analyzer_mod
import director_ai.core.meta_guard.guard as guard_mod
from director_ai.core.meta_guard import (
    DecisionLog,
    MetaAnalyzer,
    MetaGuard,
    MetaGuardProductionPolicy,
    MetaVerdict,
    ProductionMetaGuardDecision,
    ScoringDecision,
    ThresholdAdjuster,
    ThresholdBundle,
)

# --- ScoringDecision ------------------------------------------------


class TestScoringDecision:
    def test_valid(self):
        d = ScoringDecision(prompt_hash="h", score=0.5, action="allow", timestamp=0.0)
        assert d.ground_truth is None

    def test_empty_hash_rejected(self):
        with pytest.raises(ValueError, match="prompt_hash"):
            ScoringDecision(prompt_hash="", score=0.5, action="allow", timestamp=0.0)

    def test_score_out_of_range(self):
        with pytest.raises(ValueError, match="score"):
            ScoringDecision(prompt_hash="h", score=1.5, action="allow", timestamp=0.0)

    def test_bad_action(self):
        bad = cast(Any, "ignore")
        with pytest.raises(ValueError, match="action"):
            ScoringDecision(prompt_hash="h", score=0.5, action=bad, timestamp=0.0)

    def test_ground_truth_range(self):
        with pytest.raises(ValueError, match="ground_truth"):
            ScoringDecision(
                prompt_hash="h",
                score=0.5,
                action="allow",
                ground_truth=1.2,
                timestamp=0.0,
            )


# --- DecisionLog ----------------------------------------------------


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class TestDecisionLog:
    def test_append_and_len(self):
        log = DecisionLog(clock=_FakeClock())
        log.append(
            ScoringDecision(prompt_hash="h", score=0.5, action="allow", timestamp=0.0)
        )
        assert len(log) == 1

    def test_record_hashes_by_default(self):
        clock = _FakeClock()
        log = DecisionLog(clock=clock)
        decision = log.record(prompt="hello", score=0.5, action="allow")
        # SHA-256 hex prefix is 16 chars.
        assert len(decision.prompt_hash) == 16
        assert decision.prompt_hash != "hello"

    def test_record_hash_disabled(self):
        log = DecisionLog(hash_prompts=False, clock=_FakeClock())
        decision = log.record(prompt="hello", score=0.5, action="allow")
        assert decision.prompt_hash == "hello"

    def test_custom_hasher(self):
        log = DecisionLog(hasher=lambda p: f"custom:{p}", clock=_FakeClock())
        decision = log.record(prompt="x", score=0.5, action="allow")
        assert decision.prompt_hash == "custom:x"

    def test_empty_prompt_rejected(self):
        log = DecisionLog(clock=_FakeClock())
        with pytest.raises(ValueError, match="prompt"):
            log.record(prompt="", score=0.5, action="allow")

    def test_capacity_eviction(self):
        log = DecisionLog(capacity=3, clock=_FakeClock())
        for i in range(5):
            log.append(
                ScoringDecision(
                    prompt_hash=f"h{i}", score=0.1, action="allow", timestamp=float(i)
                )
            )
        assert len(log) == 3
        snap = log.snapshot()
        assert [d.prompt_hash for d in snap] == ["h2", "h3", "h4"]

    def test_bad_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            DecisionLog(capacity=0)

    def test_window_last_n(self):
        clock = _FakeClock()
        log = DecisionLog(clock=clock)
        for i in range(10):
            clock.now = float(i)
            log.record(prompt=f"p{i}", score=0.1, action="allow")
        window = log.window(last_n=3)
        assert len(window) == 3
        assert [d.timestamp for d in window] == [7.0, 8.0, 9.0]

    def test_window_since_seconds(self):
        clock = _FakeClock(start=1_000.0)
        log = DecisionLog(clock=clock)
        for dt in (50.0, 30.0, 10.0, 0.0):
            clock.now = 1_000.0 - dt
            log.record(prompt=f"p{dt}", score=0.1, action="allow")
        clock.now = 1_000.0
        window = log.window(since_seconds=20.0)
        # Anything older than 20s is excluded.
        assert all(d.timestamp >= 980.0 for d in window)

    def test_window_needs_exactly_one_param(self):
        log = DecisionLog(clock=_FakeClock())
        with pytest.raises(ValueError, match="exactly one"):
            log.window()
        with pytest.raises(ValueError, match="exactly one"):
            log.window(last_n=1, since_seconds=1.0)

    def test_window_negative_values_rejected(self):
        log = DecisionLog(clock=_FakeClock())
        with pytest.raises(ValueError, match="last_n"):
            log.window(last_n=0)
        with pytest.raises(ValueError, match="since_seconds"):
            log.window(since_seconds=-1.0)

    def test_iter_windowed(self):
        log = DecisionLog(clock=_FakeClock())
        for _ in range(3):
            log.record(prompt="p", score=0.1, action="allow")
        items = list(log.iter_windowed(last_n=2))
        assert len(items) == 2

    def test_concurrent_appends(self):
        log = DecisionLog(capacity=5_000, clock=_FakeClock())

        def writer(tag: str) -> None:
            for i in range(100):
                log.append(
                    ScoringDecision(
                        prompt_hash=f"{tag}-{i}",
                        score=0.5,
                        action="allow",
                        timestamp=float(i),
                    )
                )

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(log) == 800


# --- MetaAnalyzer ---------------------------------------------------


def _steady_window(n: int, score: float = 0.3) -> list[ScoringDecision]:
    return [
        ScoringDecision(
            prompt_hash=f"h{i}", score=score, action="allow", timestamp=float(i)
        )
        for i in range(n)
    ]


def _drift_window(
    n: int, pre: float = 0.2, post: float = 0.8, switch: int = 20
) -> list[ScoringDecision]:
    return [
        ScoringDecision(
            prompt_hash=f"h{i}",
            score=(pre if i < switch else post),
            action=("allow" if i < switch else "halt"),
            timestamp=float(i),
        )
        for i in range(n)
    ]


class TestMetaAnalyzer:
    def test_empty_window(self):
        analyzer = MetaAnalyzer(reference_mean=0.3, min_window=4)
        analysis = analyzer.analyse([])
        assert analysis.window_size == 0
        assert not analysis.any_alarm

    def test_steady_window_no_alarm(self):
        analyzer = MetaAnalyzer(reference_mean=0.3, min_window=8)
        analysis = analyzer.analyse(_steady_window(64))
        assert not analysis.page_hinkley_alarm
        assert analysis.mean_score == pytest.approx(0.3)

    def test_upward_drift_triggers_page_hinkley(self):
        analyzer = MetaAnalyzer(reference_mean=0.2, ph_threshold=0.5, min_window=8)
        analysis = analyzer.analyse(_drift_window(64))
        assert analysis.page_hinkley_alarm
        assert analysis.page_hinkley_statistic > 0.5

    def test_downward_drift_triggers_page_hinkley(self):
        analyzer = MetaAnalyzer(reference_mean=0.8, ph_threshold=0.5, min_window=8)
        analysis = analyzer.analyse(_drift_window(64, pre=0.9, post=0.2, switch=32))
        assert analysis.page_hinkley_alarm

    def test_brier_alarm_fires_under_miscalibration(self):
        analyzer = MetaAnalyzer(
            reference_mean=0.5,
            ph_threshold=1e9,  # never fires
            reference_brier=0.05,
            brier_tolerance=0.1,
            min_window=16,
        )
        # Scores say 0.9 but truth is 0.0 — huge calibration error.
        window = [
            ScoringDecision(
                prompt_hash=f"h{i}",
                score=0.9,
                action="halt",
                ground_truth=0.0,
                timestamp=float(i),
            )
            for i in range(32)
        ]
        analysis = analyzer.analyse(window)
        assert analysis.brier_alarm
        assert analysis.brier_score is not None
        assert analysis.brier_score > 0.5

    def test_brier_channel_disabled(self):
        analyzer = MetaAnalyzer(reference_mean=0.5, reference_brier=None)
        analysis = analyzer.analyse(_steady_window(32))
        assert analysis.brier_score is None

    def test_brier_channel_waits_for_labelled_min_window(self):
        analyzer = MetaAnalyzer(reference_mean=0.5, reference_brier=0.05, min_window=8)
        window = [
            ScoringDecision(
                prompt_hash=f"h{i}",
                score=0.5,
                action="allow",
                ground_truth=(1.0 if i == 0 else None),
                timestamp=float(i),
            )
            for i in range(8)
        ]

        analysis = analyzer.analyse(window)

        assert analysis.brier_score is None
        assert analysis.brier_alarm is False

    def test_action_rate_divergence(self):
        analyzer = MetaAnalyzer(
            reference_mean=0.5,
            ph_threshold=1e9,
            reference_action_rates={"allow": 0.9, "warn": 0.05, "halt": 0.05},
            action_tolerance=0.1,
            min_window=8,
        )
        # Observed: all halt. Rate divergence from reference is large.
        window = [
            ScoringDecision(
                prompt_hash=f"h{i}", score=0.9, action="halt", timestamp=float(i)
            )
            for i in range(32)
        ]
        analysis = analyzer.analyse(window)
        assert analysis.action_alarm
        assert analysis.action_divergence > 0.1

    def test_action_channel_disabled(self):
        analyzer = MetaAnalyzer(reference_mean=0.5, reference_action_rates=None)
        analysis = analyzer.analyse(_steady_window(16))
        assert not analysis.action_alarm
        assert analysis.action_divergence == 0.0

    def test_zero_reference_action_rate_with_observed_action_is_infinite(self):
        analyzer = MetaAnalyzer(
            reference_mean=0.5,
            reference_action_rates={"allow": 1.0, "warn": 0.0, "halt": 0.0},
            action_tolerance=0.1,
            min_window=2,
        )
        window = [
            ScoringDecision(
                prompt_hash="h0",
                score=0.5,
                action="warn",
                timestamp=0.0,
            ),
            ScoringDecision(
                prompt_hash="h1",
                score=0.5,
                action="allow",
                timestamp=1.0,
            ),
        ]

        analysis = analyzer.analyse(window)

        assert analysis.action_divergence == float("inf")
        assert analysis.action_alarm is True

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"reference_mean": 1.5}, "reference_mean"),
            ({"reference_mean": 0.5, "ph_delta": -0.1}, "ph_delta"),
            ({"reference_mean": 0.5, "ph_threshold": 0}, "ph_threshold"),
            ({"reference_mean": 0.5, "reference_brier": 2.0}, "reference_brier"),
            ({"reference_mean": 0.5, "brier_tolerance": -0.1}, "brier_tolerance"),
            ({"reference_mean": 0.5, "action_tolerance": -0.1}, "action_tolerance"),
            ({"reference_mean": 0.5, "min_window": 0}, "min_window"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        with pytest.raises(ValueError, match=match):
            MetaAnalyzer(**kwargs)

    def test_action_rates_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            MetaAnalyzer(
                reference_mean=0.5,
                reference_action_rates={"allow": 0.5, "warn": 0.0, "halt": 0.0},
            )

    def test_unknown_action_rejected(self):
        bad = cast(Any, {"allow": 0.5, "block": 0.5})
        with pytest.raises(ValueError, match="unknown action"):
            MetaAnalyzer(reference_mean=0.5, reference_action_rates=bad)

    def test_negative_action_rate_rejected(self):
        bad = cast(Any, {"allow": 1.1, "warn": -0.1, "halt": 0.0})
        with pytest.raises(ValueError, match="action rate"):
            MetaAnalyzer(reference_mean=0.5, reference_action_rates=bad)


class TestMetaAnalyzerRustKernels:
    def test_empty_sum_and_mean_do_not_call_rust(self, monkeypatch):
        monkeypatch.setattr(analyzer_mod, "_RUST_META_GUARD", True)

        assert analyzer_mod._sum_float([]) == 0.0
        assert analyzer_mod._mean_float([]) == 0.0

    def test_rust_mean_kernel_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(analyzer_mod, "_RUST_META_GUARD", True)
        called = {"mean": 0}

        def _mean(values: list[float]) -> float:
            called["mean"] += 1
            return sum(values) / len(values)

        monkeypatch.setattr(analyzer_mod, "rust_mean", _mean, raising=True)
        analyzer = MetaAnalyzer(reference_mean=0.3, min_window=4)
        analysis = analyzer.analyse(_steady_window(8, score=0.4))
        assert analysis.mean_score == pytest.approx(0.4)
        assert called["mean"] >= 1

    def test_rust_mean_type_error_falls_back(self, monkeypatch):
        monkeypatch.setattr(analyzer_mod, "_RUST_META_GUARD", True)
        monkeypatch.setattr(
            analyzer_mod,
            "rust_mean",
            lambda _values: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        analyzer = MetaAnalyzer(reference_mean=0.3, min_window=4)
        analysis = analyzer.analyse(_steady_window(8, score=0.4))
        assert analysis.mean_score == pytest.approx(0.4)

    def test_sum_and_mean_use_python_when_rust_disabled(self, monkeypatch):
        monkeypatch.setattr(analyzer_mod, "_RUST_META_GUARD", False)

        assert analyzer_mod._sum_float([0.2, 0.3]) == pytest.approx(0.5)
        assert analyzer_mod._mean_float([0.2, 0.4]) == pytest.approx(0.3)

    def test_rust_sum_kernel_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(analyzer_mod, "_RUST_META_GUARD", True)
        calls = {"count": 0}

        def _sum(values: list[float]) -> float:
            calls["count"] += 1
            return sum(values)

        monkeypatch.setattr(analyzer_mod, "rust_sum_f64", _sum, raising=True)

        assert analyzer_mod._sum_float([0.1, 0.2]) == pytest.approx(0.3)
        assert calls["count"] == 1


# --- ThresholdAdjuster ---------------------------------------------


class TestThresholdAdjuster:
    def _initial(self) -> ThresholdBundle:
        return ThresholdBundle(warn_threshold=0.3, halt_threshold=0.7)

    def test_bad_initial(self):
        with pytest.raises(ValueError, match="thresholds"):
            ThresholdBundle(warn_threshold=0.7, halt_threshold=0.3)

    def test_quiet_analysis_holds_thresholds(self):
        adj = ThresholdAdjuster(initial=self._initial())
        analysis = _mock_analysis(ph_alarm=False)
        assert adj.observe(analysis) is None

    def test_hysteresis_prevents_single_spike(self):
        adj = ThresholdAdjuster(initial=self._initial(), hysteresis_strikes=2)
        analysis = _mock_analysis(ph_alarm=True, mean_score=0.5)
        first = adj.observe(analysis)
        assert first is None

    def test_consistent_drift_triggers_after_strikes(self):
        adj = ThresholdAdjuster(initial=self._initial(), hysteresis_strikes=2)
        tight_analysis = _mock_analysis(ph_alarm=True, mean_score=0.5)
        adj.observe(tight_analysis)  # strike 1
        result = adj.observe(tight_analysis)  # strike 2 → move
        assert result is not None
        assert result.warn_threshold > 0.3

    def test_direction_reset_on_flip(self):
        adj = ThresholdAdjuster(initial=self._initial(), hysteresis_strikes=2)
        tight = _mock_analysis(ph_alarm=True, mean_score=0.5)
        loose = _mock_analysis(ph_alarm=True, mean_score=0.01)
        adj.observe(tight)  # strike in tighten direction
        result = adj.observe(loose)  # flip — resets the counter
        assert result is None

    def test_brier_triggers_centre_squeeze(self):
        adj = ThresholdAdjuster(
            initial=self._initial(), hysteresis_strikes=1, max_step=0.1
        )
        analysis = _mock_analysis(brier_alarm=True)
        result = adj.observe(analysis)
        assert result is not None
        # Both thresholds move toward the centre.
        assert result.warn_threshold > 0.3
        assert result.halt_threshold < 0.7

    def test_floor_ceiling_respected(self):
        adj = ThresholdAdjuster(
            initial=ThresholdBundle(warn_threshold=0.1, halt_threshold=0.95),
            hysteresis_strikes=1,
            max_step=0.5,
            floor_warn=0.05,
            ceiling_halt=0.95,
        )
        tight = _mock_analysis(ph_alarm=True, mean_score=0.5)
        result = adj.observe(tight)
        assert result is not None
        assert result.halt_threshold <= 0.95

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"max_step": 0}, "max_step"),
            ({"hysteresis_strikes": 0}, "hysteresis_strikes"),
            ({"floor_warn": 0.9, "ceiling_halt": 0.5}, "floor_warn"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        with pytest.raises(ValueError, match=match):
            ThresholdAdjuster(initial=self._initial(), **kwargs)


def _mock_analysis(
    *,
    ph_alarm: bool = False,
    mean_score: float = 0.3,
    brier_alarm: bool = False,
    action_alarm: bool = False,
):
    from director_ai.core.meta_guard.analyzer import MetaAnalysis

    return MetaAnalysis(
        window_size=64,
        mean_score=mean_score,
        page_hinkley_statistic=1.0 if ph_alarm else 0.0,
        page_hinkley_alarm=ph_alarm,
        brier_score=0.2 if brier_alarm else 0.05,
        brier_delta=0.15 if brier_alarm else 0.0,
        brier_alarm=brier_alarm,
        action_rates={"allow": 0.8, "warn": 0.1, "halt": 0.1},
        action_divergence=1.0 if action_alarm else 0.0,
        action_alarm=action_alarm,
    )


# --- MetaGuard -----------------------------------------------------


class TestMetaGuard:
    def _guard(
        self,
        *,
        with_adjuster: bool = True,
        reference_mean: float = 0.3,
    ) -> MetaGuard:
        log = DecisionLog(clock=_FakeClock())
        analyzer = MetaAnalyzer(
            reference_mean=reference_mean, ph_threshold=0.1, min_window=8
        )
        adjuster = (
            ThresholdAdjuster(
                initial=ThresholdBundle(warn_threshold=0.3, halt_threshold=0.7),
                hysteresis_strikes=2,
                max_step=0.05,
            )
            if with_adjuster
            else None
        )
        return MetaGuard(log=log, analyzer=analyzer, adjuster=adjuster)

    def test_record_returns_verdict(self):
        guard = self._guard()
        verdict = guard.record(prompt="p", score=0.3, action="allow")
        assert isinstance(verdict, MetaVerdict)
        assert verdict.decision.score == 0.3

    def test_observe_only_mode(self):
        guard = self._guard(with_adjuster=False)
        verdict = guard.record(prompt="p", score=0.3, action="allow")
        assert verdict.thresholds is None
        assert not verdict.adjusted
        assert guard.adjuster is None

    def test_drift_triggers_adjustment(self):
        guard = self._guard(reference_mean=0.1)
        # Prime with 40 high-score decisions — Page-Hinkley goes
        # upward, so the adjuster eventually tightens.
        adjusted_any = False
        for _ in range(40):
            verdict = guard.record(prompt="adversarial", score=0.9, action="halt")
            if verdict.adjusted:
                adjusted_any = True
        assert adjusted_any

    def test_latest_analysis(self):
        guard = self._guard()
        for _ in range(10):
            guard.record(prompt="x", score=0.3, action="allow")
        analysis = guard.latest_analysis()
        assert analysis.window_size == 10

    def test_bad_window(self):
        with pytest.raises(ValueError, match="window_last_n"):
            MetaGuard(
                log=DecisionLog(clock=_FakeClock()),
                analyzer=MetaAnalyzer(reference_mean=0.5),
                window_last_n=0,
            )

    def test_log_access(self):
        guard = self._guard()
        guard.record(prompt="p", score=0.3, action="allow")
        assert len(guard.log) == 1

    def test_production_policy_blocks_single_tenant_recursive_adjustment(self):
        policy = MetaGuardProductionPolicy(
            min_labelled_fraction=0.0,
            max_single_tenant_fraction=0.5,
            max_duplicate_prompt_fraction=1.0,
        )
        guard = MetaGuard(
            log=DecisionLog(clock=_FakeClock()),
            analyzer=MetaAnalyzer(reference_mean=0.1, ph_threshold=0.1, min_window=8),
            adjuster=ThresholdAdjuster(
                initial=ThresholdBundle(warn_threshold=0.3, halt_threshold=0.7),
                hysteresis_strikes=1,
                max_step=0.05,
            ),
            production_policy=policy,
        )

        blocked = False
        for i in range(16):
            verdict = guard.record(
                prompt=f"attack-{i}",
                score=0.9,
                action="halt",
                tenant_id="tenant-a",
            )
            blocked = blocked or verdict.production.blocked

        assert blocked is True
        assert verdict.thresholds is None
        assert guard.adjuster is not None
        assert guard.adjuster.current == ThresholdBundle(
            warn_threshold=0.3,
            halt_threshold=0.7,
        )
        assert verdict.production.block_reason == "single_tenant_dominance"

    def test_production_policy_blocks_duplicate_prompt_evasion(self):
        policy = MetaGuardProductionPolicy(
            min_labelled_fraction=0.0,
            max_single_tenant_fraction=1.0,
            max_duplicate_prompt_fraction=0.4,
        )
        guard = MetaGuard(
            log=DecisionLog(clock=_FakeClock()),
            analyzer=MetaAnalyzer(reference_mean=0.1, ph_threshold=0.1, min_window=8),
            adjuster=ThresholdAdjuster(
                initial=ThresholdBundle(warn_threshold=0.3, halt_threshold=0.7),
                hysteresis_strikes=1,
                max_step=0.05,
            ),
            production_policy=policy,
        )

        for i in range(16):
            tenant_id = f"tenant-{i % 4}"
            verdict = guard.record(
                prompt="same jailbreak probe",
                score=0.9,
                action="halt",
                tenant_id=tenant_id,
            )

        assert verdict.production.blocked is True
        assert verdict.production.block_reason == "duplicate_prompt_dominance"
        assert verdict.production.evasion_score > 0.4
        assert verdict.thresholds is None

    def test_production_policy_allows_empty_alarm_window(self):
        policy = MetaGuardProductionPolicy()

        decision = policy.evaluate(window=[], analysis=_mock_analysis(ph_alarm=True))

        assert decision.enabled is True
        assert decision.blocked is False
        assert decision.window_size == 0

    def test_production_policy_blocks_insufficient_labels(self):
        policy = MetaGuardProductionPolicy(
            min_labelled_fraction=0.75,
            max_single_tenant_fraction=0.5,
            max_duplicate_prompt_fraction=0.5,
        )
        window = [
            ScoringDecision(
                prompt_hash=f"h{i}",
                score=0.9,
                action="halt",
                timestamp=float(i),
                tenant_id=f"tenant-{i % 4}",
                ground_truth=1.0 if i == 0 else None,
            )
            for i in range(8)
        ]

        decision = policy.evaluate(
            window=window,
            analysis=_mock_analysis(ph_alarm=True),
        )

        assert decision.blocked is True
        assert decision.block_reason == "insufficient_labels"
        assert decision.labelled_fraction == pytest.approx(0.125)

    def test_production_policy_allows_labelled_diverse_adjustment(self):
        policy = MetaGuardProductionPolicy(
            min_labelled_fraction=0.75,
            max_single_tenant_fraction=0.5,
            max_duplicate_prompt_fraction=0.5,
        )
        guard = MetaGuard(
            log=DecisionLog(clock=_FakeClock()),
            analyzer=MetaAnalyzer(reference_mean=0.1, ph_threshold=0.1, min_window=8),
            adjuster=ThresholdAdjuster(
                initial=ThresholdBundle(warn_threshold=0.3, halt_threshold=0.7),
                hysteresis_strikes=1,
                max_step=0.05,
            ),
            production_policy=policy,
        )

        adjusted = False
        for i in range(16):
            verdict = guard.record(
                prompt=f"calibrated-{i}",
                score=0.9,
                action="halt",
                ground_truth=1.0,
                tenant_id=f"tenant-{i % 4}",
            )
            adjusted = adjusted or verdict.adjusted

        assert adjusted is True
        assert isinstance(verdict.production, ProductionMetaGuardDecision)
        assert verdict.production.blocked is False
        assert verdict.production.labelled_fraction == pytest.approx(1.0)

    def test_production_policy_rejects_invalid_fraction(self):
        with pytest.raises(ValueError, match="max_single_tenant_fraction"):
            MetaGuardProductionPolicy(max_single_tenant_fraction=1.5)

    def test_window_metrics_empty_window(self):
        assert guard_mod._window_metrics([]) == {
            "window_size": 0,
            "labelled_fraction": 0.0,
            "single_tenant_fraction": 0.0,
            "duplicate_prompt_fraction": 0.0,
            "evasion_score": 0.0,
        }

    def test_sum_int_python_floor_when_rust_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(guard_mod, "_RUST_META_GUARD", False)

        assert guard_mod._sum_int([1, 2, 3]) == 6
