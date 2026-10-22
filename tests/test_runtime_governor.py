# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — runtime threshold governor tests

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import pytest

from director_ai.core.calibration.adaptive_threshold import (
    AdaptiveThresholdRecommendation,
    AdaptiveThresholdReport,
)
from director_ai.core.calibration.conformal import PredictionInterval
from director_ai.core.calibration.runtime_governor import (
    EffectiveThreshold,
    RuntimeThresholdGovernor,
    ThresholdChange,
)
from director_ai.core.calibration.segmented_threshold import (
    SegmentedThresholdLearner,
    SegmentRecommendation,
)
from director_ai.core.routing.uncertainty_router import UncertaintyRouter


def _recommendation(
    *,
    current: float = 0.5,
    recommended: float | None = 0.7,
    requires_human_approval: bool = False,
) -> AdaptiveThresholdRecommendation:
    return AdaptiveThresholdRecommendation(
        current_threshold=current,
        recommended_threshold=recommended,
        expected_success_probability=0.9,
        current_success_probability=0.8,
        expected_lift=0.1,
        reason="replayed evidence favours a higher threshold",
        requires_human_approval=requires_human_approval,
        rollback_threshold=current,
    )


class _StubLearner(SegmentedThresholdLearner):
    """Controllable stand-in for SegmentedThresholdLearner."""

    def __init__(self, recommendation: SegmentRecommendation) -> None:
        super().__init__(candidate_thresholds=[0.5, 0.7, 0.9], current_threshold=0.5)
        self._rec = recommendation
        self.observed: list[tuple[float, bool, str]] = []

    def observe(
        self, score: float, human_approved: bool, *, segment: str
    ) -> AdaptiveThresholdReport:
        self.observed.append((score, human_approved, segment))
        return super().observe(score, human_approved, segment=segment)

    def recommend(self, *, segment: str) -> SegmentRecommendation:
        return self._rec


def _seg_rec(
    source: str = "segment",
    recommended: float | None = 0.7,
    requires_human_approval: bool = False,
    count: int = 40,
) -> SegmentRecommendation:
    return SegmentRecommendation(
        segment="medical",
        source=source,
        feedback_count=count,
        recommendation=_recommendation(
            recommended=recommended, requires_human_approval=requires_human_approval
        ),
    )


class _GovernorOptions(TypedDict, total=False):
    """Native governor constructor fields varied by validation cases."""

    current_threshold: float
    max_step: float
    uncertainty_penalty: float


def _governor(
    rec: SegmentRecommendation,
    *,
    current_threshold: float = 0.5,
    max_step: float = 0.05,
    auto_apply: bool = False,
    uncertainty_router: UncertaintyRouter | None = None,
    uncertainty_penalty: float = 0.1,
    clock: Callable[[], float] = lambda: 1.0,
) -> RuntimeThresholdGovernor:
    """Construct the real governor around a valid controllable learner."""
    return RuntimeThresholdGovernor(
        learner=_StubLearner(rec),
        current_threshold=current_threshold,
        max_step=max_step,
        auto_apply=auto_apply,
        uncertainty_router=uncertainty_router,
        uncertainty_penalty=uncertainty_penalty,
        clock=clock,
    )


def _interval(
    *, lower: float = 0.0, upper: float = 1.0, reliable: bool = False
) -> PredictionInterval:
    """Supply real calibrated-risk values to the native uncertainty router."""
    return PredictionInterval(
        point_estimate=(lower + upper) / 2,
        lower=lower,
        upper=upper,
        coverage=0.95,
        calibration_size=40,
        is_reliable=reliable,
    )


# ── construction validation ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_threshold": 1.5},
        {"max_step": 0.0},
        {"max_step": 1.5},
        {"uncertainty_penalty": -0.1},
        {"uncertainty_penalty": 1.1},
    ],
)
def test_invalid_construction_raises(kwargs: _GovernorOptions) -> None:
    with pytest.raises(ValueError):
        _governor(_seg_rec(), **kwargs)


def test_live_threshold_defaults_to_current() -> None:
    gov = _governor(_seg_rec())
    assert gov.live_threshold("medical") == 0.5


# ── propose: gating ──────────────────────────────────────────────────────────


def test_propose_blocks_on_global_source() -> None:
    gov = _governor(_seg_rec(source="global"))
    change = gov.propose("medical")
    assert isinstance(change, ThresholdChange)
    assert change.applied is False
    assert change.reason == "insufficient segment evidence"
    assert gov.live_threshold("medical") == 0.5


def test_propose_no_change_when_recommended_none() -> None:
    gov = _governor(_seg_rec(recommended=None))
    change = gov.propose("medical")
    assert change.applied is False
    assert change.reason == "no change recommended"


def test_propose_holds_for_human_approval() -> None:
    gov = _governor(_seg_rec(requires_human_approval=True))
    change = gov.propose("medical")
    assert change.applied is False
    assert change.requires_approval is True
    assert change.reason == "pending human approval"
    assert gov.live_threshold("medical") == 0.5


def test_propose_applies_when_no_approval_needed() -> None:
    gov = _governor(_seg_rec(recommended=0.7), max_step=0.05)
    change = gov.propose("medical")
    assert change.applied is True
    # stepped from 0.5 toward 0.7 by at most max_step
    assert change.to_threshold == pytest.approx(0.55)
    assert gov.live_threshold("medical") == pytest.approx(0.55)


def test_auto_apply_overrides_human_approval() -> None:
    gov = _governor(_seg_rec(requires_human_approval=True), auto_apply=True)
    change = gov.propose("medical")
    assert change.applied is True


# ── bounded stepping ─────────────────────────────────────────────────────────


def test_step_is_bounded_by_max_step() -> None:
    gov = _governor(_seg_rec(recommended=0.95), max_step=0.05)
    change = gov.propose("medical")
    assert change.to_threshold == pytest.approx(0.55)  # not 0.95


def test_step_reaches_target_when_within_max_step() -> None:
    gov = _governor(_seg_rec(recommended=0.52), max_step=0.05)
    change = gov.propose("medical")
    assert change.to_threshold == pytest.approx(0.52)


def test_step_can_move_downward() -> None:
    gov = _governor(_seg_rec(recommended=0.2), max_step=0.05)
    change = gov.propose("medical")
    assert change.to_threshold == pytest.approx(0.45)


# ── apply with approval ──────────────────────────────────────────────────────


def test_apply_with_approval_applies_human_gated_change() -> None:
    gov = _governor(_seg_rec(requires_human_approval=True))
    change = gov.apply("medical", approve=True)
    assert change.applied is True
    assert gov.live_threshold("medical") == pytest.approx(0.55)


def test_apply_without_approval_blocks() -> None:
    gov = _governor(_seg_rec(requires_human_approval=True))
    change = gov.apply("medical", approve=False)
    assert change.applied is False
    assert change.reason == "approval required"


def test_apply_nothing_to_apply_on_global_source() -> None:
    gov = _governor(_seg_rec(source="global"))
    change = gov.apply("medical", approve=True)
    assert change.applied is False
    assert change.reason == "nothing to apply"


def test_apply_no_approval_needed() -> None:
    gov = _governor(_seg_rec(requires_human_approval=False))
    change = gov.apply("medical")
    assert change.applied is True


# ── conformal / uncertainty tie-in ───────────────────────────────────────────


def test_effective_threshold_without_interval() -> None:
    gov = _governor(_seg_rec())
    eff = gov.effective_threshold("medical")
    assert isinstance(eff, EffectiveThreshold)
    assert eff.threshold == 0.5
    assert eff.uncertainty_adjusted is False
    assert eff.action is None


def test_effective_threshold_tightens_on_uncertainty() -> None:
    gov = _governor(
        _seg_rec(),
        uncertainty_router=UncertaintyRouter(),
        uncertainty_penalty=0.1,
    )
    eff = gov.effective_threshold("medical", interval=_interval())
    assert eff.uncertainty_adjusted is True
    assert eff.threshold == pytest.approx(0.4)  # 0.5 - 0.1
    assert eff.action == "escalate_human"


def test_effective_threshold_allows_when_router_confident() -> None:
    gov = _governor(_seg_rec(), uncertainty_router=UncertaintyRouter())
    eff = gov.effective_threshold(
        "medical", interval=_interval(lower=0.0, upper=0.1, reliable=True)
    )
    assert eff.uncertainty_adjusted is False
    assert eff.threshold == 0.5
    assert eff.action == "allow"


def test_effective_threshold_tighten_clamps_at_zero() -> None:
    gov = _governor(
        _seg_rec(),
        current_threshold=0.05,
        uncertainty_router=UncertaintyRouter(),
        uncertainty_penalty=0.1,
    )
    eff = gov.effective_threshold(
        "medical", interval=_interval(lower=0.3, upper=0.7, reliable=True)
    )
    assert eff.threshold == 0.0  # clamped, not negative


# ── audit history + observe ──────────────────────────────────────────────────


def test_history_records_only_applied_changes() -> None:
    gov = _governor(_seg_rec(recommended=0.7))
    gov.propose("medical")  # applied
    assert len(gov.history()) == 1
    assert gov.history("medical")[0].applied is True
    assert gov.history("other") == ()


def test_observe_forwards_to_learner() -> None:
    learner = _StubLearner(_seg_rec())
    gov = RuntimeThresholdGovernor(
        learner=learner, current_threshold=0.5, clock=lambda: 1.0
    )
    gov.observe(segment="medical", score=0.7, human_approved=True)
    assert learner.observed == [(0.7, True, "medical")]


def test_change_timestamp_from_clock() -> None:
    gov = _governor(_seg_rec(recommended=0.52), clock=lambda: 42.0)
    change = gov.propose("medical")
    assert change.at == 42.0


# ── ProductionGuard wiring ──────────────────────────────────────────────────


def test_guard_builds_threshold_governor() -> None:
    from director_ai.core.config import DirectorConfig
    from director_ai.guard import ProductionGuard

    guard = ProductionGuard(
        config=DirectorConfig(use_nli=False, coherence_threshold=0.6)
    )
    gov = guard.new_threshold_governor()
    assert isinstance(gov, RuntimeThresholdGovernor)
    assert gov.live_threshold("any-segment") == 0.6  # seeded from the guard threshold
    # fresh instances keep isolated state
    assert guard.new_threshold_governor() is not gov
