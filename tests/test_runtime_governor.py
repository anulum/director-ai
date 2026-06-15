# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — runtime threshold governor tests

from __future__ import annotations

import pytest

from director_ai.core.calibration.adaptive_threshold import (
    AdaptiveThresholdRecommendation,
)
from director_ai.core.calibration.runtime_governor import (
    EffectiveThreshold,
    RuntimeThresholdGovernor,
    ThresholdChange,
)
from director_ai.core.calibration.segmented_threshold import SegmentRecommendation


def _recommendation(
    *, current=0.5, recommended=0.7, requires_human_approval=False
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


class _StubLearner:
    """Controllable stand-in for SegmentedThresholdLearner."""

    def __init__(self, recommendation: SegmentRecommendation):
        self._rec = recommendation
        self.observed: list[tuple[float, bool, str]] = []

    def observe(self, score: float, human_approved: bool, *, segment: str):
        self.observed.append((score, human_approved, segment))

    def recommend(self, *, segment: str) -> SegmentRecommendation:
        return self._rec


def _seg_rec(
    source="segment", recommended=0.7, requires_human_approval=False, count=40
):
    return SegmentRecommendation(
        segment="medical",
        source=source,
        feedback_count=count,
        recommendation=_recommendation(
            recommended=recommended, requires_human_approval=requires_human_approval
        ),
    )


def _governor(rec: SegmentRecommendation, **kwargs) -> RuntimeThresholdGovernor:
    kwargs.setdefault("current_threshold", 0.5)
    kwargs.setdefault("clock", lambda: 1.0)
    return RuntimeThresholdGovernor(learner=_StubLearner(rec), **kwargs)


class _StubDecision:
    def __init__(self, action: str, reason: str = "stub"):
        self.action = action
        self.reason = reason


class _StubRouter:
    def __init__(self, action: str):
        self._action = action

    def route(self, interval):
        return _StubDecision(self._action)


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
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        _governor(_seg_rec(), **kwargs)


def test_live_threshold_defaults_to_current():
    gov = _governor(_seg_rec())
    assert gov.live_threshold("medical") == 0.5


# ── propose: gating ──────────────────────────────────────────────────────────


def test_propose_blocks_on_global_source():
    gov = _governor(_seg_rec(source="global"))
    change = gov.propose("medical")
    assert isinstance(change, ThresholdChange)
    assert change.applied is False
    assert change.reason == "insufficient segment evidence"
    assert gov.live_threshold("medical") == 0.5


def test_propose_no_change_when_recommended_none():
    gov = _governor(_seg_rec(recommended=None))
    change = gov.propose("medical")
    assert change.applied is False
    assert change.reason == "no change recommended"


def test_propose_holds_for_human_approval():
    gov = _governor(_seg_rec(requires_human_approval=True))
    change = gov.propose("medical")
    assert change.applied is False
    assert change.requires_approval is True
    assert change.reason == "pending human approval"
    assert gov.live_threshold("medical") == 0.5


def test_propose_applies_when_no_approval_needed():
    gov = _governor(_seg_rec(recommended=0.7), max_step=0.05)
    change = gov.propose("medical")
    assert change.applied is True
    # stepped from 0.5 toward 0.7 by at most max_step
    assert change.to_threshold == pytest.approx(0.55)
    assert gov.live_threshold("medical") == pytest.approx(0.55)


def test_auto_apply_overrides_human_approval():
    gov = _governor(_seg_rec(requires_human_approval=True), auto_apply=True)
    change = gov.propose("medical")
    assert change.applied is True


# ── bounded stepping ─────────────────────────────────────────────────────────


def test_step_is_bounded_by_max_step():
    gov = _governor(_seg_rec(recommended=0.95), max_step=0.05)
    change = gov.propose("medical")
    assert change.to_threshold == pytest.approx(0.55)  # not 0.95


def test_step_reaches_target_when_within_max_step():
    gov = _governor(_seg_rec(recommended=0.52), max_step=0.05)
    change = gov.propose("medical")
    assert change.to_threshold == pytest.approx(0.52)


def test_step_can_move_downward():
    gov = _governor(_seg_rec(recommended=0.2), max_step=0.05)
    change = gov.propose("medical")
    assert change.to_threshold == pytest.approx(0.45)


# ── apply with approval ──────────────────────────────────────────────────────


def test_apply_with_approval_applies_human_gated_change():
    gov = _governor(_seg_rec(requires_human_approval=True))
    change = gov.apply("medical", approve=True)
    assert change.applied is True
    assert gov.live_threshold("medical") == pytest.approx(0.55)


def test_apply_without_approval_blocks():
    gov = _governor(_seg_rec(requires_human_approval=True))
    change = gov.apply("medical", approve=False)
    assert change.applied is False
    assert change.reason == "approval required"


def test_apply_nothing_to_apply_on_global_source():
    gov = _governor(_seg_rec(source="global"))
    change = gov.apply("medical", approve=True)
    assert change.applied is False
    assert change.reason == "nothing to apply"


def test_apply_no_approval_needed():
    gov = _governor(_seg_rec(requires_human_approval=False))
    change = gov.apply("medical")
    assert change.applied is True


# ── conformal / uncertainty tie-in ───────────────────────────────────────────


def test_effective_threshold_without_interval():
    gov = _governor(_seg_rec())
    eff = gov.effective_threshold("medical")
    assert isinstance(eff, EffectiveThreshold)
    assert eff.threshold == 0.5
    assert eff.uncertainty_adjusted is False
    assert eff.action is None


def test_effective_threshold_tightens_on_uncertainty():
    gov = _governor(
        _seg_rec(),
        uncertainty_router=_StubRouter("escalate_human"),
        uncertainty_penalty=0.1,
    )
    eff = gov.effective_threshold("medical", interval=object())
    assert eff.uncertainty_adjusted is True
    assert eff.threshold == pytest.approx(0.4)  # 0.5 - 0.1
    assert eff.action == "escalate_human"


def test_effective_threshold_allows_when_router_confident():
    gov = _governor(_seg_rec(), uncertainty_router=_StubRouter("allow"))
    eff = gov.effective_threshold("medical", interval=object())
    assert eff.uncertainty_adjusted is False
    assert eff.threshold == 0.5
    assert eff.action == "allow"


def test_effective_threshold_tighten_clamps_at_zero():
    gov = _governor(
        _seg_rec(),
        current_threshold=0.05,
        uncertainty_router=_StubRouter("escalate_model"),
        uncertainty_penalty=0.1,
    )
    eff = gov.effective_threshold("medical", interval=object())
    assert eff.threshold == 0.0  # clamped, not negative


# ── audit history + observe ──────────────────────────────────────────────────


def test_history_records_only_applied_changes():
    gov = _governor(_seg_rec(recommended=0.7))
    gov.propose("medical")  # applied
    assert len(gov.history()) == 1
    assert gov.history("medical")[0].applied is True
    assert gov.history("other") == ()


def test_observe_forwards_to_learner():
    learner = _StubLearner(_seg_rec())
    gov = RuntimeThresholdGovernor(
        learner=learner, current_threshold=0.5, clock=lambda: 1.0
    )
    gov.observe(segment="medical", score=0.7, human_approved=True)
    assert learner.observed == [(0.7, True, "medical")]


def test_change_timestamp_from_clock():
    gov = _governor(_seg_rec(recommended=0.52), clock=lambda: 42.0)
    change = gov.propose("medical")
    assert change.at == 42.0


# ── ProductionGuard wiring ──────────────────────────────────────────────────


def test_guard_builds_threshold_governor():
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
