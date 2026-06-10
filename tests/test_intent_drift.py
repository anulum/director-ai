# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — intent-drift interlock tests
"""Multi-angle tests for the long-context intent-drift interlock.

Covers constructor validation, the EMA fold (first-turn seeding, decay across
turns, input clamping), the windowed escalation slope (rising vs flat, the
four-sample minimum), the combined drift risk and its trip condition (the
crescendo case that no single turn reveals, the benign steady case that never
trips, the ``min_turns`` guard against a lone spike, the negative-trend floor),
``reset``, ``DriftState.to_dict``, and the ``ConversationSession`` /
``CoherenceScorer`` wiring including default-off neutrality.
"""

from __future__ import annotations

import pytest

from director_ai.core.runtime.intent_drift import DriftState, IntentDriftInterlock


class TestConstruction:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"half_life_turns": 0},
            {"half_life_turns": -1},
            {"window": 1},
            {"trigger_threshold": 0.0},
            {"trigger_threshold": 1.1},
            {"min_turns": 0},
        ],
    )
    def test_invalid_params_raise(self, kwargs):
        with pytest.raises(ValueError):
            IntentDriftInterlock(**kwargs)

    def test_defaults_construct(self):
        il = IntentDriftInterlock()
        assert il.turn_count == 0


class TestEMA:
    def test_first_turn_seeds_ema(self):
        il = IntentDriftInterlock()
        state = il.update(intent_divergence=0.6, injection_risk=0.4)
        # No decay bias toward zero on the opening turn.
        assert state.sustained_divergence == pytest.approx(0.6)
        assert state.injection_pressure == pytest.approx(0.4)

    def test_ema_decays_prior_turns(self):
        il = IntentDriftInterlock(half_life_turns=1.0)  # decay = 0.5
        il.update(intent_divergence=1.0)
        state = il.update(intent_divergence=0.0)
        # 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert state.sustained_divergence == pytest.approx(0.5)

    def test_inputs_clamped(self):
        il = IntentDriftInterlock()
        state = il.update(intent_divergence=5.0, injection_risk=-2.0)
        assert state.sustained_divergence == pytest.approx(1.0)
        assert state.injection_pressure == pytest.approx(0.0)

    def test_turn_count_increments(self):
        il = IntentDriftInterlock()
        for _ in range(3):
            il.update(intent_divergence=0.1)
        assert il.turn_count == 3


class TestEscalationSlope:
    def test_rising_window_positive_escalation(self):
        il = IntentDriftInterlock()
        for d in (0.1, 0.2, 0.4, 0.6):
            state = il.update(intent_divergence=d)
        assert state.escalation > 0.0

    def test_flat_window_zero_escalation(self):
        il = IntentDriftInterlock()
        for _ in range(6):
            state = il.update(intent_divergence=0.3)
        assert state.escalation == pytest.approx(0.0)

    def test_needs_four_samples(self):
        il = IntentDriftInterlock()
        il.update(intent_divergence=0.1)
        il.update(intent_divergence=0.5)
        state = il.update(intent_divergence=0.9)  # only 3 samples
        assert state.escalation == pytest.approx(0.0)


class TestDriftTrip:
    def test_crescendo_trips_though_no_single_turn_does(self):
        # Each step is individually below a 0.7 per-turn block, but the
        # sustained drift + rising slope accumulate and trip the interlock.
        il = IntentDriftInterlock(trigger_threshold=0.45, min_turns=3)
        states = [
            il.update(intent_divergence=d, injection_risk=0.1, contradiction_trend=0.05)
            for d in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        ]
        assert all(d <= 0.7 for d in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
        assert not states[0].triggered
        assert states[-1].triggered

    def test_benign_steady_never_trips(self):
        il = IntentDriftInterlock()
        triggered = [il.update(intent_divergence=0.1).triggered for _ in range(10)]
        assert not any(triggered)

    def test_lone_spike_before_min_turns_does_not_trip(self):
        il = IntentDriftInterlock(min_turns=3)
        state = il.update(intent_divergence=1.0, injection_risk=1.0)
        assert state.turn_count == 1
        assert not state.triggered

    def test_negative_contradiction_trend_floored(self):
        # A negative (improving) contradiction trend cannot raise drift risk.
        il = IntentDriftInterlock()
        with_neg = il.update(intent_divergence=0.3, contradiction_trend=-0.9)
        il2 = IntentDriftInterlock()
        with_zero = il2.update(intent_divergence=0.3, contradiction_trend=0.0)
        assert with_neg.contradiction_pressure == pytest.approx(0.0)
        assert with_neg.drift_risk == pytest.approx(with_zero.drift_risk)

    def test_injection_pressure_raises_risk(self):
        clean = IntentDriftInterlock().update(intent_divergence=0.3, injection_risk=0.0)
        dirty = IntentDriftInterlock().update(intent_divergence=0.3, injection_risk=0.9)
        assert dirty.drift_risk > clean.drift_risk

    def test_drift_risk_clamped(self):
        il = IntentDriftInterlock()
        for _ in range(8):
            state = il.update(
                intent_divergence=1.0, injection_risk=1.0, contradiction_trend=1.0
            )
        assert 0.0 <= state.drift_risk <= 1.0


class TestResetAndDict:
    def test_reset_clears_state(self):
        il = IntentDriftInterlock()
        for _ in range(5):
            il.update(intent_divergence=0.8)
        il.reset()
        assert il.turn_count == 0
        state = il.update(intent_divergence=0.2)
        assert state.sustained_divergence == pytest.approx(0.2)

    def test_to_dict_shape(self):
        state = IntentDriftInterlock().update(intent_divergence=0.5)
        d = state.to_dict()
        assert set(d) == {
            "turn_count",
            "sustained_divergence",
            "escalation",
            "injection_pressure",
            "contradiction_pressure",
            "drift_risk",
            "triggered",
        }
        assert isinstance(d["triggered"], bool)
        assert isinstance(d["turn_count"], int)

    def test_state_is_dataclass(self):
        state = IntentDriftInterlock().update(intent_divergence=0.5)
        assert isinstance(state, DriftState)


class TestSessionWiring:
    def test_session_default_has_no_interlock(self):
        from director_ai.core.runtime.session import ConversationSession

        assert ConversationSession().intent_drift is None

    def test_session_opt_in_attaches_interlock(self):
        from director_ai.core.runtime.session import ConversationSession

        session = ConversationSession(track_intent_drift=True)
        assert isinstance(session.intent_drift, IntentDriftInterlock)


class TestScorerWiring:
    def test_default_off_leaves_fields_none(self):
        from director_ai.core import CoherenceScorer
        from director_ai.core.runtime.session import ConversationSession

        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        session = ConversationSession()
        _approved, score = scorer.review("q", "a", session=session)
        assert score.intent_drift_risk is None
        assert score.intent_drift_triggered is None

    def test_opt_in_populates_drift_fields(self):
        from director_ai.core import CoherenceScorer
        from director_ai.core.runtime.session import ConversationSession

        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        session = ConversationSession(track_intent_drift=True)
        for i in range(3):
            _approved, score = scorer.review(f"q{i}", f"a{i}", session=session)
        # Populated (not None) on every turn once the session tracks drift.
        assert score.intent_drift_risk is not None
        assert score.intent_drift_triggered is not None
        assert session.intent_drift.turn_count == 3
