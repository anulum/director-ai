# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — hallucination economics tests

from __future__ import annotations

import pytest

from director_ai.core.routing import (
    DEFAULT_ACTIONS,
    EconomicDecision,
    GuardAction,
    HallucinationEconomics,
)

# ── GuardAction validation ──────────────────────────────────────────────────


def test_guard_action_rejects_negative_cost():
    with pytest.raises(ValueError, match="cost"):
        GuardAction("x", cost=-0.1, catch=0.5)


@pytest.mark.parametrize("catch", [-0.01, 1.01, 2.0])
def test_guard_action_rejects_out_of_range_catch(catch):
    with pytest.raises(ValueError, match="catch"):
        GuardAction("x", cost=0.1, catch=catch)


def test_guard_action_accepts_boundaries():
    assert GuardAction("a", cost=0.0, catch=0.0).catch == 0.0
    assert GuardAction("b", cost=3.0, catch=1.0).catch == 1.0


# ── expected_cost formula ───────────────────────────────────────────────────


def test_expected_cost_formula():
    action = GuardAction("nli", cost=0.2, catch=0.9)
    # 0.2 + 0.5 * (1 - 0.9) * 10 = 0.2 + 0.5
    assert HallucinationEconomics.expected_cost(action, 0.5, 10.0) == pytest.approx(0.7)


def test_expected_cost_skip_is_pure_risk_times_cost():
    skip = GuardAction("skip", cost=0.0, catch=0.0)
    assert HallucinationEconomics.expected_cost(skip, 0.3, 20.0) == pytest.approx(6.0)


# ── decide: regime selection ────────────────────────────────────────────────


def test_decide_very_low_risk_skips():
    econ = HallucinationEconomics()
    d = econ.decide(0.01)  # hallucination_cost default 1.0
    assert isinstance(d, EconomicDecision)
    assert d.action == "skip"
    assert d.worth_guarding is False
    assert d.value == pytest.approx(0.0)
    assert "guarding not worth its cost for this request" in d.rationale


def test_decide_low_risk_low_stakes_picks_cheap_guard():
    econ = HallucinationEconomics()
    d = econ.decide(0.05)
    assert d.action == "heuristic"
    assert d.worth_guarding is True


def test_decide_high_risk_high_stakes_escalates():
    econ = HallucinationEconomics()
    d = econ.decide(0.9, hallucination_cost=100.0)
    assert d.action == "escalate"
    assert d.worth_guarding is True
    assert d.value == pytest.approx(90.0 - 3.7)
    assert "elevated hallucination risk" in d.rationale


def test_decide_high_risk_low_stakes_uses_nli_not_escalation():
    econ = HallucinationEconomics()
    d = econ.decide(0.9, hallucination_cost=1.0)
    assert d.action == "nli"


# ── decide: numeric fields ──────────────────────────────────────────────────


def test_decide_reports_baseline_value_and_residual():
    econ = HallucinationEconomics()
    d = econ.decide(0.5, hallucination_cost=10.0)
    # baseline = 0.5 * 10 = 5.0; best is nli (0.2 + 0.5*0.1*10 = 0.7)
    assert d.baseline_cost == pytest.approx(5.0)
    assert d.action == "nli"
    assert d.expected_cost == pytest.approx(0.7)
    assert d.value == pytest.approx(4.3)
    assert d.residual_risk == pytest.approx(0.5 * (1 - 0.9))


def test_decide_breakdown_covers_every_action():
    econ = HallucinationEconomics()
    d = econ.decide(0.4, hallucination_cost=5.0)
    names = {name for name, _ in d.breakdown}
    assert names == {a.name for a in DEFAULT_ACTIONS}
    # breakdown costs match expected_cost
    by_name = dict(d.breakdown)
    for a in DEFAULT_ACTIONS:
        assert by_name[a.name] == pytest.approx(
            HallucinationEconomics.expected_cost(a, 0.4, 5.0), abs=1e-6
        )


# ── custom action menus ─────────────────────────────────────────────────────


def test_custom_menu_single_action():
    only = GuardAction("only", cost=0.5, catch=0.8)
    econ = HallucinationEconomics(actions=[only], hallucination_cost=10.0)
    d = econ.decide(0.5)
    assert d.action == "only"
    assert d.expected_cost == pytest.approx(0.5 + 0.5 * 0.2 * 10.0)


def test_perfect_catch_zero_cost_dominates():
    menu = [
        GuardAction("skip", 0.0, 0.0),
        GuardAction("oracle", 0.0, 1.0),  # free perfect guard
    ]
    econ = HallucinationEconomics(actions=menu)
    d = econ.decide(0.8, hallucination_cost=100.0)
    assert d.action == "oracle"
    assert d.expected_cost == pytest.approx(0.0)
    assert d.residual_risk == pytest.approx(0.0)


# ── validation ──────────────────────────────────────────────────────────────


def test_empty_action_menu_raises():
    with pytest.raises(ValueError, match="at least one"):
        HallucinationEconomics(actions=[])


def test_negative_default_hallucination_cost_raises():
    with pytest.raises(ValueError, match="hallucination_cost"):
        HallucinationEconomics(hallucination_cost=-1.0)


@pytest.mark.parametrize("risk", [-0.01, 1.01])
def test_decide_rejects_out_of_range_risk(risk):
    with pytest.raises(ValueError, match="risk"):
        HallucinationEconomics().decide(risk)


def test_decide_rejects_negative_per_call_hallucination_cost():
    with pytest.raises(ValueError, match="hallucination_cost"):
        HallucinationEconomics().decide(0.5, hallucination_cost=-2.0)


def test_default_actions_are_ordered_and_valid():
    assert DEFAULT_ACTIONS[0].name == "skip"
    assert all(0.0 <= a.catch <= 1.0 and a.cost >= 0.0 for a in DEFAULT_ACTIONS)


# ── ProductionGuard wiring ──────────────────────────────────────────────────


def _guard():
    from director_ai.core.config import DirectorConfig
    from director_ai.guard import ProductionGuard

    return ProductionGuard(config=DirectorConfig(use_nli=False))


def test_guard_economics_property_persists():
    guard = _guard()
    first = guard.economics
    assert isinstance(first, HallucinationEconomics)
    assert guard.economics is first


def test_guard_economics_decision():
    guard = _guard()
    d = guard.guard_economics(0.9, hallucination_cost=100.0)
    assert isinstance(d, EconomicDecision)
    assert d.action == "escalate"
    assert d.worth_guarding is True


def test_guard_economics_low_risk_skips():
    guard = _guard()
    d = guard.guard_economics(0.01)
    assert d.action == "skip"
    assert d.worth_guarding is False


# ── Property-based invariants (hypothesis) ───────────────────────────────────

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_RISK = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_COST = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(risk=_RISK, cost=_COST)
@settings(max_examples=200, deadline=None)
def test_economics_decision_is_consistent(risk, cost):
    decision = HallucinationEconomics().decide(risk, hallucination_cost=cost)
    names = {a.name for a in DEFAULT_ACTIONS}
    assert decision.action in names
    assert decision.expected_cost >= 0.0
    assert decision.residual_risk >= 0.0
    # the chosen action is never worse than doing nothing (skip is in the menu)
    assert decision.expected_cost <= decision.baseline_cost + 1e-6
    # value is the loss avoided versus the no-guard baseline
    assert decision.value >= -1e-6
    if decision.worth_guarding:
        assert decision.value > 0.0


@given(risk=_RISK, cost=_COST)
@settings(max_examples=100, deadline=None)
def test_expected_cost_matches_breakdown_minimum(risk, cost):
    decision = HallucinationEconomics().decide(risk, hallucination_cost=cost)
    best = min(c for _, c in decision.breakdown)
    assert decision.expected_cost == best
