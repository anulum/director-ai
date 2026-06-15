# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — property tests for this session's new guard modules

"""Invariant fuzz tests for the consensus, economics, and swarm-coherence modules.

These assert structural guarantees that must hold for *any* input — bounds,
symmetry, monotonicity — the kind of edge behaviour example-based tests miss.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from director_ai.core.consensus import CrossModelConsensus, ModelResponse
from director_ai.core.routing import DEFAULT_ACTIONS, HallucinationEconomics
from director_ai.core.swarm_coherence import SwarmCoherenceMonitor

_TEXT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=40).filter(
    lambda s: s.strip()
)
# A claim with >= 3 words (so _split_claims keeps it), built without filtering.
_WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=8)
_CLAIM = st.lists(_WORD, min_size=3, max_size=8).map(lambda ws: " ".join(ws) + ".")
_RISK = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_COST = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


# ── Cross-model consensus (lexical) ──────────────────────────────────────────


@given(texts=st.lists(_TEXT, min_size=2, max_size=6))
@settings(max_examples=120, deadline=None)
def test_consensus_is_bounded_and_matrix_symmetric(texts):
    panel = [ModelResponse(f"m{i}", t) for i, t in enumerate(texts)]
    res = CrossModelConsensus().consensus(panel)
    assert 0.0 <= res.consensus <= 1.0
    assert res.n_models == len(texts)
    matrix = res.agreement_matrix
    for i in range(len(texts)):
        assert matrix[i][i] == 1.0
        for j in range(len(texts)):
            assert matrix[i][j] == matrix[j][i]
            assert 0.0 <= matrix[i][j] <= 1.0
    assert res.recommendation in ("accept", "review", "escalate")


@given(text=_TEXT, n=st.integers(min_value=2, max_value=5))
@settings(max_examples=60, deadline=None)
def test_identical_responses_reach_full_consensus(text, n):
    panel = [ModelResponse(f"m{i}", text) for i in range(n)]
    res = CrossModelConsensus().consensus(panel)
    assert res.consensus == 1.0
    assert res.recommendation == "accept"


# ── Hallucination economics ──────────────────────────────────────────────────


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


# ── Swarm coherence ──────────────────────────────────────────────────────────


class _StubNLI:
    threshold = 0.5

    def __init__(self, table):
        self._table = table

    def contradiction(self, premise, hypothesis):
        return self._table.get((premise, hypothesis), 0.0)


@given(texts=st.lists(_TEXT, min_size=1, max_size=6))
@settings(max_examples=120, deadline=None)
def test_lexical_swarm_never_flags_and_coherence_bounded(texts):
    mon = SwarmCoherenceMonitor()  # no NLI -> never flags contradictions
    last = None
    for i, t in enumerate(texts):
        last = mon.observe(f"agent{i}", t)
        assert 0.0 <= last.coherence <= 1.0
        assert last.halted is False
        assert last.contradictions == ()
    assert last is not None


@given(
    prior=_CLAIM,
    new=_CLAIM,
    tail=st.lists(_TEXT, max_size=3),
)
@settings(max_examples=60, deadline=None)
def test_swarm_stays_halted_once_contradiction_seen(prior, new, tail):
    mon = SwarmCoherenceMonitor(nli=_StubNLI({(prior, new): 0.95, (new, prior): 0.95}))
    mon.observe("a", prior)
    update = mon.observe("b", new)
    if update.halted:
        # monotonic: every later observation stays halted
        for i, t in enumerate(tail):
            assert mon.observe(f"c{i}", t).halted is True
