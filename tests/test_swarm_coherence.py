# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — swarm coherence tests

from __future__ import annotations

import pytest

from director_ai.core.swarm_coherence import (
    AgentMessage,
    CascadeContradiction,
    CoherenceUpdate,
    SwarmCoherenceMonitor,
)
from director_ai.core.swarm_coherence.cascade_monitor import (
    _lexical_overlap,
    _split_claims,
)


class _StubNLI:
    """Directional NLI stub keyed by (premise, hypothesis); default 0.0."""

    threshold = 0.5

    def __init__(self, table: dict[tuple[str, str], float] | None = None):
        self._table = table or {}

    def contradiction(self, premise: str, hypothesis: str) -> float:
        return self._table.get((premise, hypothesis), 0.0)


# ── helpers ──────────────────────────────────────────────────────────────────


def test_split_claims_filters_and_caps():
    assert _split_claims("A real claim here. No. Another real claim here.") == [
        "A real claim here.",
        "Another real claim here.",
    ]
    assert (
        len(_split_claims(" ".join(["Sentence with content here."] * 30), cap=4)) == 4
    )


def test_lexical_overlap_parity_and_fallback(monkeypatch):
    from director_ai.core.swarm_coherence import cascade_monitor as cm

    dispatched = _lexical_overlap("alpha beta gamma", "alpha beta delta")
    monkeypatch.setattr(cm, "_RUST_SWARM", False)
    monkeypatch.setattr(cm, "rust_word_overlap", None)
    assert cm._lexical_overlap("alpha beta gamma", "alpha beta delta") == pytest.approx(
        dispatched
    )
    assert cm._lexical_overlap("", "") == 0.0


# ── coherent cascade ─────────────────────────────────────────────────────────


def test_coherent_cascade_does_not_halt():
    mon = SwarmCoherenceMonitor(nli=_StubNLI())
    u1 = mon.observe("planner", "The deadline is the first of March.")
    u2 = mon.observe("researcher", "The budget is fifty thousand francs.")
    assert isinstance(u1, CoherenceUpdate)
    assert not u1.halted and not u2.halted
    assert u2.coherence == 1.0
    assert "coherent with established claims" in u2.rationale


# ── contradiction halts cascade with evidence ───────────────────────────────


def test_contradiction_halts_cascade_with_evidence():
    prior = "The deadline is the first of March."
    new = "The deadline is the first of April."
    mon = SwarmCoherenceMonitor(nli=_StubNLI({(prior, new): 0.93}))
    mon.observe("planner", prior)
    update = mon.observe("editor", new)
    assert update.halted is True
    assert mon.halted is True
    assert len(update.contradictions) == 1
    c = update.contradictions[0]
    assert isinstance(c, CascadeContradiction)
    assert c.new_agent == "editor" and c.prior_agent == "planner"
    assert c.new_claim == new and c.prior_claim == prior
    assert c.contradiction == pytest.approx(0.93)
    assert 0.0 <= c.topical_overlap <= 1.0
    assert "cascade halted" in update.rationale[0]
    assert update.coherence < 1.0


def test_same_agent_claims_do_not_self_contradict():
    a = "The sky is blue."
    b = "The sky is green."
    mon = SwarmCoherenceMonitor(nli=_StubNLI({(a, b): 0.9, (b, a): 0.9}))
    # both claims from the SAME agent in one message -> no cross-agent check
    update = mon.observe("solo", f"{a} {b}")
    assert update.halted is False
    assert update.contradictions == ()


def test_same_agent_across_messages_is_not_cross_agent():
    first = "The deadline is in March."
    second = "The deadline is in April."
    # Even though the two messages conflict, they are from the SAME agent across
    # turns, so the cross-agent cascade check skips them (no swarm contradiction).
    mon = SwarmCoherenceMonitor(nli=_StubNLI({(first, second): 0.95}))
    mon.observe("a1", first)
    update = mon.observe("a1", second)
    assert update.halted is False
    assert update.contradictions == ()


def test_observe_after_halt_short_circuits():
    prior, new = "The constraint holds firmly.", "The constraint does not hold."
    mon = SwarmCoherenceMonitor(nli=_StubNLI({(prior, new): 0.9}))
    mon.observe("a1", prior)
    mon.observe("a2", new)  # halts
    after = mon.observe("a3", "Some entirely new statement appears here.")
    assert after.halted is True
    assert after.contradictions == ()
    assert "cascade already halted" in after.rationale


# ── lexical-only mode ────────────────────────────────────────────────────────


def test_without_nli_never_flags_contradiction():
    mon = SwarmCoherenceMonitor()
    prior, new = "The total is ten units.", "The total is twelve units."
    mon.observe("a1", prior)
    update = mon.observe("a2", new)
    assert update.halted is False
    assert update.contradictions == ()
    assert "lexical novelty only (no NLI scorer supplied)" in update.rationale


# ── coherence score and accounting ──────────────────────────────────────────


def test_coherence_score_reflects_contradicted_fraction():
    prior, new = "Alpha is true.", "Alpha is false."
    mon = SwarmCoherenceMonitor(nli=_StubNLI({(prior, new): 0.9}))
    mon.observe("a1", prior)  # 1 claim, 0 contradicted
    update = mon.observe("a2", new)  # +1 claim, +1 contradicted
    # contradicted=1, total=2 -> coherence 0.5
    assert update.coherence == pytest.approx(0.5)


def test_empty_swarm_is_coherent():
    mon = SwarmCoherenceMonitor(nli=_StubNLI())
    assert mon._coherence() == 1.0


# ── max_claims cap ───────────────────────────────────────────────────────────


def test_max_claims_drops_oldest_established():
    mon = SwarmCoherenceMonitor(nli=_StubNLI(), max_claims=2)
    mon.observe("a1", "First claim here is long. Second claim here is long.")
    mon.observe("a2", "Third claim here is long.")
    # only the 2 most recent established claims are retained
    assert len(mon._established) == 2
    assert mon._established[-1][1] == "Third claim here is long."


# ── observe_message wrapper ──────────────────────────────────────────────────


def test_observe_message_wrapper():
    prior, new = "The price went up.", "The price went down."
    mon = SwarmCoherenceMonitor(nli=_StubNLI({(prior, new): 0.9}))
    mon.observe_message(AgentMessage("a1", (prior,)))
    update = mon.observe_message(AgentMessage("a2", (new,)))
    assert update.halted is True


# ── reset ────────────────────────────────────────────────────────────────────


def test_reset_clears_state_and_unhalts():
    prior, new = "It rained all day.", "It did not rain."
    mon = SwarmCoherenceMonitor(nli=_StubNLI({(prior, new): 0.9}))
    mon.observe("a1", prior)
    mon.observe("a2", new)
    assert mon.halted
    mon.reset()
    assert not mon.halted
    assert mon._established == []
    assert mon._coherence() == 1.0


# ── thresholds ───────────────────────────────────────────────────────────────


def test_flag_threshold_defaults_and_overrides():
    assert SwarmCoherenceMonitor(nli=_StubNLI())._flag_threshold == 0.5
    assert (
        SwarmCoherenceMonitor(
            nli=_StubNLI(), contradiction_threshold=0.3
        )._flag_threshold
        == 0.3
    )
    assert SwarmCoherenceMonitor()._flag_threshold == 0.5


def test_threshold_gates_weak_contradictions():
    prior, new = "The claim P holds.", "The claim P fails."
    mon = SwarmCoherenceMonitor(
        nli=_StubNLI({(prior, new): 0.4}), contradiction_threshold=0.5
    )
    mon.observe("a1", prior)
    update = mon.observe("a2", new)
    assert update.halted is False  # 0.4 < 0.5


# ── validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs", [{"contradiction_threshold": 1.5}, {"max_claims": 0}]
)
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        SwarmCoherenceMonitor(**kwargs)


# ── multi-contradiction ordering ─────────────────────────────────────────────


def test_guard_factory_returns_fresh_isolated_monitors():
    from director_ai.core.config import DirectorConfig
    from director_ai.guard import ProductionGuard

    guard = ProductionGuard(config=DirectorConfig(use_nli=False))
    prior, new = "The server is online.", "The server is offline."
    m1 = guard.new_swarm_monitor(nli=_StubNLI({(prior, new): 0.9}))
    m2 = guard.new_swarm_monitor(nli=_StubNLI({(prior, new): 0.9}))
    assert m1 is not m2
    m1.observe("a1", prior)
    u = m1.observe("a2", new)
    assert u.halted is True
    assert m2.halted is False  # isolated state


def test_multiple_contradictions_sorted_strongest_first():
    p1, p2 = "Fact one stands firm.", "Fact two stands firm."
    new = "Both facts collapse entirely now."
    mon = SwarmCoherenceMonitor(
        nli=_StubNLI({(p1, new): 0.6, (p2, new): 0.95}), contradiction_threshold=0.5
    )
    mon.observe("a1", f"{p1} {p2}")
    update = mon.observe("a2", new)
    assert update.halted is True
    assert len(update.contradictions) == 2
    assert update.contradictions[0].contradiction == pytest.approx(0.95)
    assert update.contradictions[1].contradiction == pytest.approx(0.6)
