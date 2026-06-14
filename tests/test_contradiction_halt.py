# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — contradiction-driven streaming halt tests (offline)

from __future__ import annotations

from director_ai.core.runtime.contradiction_halt import (
    ContradictionHalt,
    _as_facts,
)


class _StubScorer:
    """Returns contradiction = max keyword score per (premise, claim)."""

    threshold = 0.2

    def __init__(self, contra_by_fact: dict[str, float]) -> None:
        self._by_fact = contra_by_fact
        self.batches: list[list[tuple[str, str]]] = []

    def contradiction_batch(self, pairs):
        self.batches.append(list(pairs))
        return [self._by_fact.get(premise, 0.0) for premise, _ in pairs]


def test_as_facts_splits_context_string():
    assert _as_facts("a fact; another fact; third") == [
        "a fact",
        "another fact",
        "third",
    ]
    assert _as_facts(["x", " y ", ""]) == ["x", "y"]
    assert _as_facts(None) == []
    assert _as_facts("") == []


def test_halts_on_contradicting_fact():
    scorer = _StubScorer({"fact-A": 0.05, "fact-B": 0.9})
    halt = ContradictionHalt(scorer, lambda _c: "fact-A; fact-B")
    d = halt.should_halt("some claim")
    assert d.halt is True
    assert d.contradiction == 0.9
    assert d.fact == "fact-B"  # the strongest contradiction is reported


def test_no_halt_when_all_facts_consistent():
    scorer = _StubScorer({"fact-A": 0.05, "fact-B": 0.1})
    halt = ContradictionHalt(scorer, lambda _c: ["fact-A", "fact-B"])
    d = halt.should_halt("some claim")
    assert d.halt is False
    assert d.contradiction == 0.1
    assert d.fact == ""


def test_ungrounded_claim_never_halts():
    # No retrieval -> unsupported, not a contradiction -> must not halt,
    # and the scorer must not even be invoked.
    scorer = _StubScorer({"x": 0.99})
    halt = ContradictionHalt(scorer, lambda _c: None)
    d = halt.should_halt("an ungrounded but possibly-correct claim")
    assert d.halt is False
    assert scorer.batches == []


def test_empty_claim_never_halts():
    scorer = _StubScorer({"x": 0.99})
    halt = ContradictionHalt(scorer, lambda _c: "x")
    assert halt.should_halt("   ").halt is False
    assert scorer.batches == []


def test_threshold_override():
    scorer = _StubScorer({"f": 0.3})
    strict = ContradictionHalt(scorer, lambda _c: "f", threshold=0.5)
    assert strict.should_halt("c").halt is False  # 0.3 < 0.5
    assert strict.threshold == 0.5
    lenient = ContradictionHalt(scorer, lambda _c: "f", threshold=0.25)
    assert lenient.should_halt("c").halt is True  # 0.3 >= 0.25


def test_defaults_to_scorer_threshold():
    scorer = _StubScorer({"f": 0.25})
    halt = ContradictionHalt(scorer, lambda _c: "f")
    assert halt.threshold == 0.2  # scorer.threshold
    assert halt.should_halt("c").halt is True  # 0.25 >= 0.2


class _StubHalt:
    """Halts when ``marker`` appears in the completed claim."""

    def __init__(self, marker: str) -> None:
        self._marker = marker
        self.claims: list[str] = []

    def should_halt(self, claim: str):
        from director_ai.core.runtime.contradiction_halt import (
            ContradictionHaltDecision,
        )

        self.claims.append(claim)
        bad = self._marker in claim
        return ContradictionHaltDecision(
            halt=bad, contradiction=0.9 if bad else 0.0, fact="kb" if bad else ""
        )


def test_agent_stream_halts_on_contradicting_claim():
    import asyncio
    from collections.abc import AsyncIterator

    from director_ai.core.agent import CoherenceAgent

    agent = CoherenceAgent()
    agent.contradiction_halt = _StubHalt("cheese")

    class _TwoClaims:
        async def stream_tokens(self, prompt: str) -> AsyncIterator[str]:
            for w in ["the", "moon", "is", "cheese.", "second", "claim."]:
                yield w

    agent.generator = _TwoClaims()

    async def run():
        return [(t, c) async for t, c in agent.stream("q")]

    pairs = asyncio.run(run())
    tokens = [t for t, _ in pairs]
    # halted at the first completed claim; the second claim never streamed
    assert tokens[-1] == "cheese."
    assert "second" not in tokens
    assert pairs[-1][1] < 0.5  # coherence ≈ 1 − contradiction(0.9)


def test_agent_stream_completes_when_no_contradiction():
    import asyncio
    from collections.abc import AsyncIterator

    from director_ai.core.agent import CoherenceAgent

    agent = CoherenceAgent()
    agent.contradiction_halt = _StubHalt("never-matches")

    class _Gen:
        async def stream_tokens(self, prompt: str) -> AsyncIterator[str]:
            for w in ["all", "claims", "are", "fine."]:
                yield w

    agent.generator = _Gen()

    async def run():
        return [t async for t, _ in agent.stream("q")]

    assert asyncio.run(run()) == ["all", "claims", "are", "fine."]


class _StubStore:
    """Minimal grounding store exposing only the retriever the halt needs."""

    def retrieve_context(self, query: str) -> str | None:
        return "the sky is blue"


def test_build_contradiction_halt_disabled_returns_none():
    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig(streaming_contradiction_halt=False)
    assert cfg.build_contradiction_halt(_StubStore()) is None


def test_build_contradiction_halt_degrades_when_model_unavailable(monkeypatch):
    from director_ai.core.config import DirectorConfig
    from director_ai.core.scoring import contradiction as contra_mod

    def _boom(*args, **kwargs):
        raise ImportError("no transformers")

    monkeypatch.setattr(
        contra_mod.ContradictionScorer, "from_pretrained", _boom
    )
    cfg = DirectorConfig(streaming_contradiction_halt=True)
    # Model cannot load → keep the coherence halt, never crash startup.
    assert cfg.build_contradiction_halt(_StubStore()) is None


def test_build_contradiction_halt_wires_scorer_and_retriever(monkeypatch):
    from director_ai.core.config import DirectorConfig
    from director_ai.core.runtime.contradiction_halt import ContradictionHalt
    from director_ai.core.scoring import contradiction as contra_mod

    captured: dict = {}

    def _fake_from_pretrained(model_id, *, device=-1, threshold=0.2, **kwargs):
        captured.update(model_id=model_id, device=device, threshold=threshold)
        return _StubScorer({})

    monkeypatch.setattr(
        contra_mod.ContradictionScorer, "from_pretrained", _fake_from_pretrained
    )
    store = _StubStore()
    cfg = DirectorConfig(
        streaming_contradiction_halt=True,
        streaming_contradiction_threshold=0.35,
        streaming_contradiction_device=0,
    )
    halt = cfg.build_contradiction_halt(store)
    assert isinstance(halt, ContradictionHalt)
    # config fields reach the scorer factory
    assert captured["model_id"] == cfg.streaming_contradiction_model
    assert captured["device"] == 0
    assert captured["threshold"] == 0.35
    # the store's retriever is the one the halt queries
    assert halt._retrieve == store.retrieve_context
