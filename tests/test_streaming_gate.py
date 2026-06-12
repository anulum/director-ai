# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — claim-boundary streaming-gate tests

from __future__ import annotations

import pytest

from director_ai.core.runtime.streaming_gate import StreamingCoherenceGate


class _CountingScorer:
    """Returns a fixed score and records every text it was asked to score."""

    def __init__(self, score: float = 0.2) -> None:
        self._score = score
        self.calls: list[str] = []

    def __call__(self, text: str) -> float:
        self.calls.append(text)
        return self._score


def test_holds_pass_score_until_first_claim_completes() -> None:
    scorer = _CountingScorer(0.05)  # would halt if it ever ran on a fragment
    gate = StreamingCoherenceGate(scorer, min_words=4)
    # opening fragment, no terminator -> grace pass, model not called
    assert gate.update("Water boils at") == 1.0
    assert gate.update("Water boils at one") == 1.0
    assert scorer.calls == []


def test_rescores_at_sentence_boundary() -> None:
    scorer = _CountingScorer(0.2)
    gate = StreamingCoherenceGate(scorer, min_words=4)
    assert gate.update("Water boils at one hundred degrees.") == pytest.approx(0.2)
    assert scorer.calls == ["Water boils at one hundred degrees."]


def test_holds_between_boundaries() -> None:
    scores = iter([0.9, 0.1])
    gate = StreamingCoherenceGate(lambda _t: next(scores), min_words=3)
    assert gate.update("first claim here.") == pytest.approx(0.9)  # scored
    assert gate.update("first claim here. then more") == pytest.approx(0.9)  # held
    assert gate.update("first claim here. then more words now?") == pytest.approx(0.1)


def test_rescore_cap_forces_score_on_runon_text() -> None:
    scorer = _CountingScorer(0.3)
    gate = StreamingCoherenceGate(scorer, min_words=2, rescore_cap=5)
    # no punctuation, but the cap forces a score once enough words accrue
    held = gate.update("one two three four")  # 4 words < cap 5 -> held pass
    assert held == 1.0
    forced = gate.update("one two three four five six")  # >= cap -> scored
    assert forced == pytest.approx(0.3)
    assert scorer.calls == ["one two three four five six"]


def test_min_words_grace_skips_tiny_fragments() -> None:
    scorer = _CountingScorer(0.0)
    gate = StreamingCoherenceGate(scorer, min_words=5)
    # ends with a period but only 2 words -> still grace (too short to judge)
    assert gate.update("Hello there.") == 1.0
    assert scorer.calls == []


def test_reset_restores_pass_score() -> None:
    gate = StreamingCoherenceGate(_CountingScorer(0.1), min_words=2)
    gate.update("a real claim.")
    assert gate.last_score == pytest.approx(0.1)
    gate.reset()
    assert gate.last_score == 1.0


@pytest.mark.parametrize("bad", [{"min_words": 0}, {"rescore_cap": 0}])
def test_validates_construction(bad) -> None:
    with pytest.raises(ValueError):
        StreamingCoherenceGate(_CountingScorer(), **bad)


def test_completed_incoherent_claim_still_halts() -> None:
    # The gate changes WHEN scoring happens, not the verdict: a finished claim
    # that scores low is returned as-is, so a hard-limit halt still fires.
    gate = StreamingCoherenceGate(_CountingScorer(0.02), min_words=3)
    assert gate.update("the moon is made of cheese.") == pytest.approx(0.02)
