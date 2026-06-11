# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Director-Lite facade tests
"""Tests for the Director-Lite streaming-halt facade.

Composition is verified with an injected stub scorer so the halt is
deterministic and model-free: a clean stream passes through untouched, a stream
that loses coherence is halted with the offending tail removed, the one-call
helper and ``safe_text`` delegate correctly, an injected scorer overrides the
heuristic default, and the symbols are reachable from the top-level package. One
end-to-end case exercises the real heuristic scorer to prove the halt fires.
"""

from __future__ import annotations

from types import SimpleNamespace

from director_ai.lite import StreamGuard, streaming_guard


class _StubScorer:
    """Scores high until a trigger token appears in the accumulated text."""

    def __init__(
        self, *, trigger: str | None = None, high: float = 0.9, low: float = 0.1
    ):
        self._trigger = trigger
        self._high = high
        self._low = low

    def review(self, prompt: str, response: str):
        bad = self._trigger is not None and self._trigger in response
        return (not bad), SimpleNamespace(score=self._low if bad else self._high)


def _tokens(text: str):
    return [t + " " for t in text.split()]


class TestStreamGuard:
    def test_clean_stream_passes(self):
        g = StreamGuard(scorer=_StubScorer(), threshold=0.5)
        session = g.guard(_tokens("Paris is the capital of France"), prompt="q")
        assert session.halted is False
        assert "Paris" in session.output
        assert "France" in session.output

    def test_incoherent_stream_halts(self):
        g = StreamGuard(scorer=_StubScorer(trigger="Berlin"), threshold=0.5)
        session = g.guard(_tokens("The capital is Berlin not Paris"), prompt="q")
        assert session.halted is True
        assert session.halt_reason
        # The offending token (and after) is removed from the surviving output.
        assert "Berlin" not in session.output

    def test_safe_text_returns_output_string(self):
        g = StreamGuard(scorer=_StubScorer())
        text = g.safe_text(_tokens("all good here"), prompt="q")
        assert isinstance(text, str)
        assert "good" in text

    def test_injected_scorer_overrides_default(self):
        # With an injected scorer, the default heuristic scorer is never built.
        stub = _StubScorer()
        g = StreamGuard(facts={"x": "y"}, scorer=stub)
        assert g._ensure_scorer() is stub


class TestStreamingGuardHelper:
    def test_one_call_delegates(self):
        session = streaming_guard(
            _tokens("clean output text"),
            scorer=_StubScorer(),
            prompt="q",
            threshold=0.5,
        )
        assert session.halted is False
        assert "clean" in session.output

    def test_one_call_halts_on_trigger(self):
        session = streaming_guard(
            _tokens("good then BADTOKEN appears"),
            scorer=_StubScorer(trigger="BADTOKEN"),
            prompt="q",
        )
        assert session.halted is True


class TestTopLevelExport:
    def test_importable_from_package_root(self):
        import director_ai

        assert hasattr(director_ai, "StreamGuard")
        assert hasattr(director_ai, "streaming_guard")
        from director_ai import StreamGuard as RootGuard

        assert RootGuard is StreamGuard


class TestHeuristicEndToEnd:
    def test_real_heuristic_scorer_halts_on_contradiction(self):
        # No injected scorer, no NLI model — the default heuristic scorer must
        # still halt a clearly contradictory stream.
        g = StreamGuard(
            facts={"capital": "Paris is the capital of France."},
            threshold=0.5,
        )
        session = g.guard(
            _tokens("The capital of France is Berlin and also Tokyo"),
            prompt="What is the capital of France?",
        )
        assert session.halted is True
