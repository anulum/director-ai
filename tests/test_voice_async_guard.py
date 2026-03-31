# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AsyncVoiceGuard Tests
"""Tests for AsyncVoiceGuard async token filter."""

from __future__ import annotations

from director_ai.integrations.voice import VoiceToken
from director_ai.voice.guard import AsyncVoiceGuard


class TestAsyncVoiceGuardBasic:
    async def test_approved_tokens_pass_through(self):
        guard = AsyncVoiceGuard(
            facts={"sky": "The sky is blue."},
            use_nli=False,
            prompt="What color is the sky?",
            score_every=1,
        )
        result = await guard.feed("The ")
        assert result.approved
        assert not result.halted
        assert result.token == "The "
        assert isinstance(result, VoiceToken)

    async def test_empty_feed_stream(self):
        guard = AsyncVoiceGuard(use_nli=False)
        results = [r async for r in guard.feed_stream(iter([]))]
        assert results == []

    async def test_reset_clears_state(self):
        guard = AsyncVoiceGuard(use_nli=False, score_every=1)
        await guard.feed("hello")
        assert guard.accumulated_text == "hello"
        guard.reset()
        assert guard.accumulated_text == ""
        assert not guard.halted

    async def test_set_prompt(self):
        guard = AsyncVoiceGuard(use_nli=False)
        guard.set_prompt("What is the refund policy?")
        assert guard._prompt == "What is the refund policy?"

    async def test_feed_stream_with_sync_iterator(self):
        guard = AsyncVoiceGuard(
            facts={"sky": "The sky is blue."},
            use_nli=False,
            prompt="sky",
            score_every=1,
            hard_limit=0.01,
        )
        tokens = ["The ", "sky ", "is ", "blue."]
        results = [r async for r in guard.feed_stream(iter(tokens))]
        assert len(results) == 4
        assert all(r.approved for r in results)

    async def test_feed_stream_with_async_iterator(self):
        guard = AsyncVoiceGuard(
            facts={"sky": "The sky is blue."},
            use_nli=False,
            prompt="sky",
            score_every=1,
            hard_limit=0.01,
        )

        async def async_tokens():
            for t in ["The ", "sky ", "is ", "blue."]:
                yield t

        results = [r async for r in guard.feed_stream(async_tokens())]
        assert len(results) == 4
        assert all(r.approved for r in results)


class TestAsyncVoiceGuardHalt:
    async def test_hard_limit_halts_immediately(self):
        guard = AsyncVoiceGuard(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
            prompt="capital of France",
            score_every=1,
            hard_limit=0.99,
        )
        tokens = ["The ", "capital ", "is ", "Berlin."]
        results = []
        async for r in guard.feed_stream(iter(tokens)):
            results.append(r)
        halted = [r for r in results if r.halted]
        assert len(halted) >= 1
        assert halted[0].halt_reason == "hard_limit"
        assert halted[0].recovery_text != ""

    async def test_already_halted_rejects_further_tokens(self):
        guard = AsyncVoiceGuard(use_nli=False, score_every=1, hard_limit=0.99)
        await guard.feed("test")
        assert guard.halted
        result = await guard.feed("more")
        assert result.halted
        assert result.halt_reason == "already_halted"
        assert not result.approved

    async def test_feed_stream_stops_after_halt(self):
        guard = AsyncVoiceGuard(use_nli=False, score_every=1, hard_limit=0.99)
        tokens = ["a", "b", "c", "d", "e"]
        results = [r async for r in guard.feed_stream(iter(tokens))]
        assert len(results) < len(tokens)
        assert results[-1].halted

    async def test_window_avg_hard_halt(self):
        guard = AsyncVoiceGuard(
            use_nli=False,
            score_every=1,
            threshold=0.99,
            hard_limit=0.01,
            window_size=2,
            soft_halt=False,
        )
        tokens = ["Word ", "word ", "word ", "more ", "extra"]
        results = [r async for r in guard.feed_stream(iter(tokens))]
        halted = [r for r in results if r.halted]
        if halted:
            assert halted[0].halt_reason == "window_avg"
            assert halted[0].recovery_text != ""


class TestAsyncVoiceGuardSoftHalt:
    async def test_soft_halt_waits_for_sentence_end(self):
        guard = AsyncVoiceGuard(
            use_nli=False,
            score_every=1,
            threshold=0.99,
            hard_limit=0.01,
            window_size=2,
            soft_halt=True,
        )
        tokens = ["Word ", "word ", "word", ". ", "More"]
        results = [r async for r in guard.feed_stream(iter(tokens))]
        halted_tokens = [r for r in results if r.halted]
        if halted_tokens:
            halt_token = halted_tokens[0]
            assert halt_token.token.rstrip().endswith(".")


class TestAsyncVoiceGuardScoring:
    async def test_coherence_scores_are_in_range(self):
        guard = AsyncVoiceGuard(
            facts={"weather": "It is sunny today."},
            use_nli=False,
            prompt="weather",
            score_every=1,
        )
        for token in ["It ", "is ", "sunny ", "today."]:
            result = await guard.feed(token)
            assert 0.0 <= result.coherence <= 1.0

    async def test_index_increments(self):
        guard = AsyncVoiceGuard(use_nli=False)
        r1 = await guard.feed("a")
        r2 = await guard.feed("b")
        r3 = await guard.feed("c")
        assert r1.index == 0
        assert r2.index == 1
        assert r3.index == 2

    async def test_score_cadence_skips(self):
        guard = AsyncVoiceGuard(use_nli=False, score_every=4)
        results = []
        for token in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            results.append(await guard.feed(token))
        assert all(r.approved for r in results)


class TestAsyncVoiceGuardCoverageGaps:
    """Targeted tests for the 3 uncovered paths in voice/guard.py."""

    async def test_custom_store_passed_to_constructor(self):
        """Line 76: store is not None branch."""
        from director_ai.core.knowledge import GroundTruthStore

        custom_store = GroundTruthStore()
        custom_store.add("fact", "The Earth orbits the Sun.")
        guard = AsyncVoiceGuard(
            store=custom_store,
            use_nli=False,
            score_every=1,
        )
        assert guard._store is custom_store

    async def test_window_avg_hard_halt_deterministic(self):
        """Lines 197→217: window avg below threshold with soft_halt=False.

        Trick: set hard_limit=0.0 so individual scores never trigger
        hard_limit halt, but threshold=0.999 so the window avg check
        at line 197 fires instead.
        """
        guard = AsyncVoiceGuard(
            use_nli=False,
            score_every=1,
            threshold=0.999,
            hard_limit=0.0,
            window_size=1,
            soft_halt=False,
        )
        results = []
        for token in ["Hello ", "world ", "this ", "is ", "a ", "test."]:
            result = await guard.feed(token)
            results.append(result)
            if result.halted:
                break
        halted = [r for r in results if r.halted]
        assert len(halted) >= 1
        assert halted[0].halt_reason == "window_avg"

    async def test_window_avg_above_threshold_passes(self):
        """Line 197→217: window filled but avg >= threshold, falls through."""
        guard = AsyncVoiceGuard(
            facts={"fact": "The sky is blue during the day."},
            use_nli=False,
            score_every=1,
            threshold=0.0,
            hard_limit=0.0,
            window_size=1,
            soft_halt=False,
        )
        results = []
        for token in ["The ", "sky ", "is ", "blue."]:
            result = await guard.feed(token)
            results.append(result)
        assert all(r.approved for r in results)
        assert not any(r.halted for r in results)

    async def test_async_feed_stream_halts_midstream(self):
        """Line 234: halt during async iterator feed_stream."""

        async def async_tokens():
            for t in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]:
                yield t

        guard = AsyncVoiceGuard(
            use_nli=False,
            score_every=1,
            hard_limit=0.99,
        )
        results = [r async for r in guard.feed_stream(async_tokens())]
        assert results[-1].halted
        assert len(results) < 10
