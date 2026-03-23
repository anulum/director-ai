# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Voice Guard Tests

"""Tests for VoiceGuard real-time token filter."""

from __future__ import annotations

import pytest

from director_ai.integrations.voice import VoiceGuard, VoiceToken


class TestVoiceGuardBasic:
    def test_approved_tokens_pass_through(self):
        guard = VoiceGuard(
            facts={"sky": "The sky is blue."},
            use_nli=False,
            prompt="What color is the sky?",
            score_every=1,
        )
        result = guard.feed("The ")
        assert result.approved
        assert not result.halted
        assert result.token == "The "

    def test_empty_feed_all(self):
        guard = VoiceGuard(use_nli=False)
        results = guard.feed_all([])
        assert results == []

    def test_reset_clears_state(self):
        guard = VoiceGuard(use_nli=False, score_every=1)
        guard.feed("hello")
        assert guard.accumulated_text == "hello"
        guard.reset()
        assert guard.accumulated_text == ""
        assert not guard.halted

    def test_set_prompt(self):
        guard = VoiceGuard(use_nli=False)
        guard.set_prompt("What is the refund policy?")
        assert guard._prompt == "What is the refund policy?"


class TestVoiceGuardHalt:
    def test_hard_limit_halts_immediately(self):
        guard = VoiceGuard(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
            prompt="capital of France",
            score_every=1,
            hard_limit=0.99,
        )
        tokens = ["The ", "capital ", "is ", "Berlin."]
        results = guard.feed_all(tokens)
        halted = [r for r in results if r.halted]
        assert len(halted) >= 1
        assert halted[0].halt_reason == "hard_limit"
        assert halted[0].recovery_text != ""

    def test_already_halted_rejects_further_tokens(self):
        guard = VoiceGuard(use_nli=False, score_every=1, hard_limit=0.99)
        guard.feed("test")
        assert guard.halted
        result = guard.feed("more")
        assert result.halted
        assert result.halt_reason == "already_halted"
        assert not result.approved

    def test_feed_all_stops_after_halt(self):
        guard = VoiceGuard(use_nli=False, score_every=1, hard_limit=0.99)
        tokens = ["a", "b", "c", "d", "e"]
        results = guard.feed_all(tokens)
        assert len(results) < len(tokens)
        assert results[-1].halted


class TestVoiceGuardSoftHalt:
    def test_soft_halt_waits_for_sentence_end(self):
        guard = VoiceGuard(
            use_nli=False,
            score_every=1,
            threshold=0.99,
            hard_limit=0.01,
            window_size=2,
            soft_halt=True,
        )
        tokens = ["Word ", "word ", "word", ". ", "More"]
        results = guard.feed_all(tokens)
        halted_tokens = [r for r in results if r.halted]
        if halted_tokens:
            halt_token = halted_tokens[0]
            assert halt_token.token.rstrip().endswith(".")


class TestVoiceGuardScoring:
    def test_coherence_scores_are_in_range(self):
        guard = VoiceGuard(
            facts={"weather": "It is sunny today."},
            use_nli=False,
            prompt="weather",
            score_every=1,
        )
        for token in ["It ", "is ", "sunny ", "today."]:
            result = guard.feed(token)
            assert 0.0 <= result.coherence <= 1.0

    def test_index_increments(self):
        guard = VoiceGuard(use_nli=False)
        r1 = guard.feed("a")
        r2 = guard.feed("b")
        r3 = guard.feed("c")
        assert r1.index == 0
        assert r2.index == 1
        assert r3.index == 2

    def test_score_cadence_skips(self):
        guard = VoiceGuard(use_nli=False, score_every=4)
        results = []
        for token in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            results.append(guard.feed(token))
        assert all(r.approved for r in results)


class TestVoiceToken:
    def test_dataclass_fields(self):
        vt = VoiceToken(
            token="hello",
            index=0,
            approved=True,
            coherence=0.85,
        )
        assert vt.token == "hello"
        assert vt.halted is False
        assert vt.recovery_text == ""
