# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Conversation Session Tests

import threading

import pytest

from director_ai.core.session import ConversationSession, Turn


class TestConversationSession:
    def test_creation_with_auto_id(self):
        s = ConversationSession()
        assert len(s.session_id) == 36  # UUID4 format

    def test_creation_with_custom_id(self):
        s = ConversationSession(session_id="test-123")
        assert s.session_id == "test-123"

    def test_add_turn(self):
        s = ConversationSession()
        turn = s.add_turn("What is AI?", "AI is...", 0.85)
        assert isinstance(turn, Turn)
        assert turn.prompt == "What is AI?"
        assert turn.response == "AI is..."
        assert turn.score == 0.85
        assert turn.turn_index == 0

    def test_turns_returns_copy(self):
        s = ConversationSession()
        s.add_turn("q1", "a1", 0.9)
        turns = s.turns
        turns.clear()
        assert len(s) == 1

    def test_max_turns_eviction(self):
        s = ConversationSession(max_turns=3)
        for i in range(5):
            s.add_turn(f"q{i}", f"a{i}", 0.8)
        assert len(s) == 3
        assert s.turns[0].response == "a2"

    def test_context_text(self):
        s = ConversationSession()
        s.add_turn("q1", "Response one.", 0.9)
        s.add_turn("q2", "Response two.", 0.8)
        assert s.context_text == "Response one. Response two."

    def test_context_text_empty(self):
        s = ConversationSession()
        assert s.context_text == ""

    def test_len(self):
        s = ConversationSession()
        assert len(s) == 0
        s.add_turn("q", "a", 0.5)
        assert len(s) == 1

    def test_invalid_max_turns(self):
        with pytest.raises(ValueError, match="max_turns"):
            ConversationSession(max_turns=0)

    def test_thread_safety(self):
        s = ConversationSession(max_turns=100)
        barrier = threading.Barrier(4)

        def worker(tid):
            barrier.wait()
            for i in range(25):
                s.add_turn(f"t{tid}-q{i}", f"t{tid}-a{i}", 0.5)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(s) == 100

    def test_turn_index_monotonic(self):
        s = ConversationSession()
        for i in range(5):
            turn = s.add_turn(f"q{i}", f"a{i}", 0.7)
            assert turn.turn_index == i


class TestScorerSessionIntegration:
    def test_review_with_session(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        session = ConversationSession()

        approved1, score1 = scorer.review(
            "What is AI?",
            "AI is technology.",
            session=session,
        )
        assert len(session) == 1
        assert score1.cross_turn_divergence is None  # no prior turns

        approved2, score2 = scorer.review(
            "Tell me more.",
            "AI uses data.",
            session=session,
        )
        assert len(session) == 2

    def test_review_without_session(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        approved, score = scorer.review("test", "test response")
        assert score.cross_turn_divergence is None

    def test_cross_turn_contradiction_detection(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(
            threshold=0.3,
            use_nli=False,
            scorer_backend="lite",
        )
        session = ConversationSession()

        scorer.review("q1", "The sky is blue and clear today.", session=session)
        _, score2 = scorer.review(
            "q2",
            "Quantum physics defines the collapse of wave functions.",
            session=session,
        )
        assert len(session) == 2
