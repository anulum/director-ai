# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Public API Freeze Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import importlib

import pytest

_MODULES_WITH_ALL = [
    "director_ai",
    "director_ai.core",
    "director_ai.core.scorer",
    "director_ai.core.nli",
    "director_ai.core.types",
    "director_ai.core.kernel",
    "director_ai.core.streaming",
    "director_ai.core.async_streaming",
    "director_ai.core.config",
    "director_ai.core.knowledge",
    "director_ai.core.cache",
    "director_ai.core.session",
    "director_ai.core.backends",
    "director_ai.core.lite_scorer",
    "director_ai.core.sharded_nli",
    "director_ai.core.sanitizer",
    "director_ai.core.agent",
    "director_ai.server",
]


class TestModulesHaveAll:
    @pytest.mark.parametrize("module_name", _MODULES_WITH_ALL)
    def test_module_has_all(self, module_name):
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "__all__"), f"{module_name} missing __all__"
        assert isinstance(mod.__all__, list | tuple)

    @pytest.mark.parametrize("module_name", _MODULES_WITH_ALL)
    def test_all_names_importable(self, module_name):
        mod = importlib.import_module(module_name)
        for name in mod.__all__:
            assert hasattr(mod, name), (
                f"{module_name}.{name} in __all__ but not importable"
            )


class TestNewExports:
    def test_conversation_session_importable(self):
        from director_ai import ConversationSession, Turn

        assert ConversationSession is not None
        assert Turn is not None

    def test_lite_scorer_importable(self):
        from director_ai import LiteScorer

        assert LiteScorer is not None

    def test_scorer_backend_importable(self):
        from director_ai import (
            ScorerBackend,
            get_backend,
            list_backends,
            register_backend,
        )

        assert ScorerBackend is not None
        assert callable(register_backend)
        assert callable(get_backend)
        assert callable(list_backends)

    def test_sharded_nli_importable(self):
        from director_ai import ShardedNLIScorer

        assert ShardedNLIScorer is not None

    def test_score_in_all(self):
        import director_ai

        assert "score" in director_ai.__all__
        assert callable(director_ai.score)


class TestDeprecated1xRemoved:
    def test_scorer_has_no_factual_entropy(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        assert not hasattr(scorer, "calculate_factual_entropy")

    def test_scorer_has_no_review_action(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        assert not hasattr(scorer, "review_action")

    def test_agent_has_no_process_query(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()
        assert not hasattr(agent, "process_query")
