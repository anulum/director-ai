# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Public API Freeze Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import importlib
import warnings

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


class TestDeprecationWarnings:
    def test_calculate_factual_entropy_warns(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer.calculate_factual_entropy("prompt", "action")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "calculate_factual_entropy" in str(w[0].message)

    def test_calculate_logical_entropy_warns(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer.calculate_logical_entropy("prompt", "action")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_simulate_future_state_warns(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer.simulate_future_state("prompt", "action")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_review_action_warns(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer.review_action("prompt", "action")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_process_query_warns(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent.process_query("What is AI?")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "process_query" in str(w[0].message)
