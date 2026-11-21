# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Public API Freeze Tests
"""Multi-angle tests for public API stability pipeline."""

from __future__ import annotations

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
    "director_ai.model_lifecycle",
]


class TestModulesHaveAll:
    """Every listed module declares an importable ``__all__``."""

    @pytest.mark.parametrize("module_name", _MODULES_WITH_ALL)
    def test_module_has_all(self, module_name: str) -> None:
        """The module defines ``__all__`` as a list or tuple."""
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "__all__"), f"{module_name} missing __all__"
        assert isinstance(mod.__all__, list | tuple)

    @pytest.mark.parametrize("module_name", _MODULES_WITH_ALL)
    def test_all_names_importable(self, module_name: str) -> None:
        """Every name in ``__all__`` resolves on the module."""
        mod = importlib.import_module(module_name)
        for name in mod.__all__:
            assert hasattr(mod, name), (
                f"{module_name}.{name} in __all__ but not importable"
            )


class TestNewExports:
    """Names added to the package root stay importable."""

    def test_conversation_session_importable(self) -> None:
        """ConversationSession and Turn are exported."""
        from director_ai import ConversationSession, Turn

        assert ConversationSession is not None
        assert Turn is not None

    def test_lite_scorer_importable(self) -> None:
        """LiteScorer is exported."""
        from director_ai import LiteScorer

        assert LiteScorer is not None

    def test_scorer_backend_importable(self) -> None:
        """The scorer backend registry is exported."""
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

    def test_sharded_nli_importable(self) -> None:
        """ShardedNLIScorer is exported."""
        from director_ai import ShardedNLIScorer

        assert ShardedNLIScorer is not None

    def test_score_in_all(self) -> None:
        """The ``score`` helper is exported and callable."""
        import director_ai

        assert "score" in director_ai.__all__
        assert callable(director_ai.score)


class TestDeprecated1xRemoved:
    """Removed 1.x methods stay removed."""

    def test_scorer_has_no_factual_entropy(self) -> None:
        """CoherenceScorer has no calculate_factual_entropy."""
        from director_ai.core.scoring.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        assert not hasattr(scorer, "calculate_factual_entropy")

    def test_scorer_has_no_review_action(self) -> None:
        """CoherenceScorer has no review_action."""
        from director_ai.core.scoring.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        assert not hasattr(scorer, "review_action")

    def test_agent_has_no_process_query(self) -> None:
        """CoherenceAgent has no process_query."""
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()
        assert not hasattr(agent, "process_query")
