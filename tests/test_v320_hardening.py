# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — v3.2.0 Hardening Tests

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from director_ai.core.batch import BatchProcessor
from director_ai.core.config import DirectorConfig
from director_ai.core.knowledge import GroundTruthStore
from director_ai.core.lite_scorer import LiteScorer
from director_ai.core.scorer import CoherenceScorer
from director_ai.core.vector_store import InMemoryBackend

# â”€â”€ Item 1: quickstart scaffolding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestQuickstartScaffolding:
    def test_generated_guard_is_valid_python(self, tmp_path, monkeypatch):
        from director_ai.cli import _cmd_quickstart

        monkeypatch.chdir(tmp_path)
        _cmd_quickstart([])
        guard_py = tmp_path / "director_guard" / "guard.py"
        assert guard_py.exists()
        source = guard_py.read_text(encoding="utf-8")
        assert "asyncio.run" not in source
        compile(source, str(guard_py), "exec")


# â”€â”€ Item 2: process_batch_async â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestBatchAsync:
    def test_process_batch_async_ordering(self):
        agent = MagicMock()
        agent.process.side_effect = lambda p, **kw: MagicMock(
            halted=False,
            coherence=MagicMock(score=0.9),
        )
        bp = BatchProcessor(agent, max_concurrency=2)
        result = asyncio.run(bp.process_batch_async(["p1", "p2", "p3"]))
        assert result.total == 3
        assert result.succeeded == 3
        assert result.failed == 0
        assert len(result.results) == 3

    def test_review_batch_async_ordering(self):
        scorer = MagicMock()
        scorer.review.return_value = (
            True,
            MagicMock(score=0.85),
        )
        bp = BatchProcessor(scorer, max_concurrency=2)
        items = [("p1", "r1"), ("p2", "r2")]
        result = asyncio.run(bp.review_batch_async(items))
        assert result.total == 2
        assert result.succeeded == 2

    def test_process_batch_async_partial_failure(self):
        agent = MagicMock()
        call_count = 0

        def side_effect(p, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("boom")
            return MagicMock(halted=False, coherence=MagicMock(score=0.9))

        agent.process.side_effect = side_effect
        bp = BatchProcessor(agent, max_concurrency=1)
        result = asyncio.run(bp.process_batch_async(["a", "b", "c"]))
        assert result.succeeded == 2
        assert result.failed == 1
        assert len(result.errors) == 1


# â”€â”€ Item 4: async VectorBackend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAsyncVectorBackend:
    def test_aadd_aquery(self):
        backend = InMemoryBackend()

        async def _test():
            await backend.aadd("d1", "the sky is blue")
            results = await backend.aquery("sky blue", n_results=1)
            assert len(results) == 1
            assert results[0]["id"] == "d1"

        asyncio.run(_test())


# â”€â”€ Item 6: LiteScorer.review() â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestLiteScorerReview:
    def test_review_returns_tuple(self):
        scorer = LiteScorer()
        approved, cs = scorer.review("The sky is blue", "The sky is blue")
        assert isinstance(approved, bool)
        assert approved is True
        assert cs.score > 0.5

    def test_review_contradiction(self):
        scorer = LiteScorer()
        approved, cs = scorer.review(
            "The sky is blue",
            "Elephants never forget anything",
        )
        assert cs.score < 0.8

    def test_review_custom_threshold(self):
        scorer = LiteScorer()
        _, cs_low = scorer.review("abc", "xyz", threshold=0.0)
        assert cs_low.approved is True
        _, cs_high = scorer.review("abc", "xyz", threshold=1.0)
        assert cs_high.approved is False


# â”€â”€ Item 7: config validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestConfigValidation:
    def test_reranker_empty_model_raises(self):
        with pytest.raises(ValueError, match="reranker_model"):
            DirectorConfig(reranker_enabled=True, reranker_model="")

    def test_embedding_model_empty_raises(self):
        with pytest.raises(ValueError, match="embedding_model"):
            DirectorConfig(vector_backend="sentence-transformer", embedding_model="")

    def test_unknown_yaml_keys_warned(self, tmp_path, caplog):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            '{"coherence_threshold": 0.7, "typo_key": 42}',
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="DirectorAI.Config"):
            cfg = DirectorConfig.from_yaml(str(yaml_file))
        assert cfg.coherence_threshold == 0.7
        assert "typo_key" in caplog.text


# â”€â”€ Item 8a: cross-turn divergence e2e â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestCrossTurnDivergence:
    def test_session_adds_turns(self):
        from director_ai.core.session import ConversationSession

        store = GroundTruthStore()
        store.add("sky", "The sky is blue")
        scorer = CoherenceScorer(threshold=0.5, use_nli=False, ground_truth_store=store)
        session = ConversationSession()

        scorer.review("What is the sky?", "The sky is blue.", session=session)
        assert len(session) == 1

        scorer.review("What is the sky?", "The sky is blue.", session=session)
        assert len(session) == 2


# â”€â”€ Item 8c: build_store sentence-transformer branch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestBuildStoreSentenceTransformer:
    def test_build_store_sentence_transformer_branch(self):
        from unittest.mock import patch

        mock_st = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            from director_ai.core.vector_store import (
                SentenceTransformerBackend,
                VectorGroundTruthStore,
            )

            with patch.object(
                SentenceTransformerBackend,
                "__init__",
                lambda self, **kw: None,
            ):
                cfg = DirectorConfig(vector_backend="sentence-transformer")
                store = cfg.build_store()
                assert isinstance(store, VectorGroundTruthStore)
