# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Integration Module Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest


@pytest.mark.consumer
class TestLangChainGuard:
    def test_check_approved(self):
        from director_ai.integrations.langchain import DirectorAIGuard

        guard = DirectorAIGuard(
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guard.check("What color is the sky?", "The sky is blue.")
        assert result["approved"] is True
        assert result["score"] > 0.5

    def test_check_blocked(self):
        from director_ai.integrations.langchain import DirectorAIGuard

        guard = DirectorAIGuard(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        result = guard.check("What color is the sky?", "The sky is green.")
        assert result["approved"] is False

    def test_raise_on_fail(self):
        from director_ai.integrations.langchain import (
            DirectorAIGuard,
            HallucinationError,
        )

        guard = DirectorAIGuard(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
            raise_on_fail=True,
        )
        with pytest.raises(HallucinationError):
            guard.check("What color is the sky?", "The sky is green.")

    def test_invoke_dict(self):
        from director_ai.integrations.langchain import DirectorAIGuard

        guard = DirectorAIGuard(
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guard.invoke({
            "query": "What color is the sky?",
            "response": "The sky is blue.",
        })
        assert result["approved"] is True

    def test_invoke_string(self):
        from director_ai.integrations.langchain import DirectorAIGuard

        guard = DirectorAIGuard(
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guard.invoke(
            "The sky is blue.",
            query="What color is the sky?",
        )
        assert result["approved"] is True


@pytest.mark.consumer
class TestLlamaIndexPostprocessor:
    def test_check_approved(self):
        from director_ai.integrations.llamaindex import (
            DirectorAIPostprocessor,
        )

        pp = DirectorAIPostprocessor(
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = pp.check("What color is the sky?", "The sky is blue.")
        assert result["approved"] is True

    def test_check_blocked(self):
        from director_ai.integrations.llamaindex import (
            DirectorAIPostprocessor,
        )

        pp = DirectorAIPostprocessor(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        result = pp.check("What color is the sky?", "The sky is green.")
        assert result["approved"] is False

    def test_validate_response(self):
        from director_ai.integrations.llamaindex import (
            DirectorAIPostprocessor,
        )

        pp = DirectorAIPostprocessor(
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        approved, score = pp.validate_response(
            "What color is the sky?", "The sky is blue."
        )
        assert approved is True
        assert score.score > 0.5

    def test_postprocess_nodes_filters(self):
        from director_ai.integrations.llamaindex import (
            DirectorAIPostprocessor,
        )

        pp = DirectorAIPostprocessor(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )

        class FakeNode:
            def __init__(self, text):
                self.text = text
                self.metadata = {}

        nodes = [
            FakeNode("The sky is blue on a clear day."),
            FakeNode("The sky is green, obviously."),
        ]

        class FakeBundle:
            query_str = "What color is the sky?"

        filtered = pp.postprocess_nodes(nodes, FakeBundle())
        assert len(filtered) == 1
        assert "director_ai_score" in filtered[0].metadata


@pytest.mark.consumer
@pytest.mark.asyncio
class TestAsyncScorer:
    async def test_areview(self):
        from director_ai.core import CoherenceScorer, GroundTruthStore

        store = GroundTruthStore()
        store.add("sky", "blue")
        scorer = CoherenceScorer(
            threshold=0.6,
            ground_truth_store=store,
            use_nli=False,
        )
        approved, score = await scorer.areview(
            "What color is the sky?",
            "The sky is blue.",
        )
        assert approved is True
        assert score.score > 0.5

    async def test_nli_ascore(self):
        from director_ai.core.nli import NLIScorer

        nli = NLIScorer(use_model=False)
        score = await nli.ascore("test", "consistent with reality")
        assert score == pytest.approx(0.1)
