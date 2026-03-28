# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LlamaIndex Integration Tests with Real SDK
"""Tests that import real llama_index.core and verify integration types.

These tests run only when llama-index-core is installed (CI extras matrix).
They verify that DirectorAIPostprocessor works correctly with real
LlamaIndex NodeWithScore and QueryBundle objects.
"""

from __future__ import annotations


class TestPostprocessorWithRealNodes:
    def test_postprocess_real_node_with_score(self):
        pytest = __import__("pytest")
        pytest.importorskip("llama_index.core")
        from llama_index.core.schema import NodeWithScore, TextNode

        from director_ai.integrations.llamaindex import DirectorAIPostprocessor

        proc = DirectorAIPostprocessor(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
            threshold=0.01,
        )

        node = NodeWithScore(
            node=TextNode(text="Paris is the capital of France."),
            score=0.95,
        )
        result = proc.postprocess_nodes([node])
        assert len(result) >= 1
        assert hasattr(result[0], "metadata")
        if "director_ai_score" in result[0].metadata:
            assert result[0].metadata["director_ai_approved"] is True

    def test_postprocess_filters_low_coherence(self):
        pytest = __import__("pytest")
        pytest.importorskip("llama_index.core")
        from llama_index.core.schema import NodeWithScore, TextNode

        from director_ai.integrations.llamaindex import DirectorAIPostprocessor

        proc = DirectorAIPostprocessor(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
            threshold=0.99,
        )

        node = NodeWithScore(
            node=TextNode(text="Something unrelated to the knowledge base."),
            score=0.5,
        )
        result = proc.postprocess_nodes([node])
        assert len(result) == 0

    def test_postprocess_with_query_bundle(self):
        pytest = __import__("pytest")
        pytest.importorskip("llama_index.core")
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

        from director_ai.integrations.llamaindex import DirectorAIPostprocessor

        proc = DirectorAIPostprocessor(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
            threshold=0.01,
        )

        node = NodeWithScore(
            node=TextNode(text="Paris is the capital of France."),
            score=0.9,
        )
        query = QueryBundle(query_str="What is the capital of France?")
        result = proc.postprocess_nodes([node], query_bundle=query)
        assert len(result) >= 1

    def test_validate_response_with_real_types(self):
        pytest = __import__("pytest")
        pytest.importorskip("llama_index.core")

        from director_ai.integrations.llamaindex import DirectorAIPostprocessor

        proc = DirectorAIPostprocessor(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
        )

        approved, cs = proc.validate_response(
            "What is the capital of France?",
            "Paris is the capital of France.",
        )
        assert isinstance(approved, bool)
        assert hasattr(cs, "score")

    def test_check_returns_dict(self):
        pytest = __import__("pytest")
        pytest.importorskip("llama_index.core")

        from director_ai.integrations.llamaindex import DirectorAIPostprocessor

        proc = DirectorAIPostprocessor(
            facts={"policy": "30-day refund policy."},
            use_nli=False,
        )
        result = proc.check("refund policy?", "30-day refund policy.")
        assert isinstance(result, dict)
        assert "approved" in result
        assert "score" in result
        assert "coherence_score" in result
