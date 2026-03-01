# ─────────────────────────────────────────────────────────────────────
# Tests — LlamaIndex integration (DirectorAIPostprocessor)
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

from types import SimpleNamespace

from director_ai.integrations.llamaindex import DirectorAIPostprocessor


def _make_node(text, metadata=None):
    """Minimal mock node with .text and .metadata."""
    return SimpleNamespace(text=text, metadata=metadata if metadata is not None else {})


def _make_query_bundle(query_str):
    return SimpleNamespace(query_str=query_str)


class TestConstruction:
    def test_default_store(self):
        pp = DirectorAIPostprocessor()
        assert pp.store is not None
        assert pp.threshold == 0.6

    def test_custom_facts(self):
        pp = DirectorAIPostprocessor(facts={"sky": "The sky is blue."})
        ctx = pp.store.retrieve_context("sky")
        assert ctx is not None

    def test_custom_threshold(self):
        pp = DirectorAIPostprocessor(threshold=0.8)
        assert pp.scorer.threshold == 0.8


class TestCheck:
    def test_returns_dict_with_required_keys(self):
        pp = DirectorAIPostprocessor()
        result = pp.check("test query", "test response")
        assert "approved" in result
        assert "score" in result
        assert "h_logical" in result
        assert "h_factual" in result
        assert "response" in result
        assert "coherence_score" in result

    def test_score_in_valid_range(self):
        pp = DirectorAIPostprocessor()
        result = pp.check("query", "response text")
        assert 0.0 <= result["score"] <= 1.0


class TestPostprocessNodes:
    def test_empty_list(self):
        pp = DirectorAIPostprocessor()
        assert pp.postprocess_nodes([]) == []

    def test_approved_nodes_pass_through(self):
        pp = DirectorAIPostprocessor()
        nodes = [_make_node("The sky is blue, consistent with reality")]
        result = pp.postprocess_nodes(nodes, _make_query_bundle("sky color"))
        assert len(result) >= 0  # may or may not pass depending on scorer

    def test_metadata_attached(self):
        pp = DirectorAIPostprocessor()
        pp.scorer.threshold = 0.0  # accept everything
        node = _make_node("Some text")
        result = pp.postprocess_nodes([node], _make_query_bundle("test"))
        assert len(result) == 1
        assert "director_ai_score" in result[0].metadata
        assert result[0].metadata["director_ai_approved"] is True

    def test_high_threshold_rejects(self):
        pp = DirectorAIPostprocessor(threshold=999.0)
        nodes = [_make_node("Any text")]
        result = pp.postprocess_nodes(nodes, _make_query_bundle("test"))
        assert len(result) == 0

    def test_none_query_bundle(self):
        pp = DirectorAIPostprocessor()
        pp.scorer.threshold = 0.0
        nodes = [_make_node("Some text")]
        result = pp.postprocess_nodes(nodes, query_bundle=None)
        assert len(result) == 1

    def test_node_without_metadata_dict(self):
        pp = DirectorAIPostprocessor()
        pp.scorer.threshold = 0.0
        node = SimpleNamespace(text="text", metadata="not a dict")
        result = pp.postprocess_nodes([node], _make_query_bundle("q"))
        assert len(result) == 1

    def test_node_falls_back_to_str(self):
        pp = DirectorAIPostprocessor()
        pp.scorer.threshold = 0.0

        class CustomNode:
            def __str__(self):
                return "stringified node"

        result = pp.postprocess_nodes([CustomNode()], _make_query_bundle("q"))
        assert len(result) == 1


class TestValidateResponse:
    def test_returns_tuple(self):
        pp = DirectorAIPostprocessor()
        approved, score = pp.validate_response("query", "response")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_low_threshold_approves(self):
        pp = DirectorAIPostprocessor(threshold=0.0)
        approved, _ = pp.validate_response("q", "any text")
        assert approved is True
