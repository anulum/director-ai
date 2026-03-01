# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — CrewAI Integration Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.integrations.crewai import DirectorAITool


@pytest.mark.consumer
class TestDirectorAITool:
    def test_run_with_pipe_separator(self):
        tool = DirectorAITool(facts={"sky color": "The sky is blue."}, use_nli=False)
        result = tool._run("What color is the sky? | The sky is blue.")
        assert "APPROVED" in result
        assert "Coherence:" in result

    def test_run_without_pipe(self):
        tool = DirectorAITool(facts={"sky color": "The sky is blue."}, use_nli=False)
        result = tool._run("The sky is blue.")
        assert "Coherence:" in result

    def test_run_rejected(self):
        tool = DirectorAITool(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        result = tool._run(
            "What color is the sky? | Mars has two moons named Phobos and Deimos."
        )
        assert "REJECTED" in result

    def test_run_alias(self):
        tool = DirectorAITool(facts={"sky color": "The sky is blue."}, use_nli=False)
        result_run = tool.run("sky | The sky is blue.")
        result_internal = tool._run("sky | The sky is blue.")
        assert result_run == result_internal

    def test_check_returns_dict(self):
        tool = DirectorAITool(facts={"sky color": "The sky is blue."}, use_nli=False)
        result = tool.check("What color is the sky?", "The sky is blue.")
        assert isinstance(result, dict)
        assert "approved" in result
        assert "score" in result
        assert "h_logical" in result
        assert "h_factual" in result
        assert "warning" in result

    def test_check_approved(self):
        tool = DirectorAITool(facts={"sky color": "The sky is blue."}, use_nli=False)
        result = tool.check("What color is the sky?", "The sky is blue.")
        assert result["approved"] is True

    def test_check_rejected(self):
        tool = DirectorAITool(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        result = tool.check(
            "What color is the sky?",
            "Mars has two moons named Phobos and Deimos.",
        )
        assert result["approved"] is False

    def test_name_and_description(self):
        tool = DirectorAITool()
        assert tool.name == "director_ai_fact_check"
        assert len(tool.description) > 0

    def test_custom_threshold(self):
        tool = DirectorAITool(
            facts={"sky color": "The sky is blue."},
            threshold=0.99,
            use_nli=False,
        )
        result = tool.check("What color is the sky?", "The sky is blue.")
        assert result["score"] < 0.99 or result["approved"] is True

    def test_no_facts_still_works(self):
        tool = DirectorAITool(use_nli=False)
        result = tool.check("anything", "some response")
        assert "score" in result

    def test_custom_store(self):
        from director_ai.core import GroundTruthStore

        store = GroundTruthStore()
        store.add("capital", "Paris is the capital of France.")
        tool = DirectorAITool(store=store, use_nli=False)
        result = tool.check("What is the capital?", "Paris is the capital of France.")
        assert result["approved"] is True
