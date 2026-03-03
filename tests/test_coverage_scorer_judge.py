"""Coverage tests for scorer.py — _parse_judge_reply, LLM judge with providers."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from director_ai.core import CoherenceScorer


class TestParseJudgeReply:
    def test_json_verdict_yes(self):
        assert CoherenceScorer._parse_judge_reply('{"verdict": "YES"}') is True

    def test_json_verdict_no(self):
        assert CoherenceScorer._parse_judge_reply('{"verdict": "NO"}') is False

    def test_json_invalid_fallback_yes(self):
        assert CoherenceScorer._parse_judge_reply("YES the answer is correct") is True

    def test_json_invalid_fallback_no(self):
        assert CoherenceScorer._parse_judge_reply("NO it does not match") is False

    def test_json_empty(self):
        assert CoherenceScorer._parse_judge_reply("") is False

    def test_json_verdict_case_insensitive(self):
        assert CoherenceScorer._parse_judge_reply('{"verdict": "yes"}') is True


class TestLLMJudgeOpenAI:
    def test_openai_judge_agree(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
        )
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        choice = MagicMock()
        choice.message.content = '{"verdict": "YES"}'
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = scorer._llm_judge_check("q", "a", 0.5)
            assert result != 0.5

    def test_openai_judge_api_error(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
        )
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API fail")

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = scorer._llm_judge_check("q", "a", 0.5)
            assert result == 0.5


class TestLLMJudgeAnthropic:
    def test_anthropic_judge(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="anthropic",
        )
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        content_block = MagicMock()
        content_block.text = '{"verdict": "NO"}'
        mock_client.messages.create.return_value = MagicMock(content=[content_block])

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = scorer._llm_judge_check("q", "a", 0.5)
            assert result != 0.5


class TestScorerEscalation:
    def test_hybrid_backend_escalation(self):
        from director_ai.core import GroundTruthStore

        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(
            use_nli=False,
            ground_truth_store=store,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
            scorer_backend="hybrid",
        )
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        choice = MagicMock()
        choice.message.content = '{"verdict": "YES"}'
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])

        with patch.dict(sys.modules, {"openai": mock_openai}):
            div, ev = scorer.calculate_factual_divergence_with_evidence(
                "sky",
                "The sky is blue.",
            )
            assert 0.0 <= div <= 1.0
