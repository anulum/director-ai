# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Tests for CostAnalyser integration with scorer
"""Multi-angle tests for CostAnalyser pipeline integration.

Covers: config field, build_scorer wiring, cost_callback in LLMJudge,
CLI cost-report command, report generation, thread safety.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from director_ai.cli import main
from director_ai.compliance.cost_analyser import CostAnalyser
from director_ai.core.scoring._llm_judge import LLMJudge

# ── CostAnalyser unit tests ─────────────────────────────────────────


class TestCostAnalyserUnit:
    """Unit tests for CostAnalyser."""

    def test_default_pricing(self):
        """Default pricing includes major models."""
        a = CostAnalyser()
        assert "gpt-4o" in a._pricing
        assert "claude-3-5-sonnet" in a._pricing

    def test_record_and_report(self):
        """record() accumulates and report() summarises."""
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=1000, output_tokens=200)
        a.record("gpt-4o", input_tokens=500, output_tokens=100)

        report = a.report()
        assert report["total_tokens"] == 1800
        assert report["models"]["gpt-4o"]["call_count"] == 2
        assert report["total_cost"] > 0

    def test_record_with_agent_id(self):
        """agent_id creates separate keys."""
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=100, agent_id="researcher")
        a.record("gpt-4o", input_tokens=200, agent_id="coder")

        report = a.report()
        assert "gpt-4o::researcher" in report["models"]
        assert "gpt-4o::coder" in report["models"]

    def test_estimate_cost(self):
        """estimate_cost returns correct CHF value."""
        a = CostAnalyser()
        cost = a.estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = (1000 / 1000 * 0.0025) + (500 / 1000 * 0.01)
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_zero_cost(self):
        """Unknown model returns zero cost."""
        a = CostAnalyser()
        assert a.estimate_cost("unknown-model", 1000, 500) == 0.0

    def test_add_pricing(self):
        """Custom pricing is applied correctly."""
        a = CostAnalyser()
        a.add_pricing("custom-model", input_per_1k=0.001, output_per_1k=0.002)
        cost = a.estimate_cost("custom-model", 1000, 1000)
        assert abs(cost - 0.003) < 1e-10

    def test_reset_clears_records(self):
        """reset() clears all accumulated data."""
        a = CostAnalyser()
        a.record("gpt-4o", input_tokens=100)
        a.reset()
        report = a.report()
        assert report["total_tokens"] == 0
        assert len(report["models"]) == 0

    def test_thread_safety(self):
        """Concurrent records don't corrupt data."""
        import threading

        a = CostAnalyser()
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    a.record("gpt-4o", input_tokens=10, output_tokens=5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        report = a.report()
        assert report["models"]["gpt-4o"]["call_count"] == 400
        assert report["total_tokens"] == 400 * 15


# ── LLMJudge cost_callback ──────────────────────────────────────────


class TestLLMJudgeCostCallback:
    """Tests for cost_callback wiring in LLMJudge."""

    def test_callback_default_none(self):
        """cost_callback is None by default."""
        j = LLMJudge()
        assert j._cost_callback is None

    def test_callback_set_in_constructor(self):
        """cost_callback can be set via constructor."""
        cb = MagicMock()
        j = LLMJudge(cost_callback=cb)
        assert j._cost_callback is cb

    def test_callback_invoked_on_openai_call(self):
        """cost_callback is called with usage data after OpenAI call."""
        pytest.importorskip("openai")
        cb = MagicMock()
        j = LLMJudge(provider="openai", model="gpt-4o-mini", cost_callback=cb)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 150
        mock_usage.completion_tokens = 20

        mock_choice = MagicMock()
        mock_choice.message.content = '{"verdict": "YES", "confidence": 85}'

        mock_result = MagicMock()
        mock_result.choices = [mock_choice]
        mock_result.usage = mock_usage

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_result
            mock_openai.return_value = mock_client

            j._call_llm_judge("gpt-4o-mini", "test prompt", 0.5)

        cb.assert_called_once_with("gpt-4o-mini", 150, 20)

    def test_callback_not_invoked_when_none(self):
        """No error when cost_callback is None."""
        pytest.importorskip("openai")
        j = LLMJudge(provider="openai", model="gpt-4o-mini")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 10

        mock_choice = MagicMock()
        mock_choice.message.content = '{"verdict": "YES", "confidence": 90}'

        mock_result = MagicMock()
        mock_result.choices = [mock_choice]
        mock_result.usage = mock_usage

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_result
            mock_openai.return_value = mock_client

            result = j._call_llm_judge("gpt-4o-mini", "test", 0.5)
            assert result is not None


# ── Config integration ───────────────────────────────────────────────


class TestCostTrackingConfig:
    """Tests for cost_tracking_enabled config field."""

    def test_default_disabled(self):
        """Cost tracking is off by default."""
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig()
        assert cfg.cost_tracking_enabled is False

    def test_enabled_attaches_analyser(self):
        """When enabled, build_scorer attaches CostAnalyser."""
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(cost_tracking_enabled=True)
        scorer = cfg.build_scorer()
        assert hasattr(scorer, "_cost_analyser")
        assert isinstance(scorer._cost_analyser, CostAnalyser)

    def test_disabled_no_analyser(self):
        """When disabled, _cost_analyser stays None."""
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(cost_tracking_enabled=False)
        scorer = cfg.build_scorer()
        assert scorer._cost_analyser is None

    def test_callback_wired_to_judge(self):
        """cost_callback on judge is set when cost tracking enabled."""
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(cost_tracking_enabled=True)
        scorer = cfg.build_scorer()
        assert scorer._judge._cost_callback is not None


# ── CLI cost-report ──────────────────────────────────────────────────


class TestCostReportCLI:
    """Tests for 'director-ai cost-report' subcommand."""

    def test_cost_report_in_help(self, capsys):
        """cost-report appears in help output."""
        main([])
        captured = capsys.readouterr()
        assert "cost-report" in captured.out

    def test_cost_report_disabled_exits(self, capsys):
        """Exits with error when cost tracking is disabled."""
        with pytest.raises(SystemExit) as exc_info:
            main(["cost-report"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower()
