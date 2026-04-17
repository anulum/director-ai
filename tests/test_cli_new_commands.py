# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Tests for kb-health, wizard, and compliance HTML CLI commands
"""Multi-angle tests for new CLI subcommands.

Covers: kb-health diagnostics, wizard (CLI + Gradio fallback),
compliance --format html, arg parsing, error paths, integration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from director_ai.cli import main

# ── kb-health ────────────────────────────────────────────────────────


class TestKBHealthCLI:
    """Tests for 'director-ai kb-health' subcommand."""

    def test_kb_health_in_commands(self):
        """kb-health is registered in the CLI commands dict."""
        from director_ai._cli_verify import _cmd_kb_health

        assert callable(_cmd_kb_health)

    def test_kb_health_in_help(self, capsys):
        """kb-health appears in help output."""
        main([])
        captured = capsys.readouterr()
        assert "kb-health" in captured.out

    @patch("director_ai.core.config.DirectorConfig.build_store")
    @patch("director_ai.core.config.DirectorConfig.from_env")
    def test_kb_health_healthy_store(self, mock_from_env, mock_build_store, capsys):
        """Healthy store exits 0 with HEALTHY summary."""
        mock_store = MagicMock()
        mock_store.backend.count.return_value = 10
        mock_store.retrieve_context.return_value = ["some result"]

        mock_cfg = MagicMock()
        mock_cfg.build_store.return_value = mock_store
        mock_from_env.return_value = mock_cfg

        with pytest.raises(SystemExit) as exc_info:
            main(["kb-health"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "HEALTHY" in captured.out

    @patch("director_ai.core.config.DirectorConfig.build_store")
    @patch("director_ai.core.config.DirectorConfig.from_env")
    def test_kb_health_unhealthy_store(self, mock_from_env, mock_build_store, capsys):
        """Empty store exits 1 with UNHEALTHY summary."""
        mock_store = MagicMock()
        mock_store.backend.count.return_value = 0
        mock_store.retrieve_context.return_value = []
        mock_store.facts = {}

        mock_cfg = MagicMock()
        mock_cfg.build_store.return_value = mock_store
        mock_from_env.return_value = mock_cfg

        with pytest.raises(SystemExit) as exc_info:
            main(["kb-health"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "UNHEALTHY" in captured.out

    @patch("director_ai.core.config.DirectorConfig.build_store")
    @patch("director_ai.core.config.DirectorConfig.from_env")
    def test_kb_health_custom_args(self, mock_from_env, mock_build_store, capsys):
        """--min-docs and --max-latency args are parsed correctly."""
        mock_store = MagicMock()
        mock_store.backend.count.return_value = 50
        mock_store.retrieve_context.return_value = ["result"]

        mock_cfg = MagicMock()
        mock_cfg.build_store.return_value = mock_store
        mock_from_env.return_value = mock_cfg

        with pytest.raises(SystemExit) as exc_info:
            main(["kb-health", "--min-docs", "5", "--max-latency", "200"])

        assert exc_info.value.code == 0

    @patch("director_ai.core.config.DirectorConfig.build_store")
    @patch("director_ai.core.config.DirectorConfig.from_env")
    def test_kb_health_shows_issues(self, mock_from_env, mock_build_store, capsys):
        """Issues are printed when store is unhealthy."""
        mock_store = MagicMock()
        mock_store.backend.count.return_value = 0
        mock_store.retrieve_context.side_effect = RuntimeError("broken")

        mock_cfg = MagicMock()
        mock_cfg.build_store.return_value = mock_store
        mock_from_env.return_value = mock_cfg

        with pytest.raises(SystemExit) as exc_info:
            main(["kb-health", "--min-docs", "10"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ISSUE" in captured.out


# ── wizard ───────────────────────────────────────────────────────────


class TestWizardCLI:
    """Tests for 'director-ai wizard' subcommand."""

    def test_wizard_in_commands(self):
        """wizard is registered in the CLI commands dict."""
        from director_ai._cli_verify import _cmd_wizard

        assert callable(_cmd_wizard)

    def test_wizard_in_help(self, capsys):
        """wizard appears in help output."""
        main([])
        captured = capsys.readouterr()
        assert "wizard" in captured.out

    @patch("director_ai.ui.config_wizard.launch_cli")
    def test_wizard_cli_mode(self, mock_launch, capsys):
        """--cli flag triggers CLI mode."""
        mock_launch.return_value = "coherence_threshold: 0.5\n"
        main(["wizard", "--cli"])
        mock_launch.assert_called_once()

    @patch("director_ai.ui.config_wizard.launch_cli")
    def test_wizard_cli_mode_with_output(self, mock_launch, capsys):
        """--cli --output writes YAML to file."""
        mock_launch.return_value = "coherence_threshold: 0.7\nuse_nli: true\n"

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            out_path = f.name

        try:
            main(["wizard", "--cli", "--output", out_path])
            content = Path(out_path).read_text()
            assert "coherence_threshold" in content
        finally:
            Path(out_path).unlink(missing_ok=True)

    @patch("director_ai.ui.config_wizard.launch_gradio")
    def test_wizard_gradio_mode(self, mock_launch, capsys):
        """Default mode launches Gradio."""
        main(["wizard"])
        mock_launch.assert_called_once_with(port=7860, share=False)

    @patch("director_ai.ui.config_wizard.launch_gradio")
    def test_wizard_gradio_port(self, mock_launch, capsys):
        """--port flag is passed to Gradio."""
        main(["wizard", "--port", "9090"])
        mock_launch.assert_called_once_with(port=9090, share=False)

    @patch("director_ai.ui.config_wizard.launch_gradio")
    def test_wizard_gradio_share(self, mock_launch, capsys):
        """--share flag is passed to Gradio."""
        main(["wizard", "--share"])
        mock_launch.assert_called_once_with(port=7860, share=True)

    @patch(
        "director_ai.ui.config_wizard.launch_gradio",
        side_effect=ImportError("no gradio"),
    )
    @patch("director_ai.ui.config_wizard.launch_cli")
    def test_wizard_gradio_fallback(self, mock_cli, mock_gradio, capsys):
        """Falls back to CLI when Gradio is not installed."""
        mock_cli.return_value = "use_nli: false\n"
        main(["wizard"])
        captured = capsys.readouterr()
        assert "Gradio not installed" in captured.out
        mock_cli.assert_called_once()


# ── compliance --format html ─────────────────────────────────────────


class TestComplianceHTMLFormat:
    """Tests for 'director-ai compliance report --format html'."""

    def test_compliance_help_shows_html(self, capsys):
        """Help text mentions html format."""
        main(["compliance"])
        captured = capsys.readouterr()
        assert "html" in captured.out

    @patch("director_ai.compliance.reporter.ComplianceReporter")
    @patch("director_ai.compliance.audit_log.AuditLog")
    def test_compliance_html_output(
        self, mock_log_cls, mock_reporter_cls, capsys, tmp_path
    ):
        """--format html produces HTML with compliance data."""
        db_file = tmp_path / "test_audit.db"
        db_file.touch()

        mock_report = MagicMock()
        mock_report.total_interactions = 1000
        mock_report.overall_hallucination_rate = 0.042
        mock_report.overall_hallucination_rate_ci = (0.03, 0.05)
        mock_report.avg_score = 0.85
        mock_report.drift_detected = False
        mock_report.incident_count = 3
        mock_report.approved_count = 950
        mock_report.rejected_count = 50
        mock_report.avg_latency_ms = 12.5
        mock_report.model_breakdown = []

        mock_reporter = MagicMock()
        mock_reporter.generate_report.return_value = mock_report
        mock_reporter_cls.return_value = mock_reporter

        main(["compliance", "report", "--db", str(db_file), "--format", "html"])

        captured = capsys.readouterr()
        assert "<!DOCTYPE html>" in captured.out
        assert "EU AI Act" in captured.out
        assert "4.20%" in captured.out

    @patch("director_ai.compliance.reporter.ComplianceReporter")
    @patch("director_ai.compliance.audit_log.AuditLog")
    def test_compliance_html_drift_detected(
        self, mock_log_cls, mock_reporter_cls, capsys, tmp_path
    ):
        """HTML report shows drift alert when detected."""
        db_file = tmp_path / "test_audit.db"
        db_file.touch()

        mock_report = MagicMock()
        mock_report.total_interactions = 500
        mock_report.overall_hallucination_rate = 0.12
        mock_report.avg_score = 0.7
        mock_report.drift_detected = True
        mock_report.incident_count = 10
        mock_report.approved_count = 400
        mock_report.rejected_count = 100
        mock_report.avg_latency_ms = 25.0
        mock_report.model_breakdown = [
            {"model": "gpt-4o", "reviews": 300, "rate": 0.10, "avg_score": 0.72},
        ]

        mock_reporter = MagicMock()
        mock_reporter.generate_report.return_value = mock_report
        mock_reporter_cls.return_value = mock_reporter

        main(["compliance", "report", "--db", str(db_file), "--format", "html"])

        captured = capsys.readouterr()
        assert "Drift DETECTED" in captured.out
        assert "gpt-4o" in captured.out


# ── Integration: generate_yaml round-trip ────────────────────────────


class TestConfigWizardIntegration:
    """Integration tests for config wizard YAML generation."""

    def test_generate_yaml_defaults(self):
        """generate_yaml with no overrides produces valid YAML."""
        from director_ai.ui.config_wizard import generate_yaml

        yaml_str = generate_yaml()
        assert "Director-AI Configuration" in yaml_str

    def test_generate_yaml_with_overrides(self):
        """generate_yaml applies overrides correctly."""
        from director_ai.ui.config_wizard import generate_yaml

        yaml_str = generate_yaml({"coherence_threshold": 0.9, "use_nli": True})
        assert "coherence_threshold: 0.9" in yaml_str
        assert "use_nli: true" in yaml_str

    def test_generate_yaml_bool_formatting(self):
        """Boolean values are formatted as true/false."""
        from director_ai.ui.config_wizard import generate_yaml

        yaml_str = generate_yaml({"use_nli": False})
        assert "use_nli: false" in yaml_str


# ── KBHealthCheck unit tests ─────────────────────────────────────────


class TestKBHealthCheckUnit:
    """Unit tests for KBHealthCheck diagnostics."""

    def test_healthy_report(self):
        """Healthy store produces passing report."""
        from director_ai.core.retrieval.kb_health import KBHealthCheck

        store = MagicMock()
        store.backend.count.return_value = 5
        store.retrieve_context.return_value = ["result"]

        check = KBHealthCheck(store)
        report = check.run()

        assert report.healthy is True
        assert report.document_count == 5
        assert "HEALTHY" in report.summary

    def test_empty_store_unhealthy(self):
        """Empty store is flagged as unhealthy."""
        from director_ai.core.retrieval.kb_health import KBHealthCheck

        store = MagicMock()
        store.backend.count.return_value = 0
        store.retrieve_context.return_value = []

        check = KBHealthCheck(store, min_documents=1)
        report = check.run()

        assert report.healthy is False
        assert len(report.issues) > 0

    def test_high_latency_warning(self):
        """High latency generates a warning."""
        import time

        from director_ai.core.retrieval.kb_health import KBHealthCheck

        store = MagicMock()
        store.backend.count.return_value = 10

        def slow_retrieve(query):
            time.sleep(0.15)
            return ["result"]

        store.retrieve_context.side_effect = slow_retrieve

        check = KBHealthCheck(store, max_query_latency_ms=50.0)
        report = check.run()

        assert len(report.warnings) > 0
        assert any("latency" in w.lower() for w in report.warnings)

    def test_unqueryable_store(self):
        """Store without retrieve_context is flagged."""
        from director_ai.core.retrieval.kb_health import KBHealthCheck

        store = MagicMock(spec=[])  # no methods at all

        check = KBHealthCheck(store)
        report = check.run()

        assert report.healthy is False


# ── Report templates unit tests ──────────────────────────────────────


class TestReportTemplatesUnit:
    """Unit tests for report template rendering."""

    def test_compliance_html_structure(self):
        """HTML report has valid structure."""
        from director_ai.compliance.report_templates import render_compliance_html

        html = render_compliance_html(
            {
                "title": "Test Report",
                "period": "2026-04-01 to 2026-04-14",
                "hallucination_rate": 0.05,
                "total_reviews": 2000,
                "approved_count": 1900,
                "rejected_count": 100,
                "avg_score": 0.88,
                "avg_latency_ms": 15.0,
                "drift_detected": False,
                "models": [],
            }
        )
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "5.00%" in html
        assert "2,000" in html

    def test_compliance_markdown(self):
        """Markdown report has correct structure."""
        from director_ai.compliance.report_templates import render_compliance_markdown

        md = render_compliance_markdown(
            {
                "title": "MD Report",
                "total_reviews": 500,
                "hallucination_rate": 0.03,
                "avg_latency_ms": 10.0,
                "avg_score": 0.92,
                "drift_detected": False,
                "models": [
                    {
                        "model": "gpt-4o",
                        "reviews": 300,
                        "rate": 0.02,
                        "avg_score": 0.95,
                    },
                ],
            }
        )
        assert "# MD Report" in md
        assert "gpt-4o" in md
        assert "500" in md

    def test_cost_html(self):
        """Cost HTML report renders correctly."""
        from director_ai.compliance.report_templates import render_cost_html

        html = render_cost_html(
            {
                "currency": "CHF",
                "total_cost": 1.2345,
                "total_tokens": 50000,
                "models": {
                    "gpt-4o": {
                        "call_count": 100,
                        "total_tokens": 50000,
                        "estimated_cost": 1.2345,
                    }
                },
            }
        )
        assert "Cost Report" in html
        assert "CHF" in html
        assert "50,000" in html

    def test_swarm_html(self):
        """Swarm HTML report renders correctly."""
        from director_ai.compliance.report_templates import render_swarm_html

        html = render_swarm_html(
            {
                "swarm": {
                    "active_agents": 3,
                    "total_handoffs": 42,
                    "quarantined_agents": 1,
                    "cascade_events": 2,
                },
                "agents": {
                    "researcher": {
                        "handoffs": 20,
                        "hallucination_rate": 0.05,
                        "quarantined": False,
                    },
                    "coder": {
                        "handoffs": 22,
                        "hallucination_rate": 0.15,
                        "quarantined": True,
                    },
                },
            }
        )
        assert "Swarm Health" in html
        assert "researcher" in html
        assert "coder" in html

    def test_html_escaping(self):
        """XSS payloads are escaped in HTML output."""
        from director_ai.compliance.report_templates import render_compliance_html

        html = render_compliance_html(
            {
                "title": '<script>alert("xss")</script>',
                "period": "test",
                "hallucination_rate": 0,
                "total_reviews": 0,
                "avg_score": 0,
                "drift_detected": False,
                "models": [],
            }
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
