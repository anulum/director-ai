# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.compliance.report_templates``.

Covers HTML and Markdown rendering for compliance, cost, and swarm
reports with structure validation, content checks, and edge cases.
"""

from __future__ import annotations

from director_ai.compliance.report_templates import (
    render_compliance_html,
    render_compliance_markdown,
    render_cost_html,
    render_swarm_html,
)

_COMPLIANCE_DATA = {
    "title": "EU AI Act Article 15 Report",
    "period": "2026-04-01 to 2026-04-13",
    "hallucination_rate": 0.042,
    "total_reviews": 15000,
    "approved_count": 14370,
    "rejected_count": 630,
    "avg_score": 0.847,
    "avg_latency_ms": 14.6,
    "drift_detected": False,
    "models": [
        {"model": "FactCG", "reviews": 10000, "rate": 0.035, "avg_score": 0.86},
        {"model": "MiniCheck", "reviews": 5000, "rate": 0.056, "avg_score": 0.82},
    ],
}

_COST_DATA = {
    "currency": "CHF",
    "total_cost": 1.2345,
    "total_tokens": 500000,
    "models": {
        "gpt-4o": {
            "model": "gpt-4o",
            "call_count": 100,
            "total_tokens": 300000,
            "estimated_cost": 0.875,
        },
        "claude-3-5-sonnet::agent-1": {
            "model": "claude-3-5-sonnet",
            "call_count": 50,
            "total_tokens": 200000,
            "estimated_cost": 0.3595,
        },
    },
}

_SWARM_DATA = {
    "swarm": {
        "active_agents": 5,
        "total_handoffs": 1200,
        "quarantined_agents": 1,
        "cascade_events": 2,
    },
    "agents": {
        "researcher-0": {
            "handoffs": 500,
            "hallucination_rate": 0.02,
            "quarantined": False,
        },
        "summariser-0": {
            "handoffs": 400,
            "hallucination_rate": 0.15,
            "quarantined": True,
        },
    },
}


# ── Compliance HTML ────────────────────────────────────────────────────


class TestComplianceHTML:
    def test_is_valid_html(self):
        html = render_compliance_html(_COMPLIANCE_DATA)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_title_present(self):
        html = render_compliance_html(_COMPLIANCE_DATA)
        assert "EU AI Act Article 15 Report" in html

    def test_metrics_present(self):
        html = render_compliance_html(_COMPLIANCE_DATA)
        assert "15,000" in html
        assert "4.20%" in html
        assert "14.6 ms" in html

    def test_no_drift(self):
        html = render_compliance_html(_COMPLIANCE_DATA)
        assert "No drift" in html

    def test_drift_detected(self):
        data = {**_COMPLIANCE_DATA, "drift_detected": True}
        html = render_compliance_html(data)
        assert "Drift DETECTED" in html
        assert 'class="alert"' in html

    def test_models_table(self):
        html = render_compliance_html(_COMPLIANCE_DATA)
        assert "FactCG" in html
        assert "MiniCheck" in html

    def test_footer(self):
        html = render_compliance_html(_COMPLIANCE_DATA)
        assert "ANULUM" in html

    def test_empty_data(self):
        html = render_compliance_html({})
        assert "<!DOCTYPE html>" in html


# ── Compliance Markdown ────────────────────────────────────────────────


class TestComplianceMarkdown:
    def test_starts_with_heading(self):
        md = render_compliance_markdown(_COMPLIANCE_DATA)
        assert md.startswith("# EU AI Act")

    def test_metrics_table(self):
        md = render_compliance_markdown(_COMPLIANCE_DATA)
        assert "| Total Reviews | 15,000 |" in md

    def test_models_listed(self):
        md = render_compliance_markdown(_COMPLIANCE_DATA)
        assert "FactCG" in md

    def test_empty_data(self):
        md = render_compliance_markdown({})
        assert "# Compliance Report" in md


# ── Cost HTML ──────────────────────────────────────────────────────────


class TestCostHTML:
    def test_is_valid_html(self):
        html = render_cost_html(_COST_DATA)
        assert "<!DOCTYPE html>" in html

    def test_total_cost(self):
        html = render_cost_html(_COST_DATA)
        assert "1.2345" in html

    def test_currency(self):
        html = render_cost_html(_COST_DATA)
        assert "CHF" in html

    def test_models_listed(self):
        html = render_cost_html(_COST_DATA)
        assert "gpt-4o" in html
        assert "agent-1" in html

    def test_empty_data(self):
        html = render_cost_html({})
        assert "<!DOCTYPE html>" in html


# ── Swarm HTML ─────────────────────────────────────────────────────────


class TestSwarmHTML:
    def test_is_valid_html(self):
        html = render_swarm_html(_SWARM_DATA)
        assert "<!DOCTYPE html>" in html

    def test_agent_count(self):
        html = render_swarm_html(_SWARM_DATA)
        assert "5" in html  # active agents

    def test_quarantined_highlighted(self):
        html = render_swarm_html(_SWARM_DATA)
        assert "summariser-0" in html
        assert "alert" in html  # quarantined row

    def test_handoffs(self):
        html = render_swarm_html(_SWARM_DATA)
        assert "1,200" in html

    def test_empty_data(self):
        html = render_swarm_html({})
        assert "<!DOCTYPE html>" in html
