# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - safety operations mixin tests

from __future__ import annotations

import json
from pathlib import Path

import yaml

from director_ai.core.metrics import MetricsCollector

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "deploy/observability/safety-ops-prometheus-rules.yml"
DASHBOARD = ROOT / "deploy/observability/safety-ops-grafana-dashboard.json"
DOC = ROOT / "docs-site/deployment/safety-dashboard.md"


class TestSafetyOpsPrometheusRules:
    def test_rules_file_is_valid_yaml(self):
        parsed = yaml.safe_load(RULES.read_text(encoding="utf-8"))

        assert parsed["groups"][0]["name"] == "director-ai-safety-ops"
        assert len(parsed["groups"][0]["rules"]) == 4

    def test_rules_cover_halt_feedback_stale_and_retune_alerts(self):
        rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))["groups"][0]["rules"]
        names = {rule["alert"] for rule in rules}

        assert names == {
            "DirectorHighHaltRate",
            "DirectorFalsePositiveRateHigh",
            "DirectorStaleKnowledgeSources",
            "DirectorFeedbackRetuneDue",
        }

    def test_rules_use_current_metric_prefix_and_safe_denominators(self):
        text = RULES.read_text(encoding="utf-8")

        assert "director_ai_halts_total" in text
        assert "director_ai_feedback_total" in text
        assert "director_ai_kb_stale_sources" in text
        assert "director_ai_retune_recommended" in text
        assert "director_hallucinations_total" not in text
        assert "director_reviews_total" not in text
        assert "clamp_min" in text


class TestSafetyOpsGrafanaDashboard:
    def test_dashboard_is_valid_json(self):
        dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))

        assert dashboard["uid"] == "director-ai-safety-ops"
        assert dashboard["title"] == "Director-AI Safety Operations"
        assert len(dashboard["panels"]) == 7

    def test_dashboard_panels_cover_operational_questions(self):
        dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
        titles = {panel["title"] for panel in dashboard["panels"]}

        assert "Halt rate by instance" in titles
        assert "False-positive feedback rate" in titles
        assert "Stale knowledge sources" in titles
        assert "Retune recommended" in titles

    def test_dashboard_uses_current_metric_prefix(self):
        dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
        payload = json.dumps(dashboard)

        assert "director_ai_halts_total" in payload
        assert "director_ai_feedback_total" in payload
        assert "director_ai_kb_stale_sources" in payload
        assert "director_hallucinations_total" not in payload


class TestSafetyOpsMetricFamilies:
    def test_metric_families_are_registered(self):
        collector = MetricsCollector()
        metrics = collector.get_metrics()

        assert "feedback_total" in metrics["counters"]
        assert "retune_recommendations_total" in metrics["counters"]
        assert "kb_stale_sources" in metrics["gauges"]
        assert "retune_recommended" in metrics["gauges"]

    def test_prometheus_export_contains_safety_ops_families(self):
        collector = MetricsCollector()
        collector.inc_labeled("feedback_total", {"outcome": "false_positive"})
        collector.inc("retune_recommendations_total")
        collector.gauge_set("kb_stale_sources", 2)
        collector.gauge_set("retune_recommended", 1)

        text = collector.prometheus_format()

        assert 'director_ai_feedback_total{outcome="false_positive"} 1.0' in text
        assert "director_ai_retune_recommendations_total 1.0" in text
        assert "director_ai_kb_stale_sources 2" in text
        assert "director_ai_retune_recommended 1" in text

    def test_docs_reference_mixin_files(self):
        doc = DOC.read_text(encoding="utf-8")

        assert "safety-ops-prometheus-rules.yml" in doc
        assert "safety-ops-grafana-dashboard.json" in doc
