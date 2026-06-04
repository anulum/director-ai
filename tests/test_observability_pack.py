# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Primary observability pack tests

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "deploy/observability/prometheus-alerts.yml"
DASHBOARD = ROOT / "deploy/observability/grafana-dashboard.json"
DOC = ROOT / "docs-site/deployment/observability.md"

STALE_METRICS = (
    "director_reviews_total",
    "director_hallucinations_total",
    "director_review_duration_seconds",
    "director_coherence_score",
    "director_streaming_active_sessions",
    "director_streaming_halts_total",
    "director_knowledge_queries_total",
    "director_knowledge_query_errors",
    "director_errors_total",
    "director_drift_score",
)


class TestPrimaryPrometheusRules:
    def test_rules_file_is_valid_yaml(self) -> None:
        parsed = yaml.safe_load(RULES.read_text(encoding="utf-8"))

        assert parsed["groups"][0]["name"] == "director-ai"
        assert len(parsed["groups"][0]["rules"]) == 6

    def test_rules_use_shipped_metric_families(self) -> None:
        text = RULES.read_text(encoding="utf-8")

        assert "director_ai_reviews_total" in text
        assert "director_ai_reviews_rejected" in text
        assert "director_ai_review_duration_seconds_bucket" in text
        assert "director_ai_halts_total" in text
        assert "director_ai_http_requests_total" in text
        assert "director_ai_kb_stale_sources" in text
        for stale in STALE_METRICS:
            assert stale not in text

    def test_rules_expose_current_alert_names(self) -> None:
        rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))["groups"][0]["rules"]
        names = {rule["alert"] for rule in rules}

        assert names == {
            "HighRejectionRate",
            "ReviewLatencyHigh",
            "HaltSpike",
            "RetuneRecommended",
            "ErrorRateHigh",
            "StaleKnowledgeSources",
        }


class TestPrimaryGrafanaDashboard:
    def test_dashboard_is_valid_json(self) -> None:
        dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))

        assert dashboard["uid"] == "director-ai-overview"
        assert dashboard["title"] == "Director-AI Guardrail Overview"
        assert len(dashboard["panels"]) == 9

    def test_dashboard_uses_shipped_metric_families(self) -> None:
        dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
        payload = json.dumps(dashboard)

        assert "director_ai_reviews_total" in payload
        assert "director_ai_reviews_rejected" in payload
        assert "director_ai_review_duration_seconds_bucket" in payload
        assert "director_ai_coherence_score_bucket" in payload
        assert "director_ai_active_requests" in payload
        assert "director_ai_halts_total" in payload
        assert "director_ai_http_requests_total" in payload
        for stale in STALE_METRICS:
            assert stale not in payload


class TestPrimaryObservabilityDocs:
    def test_docs_reference_current_metric_families(self) -> None:
        text = DOC.read_text(encoding="utf-8")

        assert "director_ai_reviews_total" in text
        assert "director_ai_reviews_rejected" in text
        assert "director_ai_review_duration_seconds_bucket" in text
        assert "director_ai_http_requests_total" in text
        assert "director_ai_kb_stale_sources" in text
        for stale in STALE_METRICS:
            assert stale not in text

    def test_docs_reference_primary_observability_assets(self) -> None:
        text = DOC.read_text(encoding="utf-8")

        assert "grafana-dashboard.json" in text
        assert "prometheus-alerts.yml" in text
