# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Server Tests
"""Multi-angle tests for FastAPI server pipeline."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

try:
    from fastapi.testclient import TestClient

    from director_ai.core.config import DirectorConfig
    from director_ai.server import create_app

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI not installed")


@pytest.fixture
def client():
    """TestClient for the Director AI server (with lifespan)."""
    config = DirectorConfig(use_nli=False, metrics_enabled=True)
    app = create_app(config)
    with TestClient(app) as c:
        yield c


class TestHealth:
    """Health endpoint tests."""

    def test_health_ok(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_has_profile(self, client):
        resp = client.get("/v1/health")
        data = resp.json()
        assert "profile" in data


class TestReview:
    """Review endpoint tests."""

    def test_review_valid(self, client):
        resp = client.post(
            "/v1/review",
            json={"prompt": "What color is the sky?", "response": "Blue"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "approved" in data
        assert "coherence" in data
        assert "h_logical" in data
        assert "h_factual" in data

    def test_review_missing_fields(self, client):
        resp = client.post("/v1/review", json={"prompt": "Hello"})
        assert resp.status_code == 422  # Validation error


class TestProcess:
    """Process endpoint tests."""

    def test_process_valid(self, client):
        resp = client.post(
            "/v1/process",
            json={"prompt": "What is the meaning of life?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "output" in data
        assert "halted" in data
        assert "candidates_evaluated" in data

    def test_process_missing_prompt(self, client):
        resp = client.post("/v1/process", json={})
        assert resp.status_code == 422


class TestBatch:
    """Batch endpoint tests."""

    def test_batch_valid(self, client):
        resp = client.post(
            "/v1/batch",
            json={
                "task": "review",
                "prompts": ["What is water?", "What is air?"],
                "responses": ["Water is H2O.", "Air is a mixture of gases."],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["succeeded"] + data["failed"] == 2
        assert len(data["results"]) + len(data["errors"]) == 2

    def test_batch_empty_prompts(self, client):
        resp = client.post("/v1/batch", json={"prompts": []})
        assert resp.status_code == 422  # min_length=1


class TestMetrics:
    """Metrics endpoint tests."""

    def test_metrics_json(self, client):
        # Trigger a review first
        client.post(
            "/v1/review",
            json={"prompt": "Q", "response": "A"},
        )
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "counters" in data
        assert "histograms" in data
        assert "gauges" in data

    def test_metrics_prometheus(self, client):
        resp = client.get("/v1/metrics/prometheus")
        assert resp.status_code == 200
        assert "director_ai_" in resp.text


class TestConfig:
    """Config endpoint tests."""

    def test_config_endpoint(self, client):
        resp = client.get("/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "coherence_threshold" in data["config"]

    def test_scorer_models_endpoint(self, client):
        resp = client.get("/v1/scorer/models")
        assert resp.status_code == 200
        data = resp.json()

        aliases = {model["alias"] for model in data["models"]}
        assert "balanced-default" in aliases
        assert "distilroberta-fast" not in aliases
        assert data["current"]["nli_model"]

    def test_scorer_models_endpoint_can_include_domain_only(self, client):
        resp = client.get("/v1/scorer/models?include_domain_only=true")
        assert resp.status_code == 200
        data = resp.json()

        aliases = {model["alias"] for model in data["models"]}
        assert "distilroberta-fast" in aliases


class TestStats:
    """Stats endpoint tests."""

    def test_stats_returns_summary(self, client):
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "approved" in data
        assert "rejected" in data
        assert isinstance(data["total"], int)

    def test_stats_after_review(self, client):
        client.post(
            "/v1/review",
            json={"prompt": "Q", "response": "A"},
        )
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_stats_hourly(self, client):
        resp = client.get("/v1/stats/hourly")
        assert resp.status_code == 200

    def test_stats_hourly_custom_days(self, client):
        resp = client.get("/v1/stats/hourly?days=1")
        assert resp.status_code == 200


class TestDashboard:
    """Dashboard endpoint tests."""

    def test_dashboard_html(self, client):
        resp = client.get("/v1/dashboard")
        assert resp.status_code == 200
        assert "Director-AI Dashboard" in resp.text
        assert "Total Reviews" in resp.text

    def test_dashboard_after_review(self, client):
        client.post(
            "/v1/review",
            json={"prompt": "Q", "response": "A"},
        )
        resp = client.get("/v1/dashboard")
        assert resp.status_code == 200
        assert "Approval Rate" in resp.text


class TestWebSocket:
    """WebSocket /v1/stream endpoint tests."""

    def test_ws_valid_prompt(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "What is 2+2?"})
            data = ws.receive_json()
            assert data["type"] == "result"
            assert "output" in data
            assert "halted" in data

    def test_ws_empty_prompt(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": ""})
            data = ws.receive_json()
            assert "error" in data

    def test_ws_missing_prompt(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"foo": "bar"})
            data = ws.receive_json()
            assert "error" in data

    def test_ws_multiple_messages(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "Q1"})
            d1 = ws.receive_json()
            ws.send_json({"prompt": "Q2"})
            d2 = ws.receive_json()
            assert d1["type"] == "result"
            assert d2["type"] == "result"

    def test_ws_prompt_exceeds_max_length(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "x" * 100_001})
            data = ws.receive_json()
            assert "error" in data
            assert "100000" in data["error"]

    def test_ws_non_string_prompt(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": 12345})
            data = ws.receive_json()
            assert "error" in data

    def test_ws_non_dict_payload(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json([1, 2, 3])
            data = ws.receive_json()
            assert "error" in data


class TestWebSocketAgentError:
    """WebSocket error handling when agent.process() raises."""

    def test_ws_agent_error_returns_error_json(self, client):
        from unittest.mock import patch

        with (
            patch(
                "director_ai.core.agent.CoherenceAgent.process",
                side_effect=RuntimeError("GPU OOM"),
            ),
            client.websocket_connect("/v1/stream") as ws,
        ):
            ws.send_json({"prompt": "trigger failure"})
            data = ws.receive_json()
            assert "error" in data
            assert "processing failed" in data["error"]

    def test_ws_agent_error_does_not_kill_connection(self, client):
        from unittest.mock import patch

        call_count = 0

        def _fail_once(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("transient error")
            from director_ai.core.agent import CoherenceAgent

            return CoherenceAgent.process(client.app, prompt)

        with client.websocket_connect("/v1/stream") as ws:
            with patch(
                "director_ai.core.agent.CoherenceAgent.process",
                side_effect=ValueError("transient"),
            ):
                ws.send_json({"prompt": "fail"})
                d1 = ws.receive_json()
                assert "error" in d1

            # Connection survives — next message works
            ws.send_json({"prompt": "ok"})
            d2 = ws.receive_json()
            assert d2["type"] == "result"


class TestServerClaimSupportEvidenceSerialization:
    """Server evidence serialization preserves claim-support diagnostics."""

    def test_evidence_without_claim_support_omits_claim_fields(self):
        from director_ai.core.types import ScoringEvidence
        from director_ai.server import _evidence_to_dict

        evidence = ScoringEvidence(
            chunks=[],
            nli_premise="premise",
            nli_hypothesis="hypothesis",
            nli_score=0.5,
        )

        encoded = _evidence_to_dict(evidence)

        assert "claim_coverage" not in encoded
        assert "per_claim_divergences" not in encoded
        assert "claims" not in encoded

    def test_evidence_with_claim_support_preserves_claim_fields(self):
        from director_ai.core.types import ScoringEvidence
        from director_ai.server import _evidence_to_dict

        evidence = ScoringEvidence(
            chunks=[],
            nli_premise="premise",
            nli_hypothesis="hypothesis",
            nli_score=0.5,
            claim_coverage=0.75,
            per_claim_divergences=[0.1, 0.8],
            claims=["Supported claim.", "Unsupported claim."],
        )

        encoded = _evidence_to_dict(evidence)

        assert encoded["claim_coverage"] == 0.75
        assert encoded["per_claim_divergences"] == [0.1, 0.8]
        assert encoded["claims"] == ["Supported claim.", "Unsupported claim."]


class TestServerOperationalReadiness:
    """Server health and readiness expose deployability boundaries."""

    def test_health_reports_model_revision_registry_failures(self):
        config = DirectorConfig(
            use_nli=False,
            nli_model="unverified-org/unverified-model",
            nli_model_revision="",
        )
        app = create_app(config)

        with TestClient(app) as client:
            response = client.get("/v1/health")

        assert response.status_code == 200
        revision_health = response.json()["model_revisions"]
        assert revision_health["ok"] is False
        assert revision_health["checks"]["nli"]["status"] == "error"

    def test_production_mode_requires_knowledge_router(self, monkeypatch):
        from unittest.mock import patch

        # A production server fail-fasts without a per-installation audit salt;
        # set one so this test reaches the knowledge-router requirement it covers.
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "test-installation-salt")
        config = DirectorConfig(
            use_nli=False,
            production_mode=True,
            api_keys=["writer"],
            llm_api_url="https://llm.internal.example/v1",
            knowledge_write_hmac_keys='{"kid-1":"signing-secret-at-least-32-chars-xx"}',
        )

        with (
            patch.dict("sys.modules", {"director_ai.knowledge_api": None}),
            pytest.raises(RuntimeError, match="knowledge API router"),
        ):
            create_app(config)

    def test_ready_endpoint_is_available_without_nli(self):
        app = create_app(DirectorConfig(use_nli=False))

        with TestClient(app) as client:
            response = client.get("/v1/ready")

        assert response.status_code == 200
        assert response.json()["ready"] is True


class TestServerCoverageGaps:
    """Dedicated server endpoint branch coverage."""

    @staticmethod
    def _fast_config() -> DirectorConfig:
        return DirectorConfig.from_profile("fast")

    def test_http_endpoint_label_handles_unmatched_and_partial_routes(self):
        from starlette.routing import Match

        from director_ai.server import _http_endpoint_label

        class NoMatcher:
            pass

        class PartialMatcher:
            path = "/partial"

            def matches(self, _scope):
                return Match.PARTIAL, {}

        request = SimpleNamespace(
            scope={"type": "http", "path": "/unknown"},
            app=SimpleNamespace(routes=[NoMatcher(), PartialMatcher()]),
        )

        assert _http_endpoint_label(request) == "/partial"

        request.app.routes = [NoMatcher()]
        assert _http_endpoint_label(request) == "__unmatched__"

    def test_create_app_uses_non_default_profile_from_environment(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_PROFILE", "fast")

        app = create_app()

        assert app.state.config.profile == "fast"

    def test_create_app_logs_loaded_managed_secrets(self, monkeypatch, caplog):
        import director_ai.core.secrets as secrets_mod

        monkeypatch.setattr(
            secrets_mod,
            "hydrate_managed_secrets",
            lambda: {"DIRECTOR_API_KEY": "loaded"},
        )

        with caplog.at_level("INFO", logger="DirectorAI.Server"):
            create_app(self._fast_config())

        assert "Loaded 1 managed secret(s) from backend" in caplog.text

    def test_lifespan_logs_commercial_and_trial_license_branches(self, caplog):
        from director_ai.core.license import LicenseInfo

        commercial = LicenseInfo(
            key="commercial-key",
            tier="enterprise",
            licensee="ACME",
            valid=True,
        )
        trial = LicenseInfo(
            key="trial-key",
            tier="trial",
            expires="2026-12-31",
            valid=True,
        )

        for license_info, expected in (
            (commercial, "Licensed to ACME"),
            (trial, "Trial license"),
        ):
            caplog.clear()
            app = create_app(self._fast_config())
            with (
                patch(
                    "director_ai.core.license.load_license", return_value=license_info
                ),
                caplog.at_level("INFO", logger="DirectorAI.Server"),
                TestClient(app),
            ):
                pass
            assert expected in caplog.text

    def test_lifespan_runs_otel_setup_when_enabled(self, monkeypatch):
        import director_ai.core.otel as otel_mod

        calls: list[str] = []
        monkeypatch.setattr(otel_mod, "setup_otel", lambda: calls.append("setup"))
        cfg = self._fast_config()
        cfg.otel_enabled = True
        app = create_app(cfg)

        with TestClient(app):
            pass

        assert calls == ["setup"]

    def test_lifespan_starts_and_stops_review_queue(self, monkeypatch):
        import director_ai.core.runtime.review_queue as review_queue_mod

        events: list[tuple[str, object]] = []

        class FakeReviewQueue:
            def __init__(self, scorer, *, max_batch, flush_timeout_ms):
                events.append(("init", scorer))
                events.append(("max_batch", max_batch))
                events.append(("flush_timeout_ms", flush_timeout_ms))

            async def start(self):
                events.append(("start", None))

            async def stop(self):
                events.append(("stop", None))

        monkeypatch.setattr(review_queue_mod, "ReviewQueue", FakeReviewQueue)
        cfg = self._fast_config()
        cfg.review_queue_enabled = True
        cfg.review_queue_max_batch = 7
        cfg.review_queue_flush_timeout_ms = 1.5
        app = create_app(cfg)

        with TestClient(app) as client:
            assert client.app.state._state["review_queue"].__class__ is FakeReviewQueue

        assert ("max_batch", 7) in events
        assert ("flush_timeout_ms", 1.5) in events
        assert ("start", None) in events
        assert ("stop", None) in events

    def test_lifespan_wires_postgres_audit_sink(self, monkeypatch):
        import director_ai.enterprise.audit_pg as audit_pg_mod

        sinks: list[str] = []

        class FakePostgresAuditSink:
            def __init__(self, db_url: str):
                sinks.append(db_url)

        monkeypatch.setattr(audit_pg_mod, "PostgresAuditSink", FakePostgresAuditSink)
        cfg = self._fast_config()
        cfg.audit_postgres_url = "postgresql://audit.example/db"
        app = create_app(cfg)

        with TestClient(app) as client:
            audit_logger = client.app.state._state["audit"]

        assert sinks == ["postgresql://audit.example/db"]
        assert audit_logger is not None

    def test_health_reports_commercial_and_trial_license_branches(self):
        app = create_app(self._fast_config())

        with TestClient(app) as client:
            client.app.state._license = SimpleNamespace(
                is_commercial=True,
                is_trial=False,
                tier="enterprise",
                licensee="ACME",
            )
            commercial = client.get("/v1/health").json()
            client.app.state._license = SimpleNamespace(
                is_commercial=False,
                is_trial=True,
                expires="2026-12-31",
            )
            trial = client.get("/v1/health").json()

        assert commercial["status"] == "ok"
        assert trial["status"] == "ok"

    def test_source_endpoint_commercial_and_disabled_paths(self):
        commercial_app = create_app(self._fast_config())
        with TestClient(commercial_app) as client:
            client.app.state._license = SimpleNamespace(
                is_commercial=True,
                is_trial=False,
                tier="enterprise",
                licensee="ACME",
            )
            response = client.get("/v1/source")
            assert response.status_code == 200
            body = response.json()
            assert body["license"] == "commercial"
            # The auth-exempt endpoint must not leak the commercial tier/licensee.
            assert body.get("tier", "") == ""
            assert body.get("licensee", "") == ""

        disabled_cfg = self._fast_config()
        disabled_cfg.source_endpoint_enabled = False
        disabled_app = create_app(disabled_cfg)
        with TestClient(disabled_app) as client:
            response = client.get("/v1/source")
            assert response.status_code == 404

        disabled_commercial_cfg = self._fast_config()
        disabled_commercial_cfg.source_endpoint_enabled = False
        disabled_commercial_app = create_app(disabled_commercial_cfg)
        with TestClient(disabled_commercial_app) as client:
            client.app.state._license = SimpleNamespace(
                is_commercial=True,
                is_trial=False,
                tier="enterprise",
                licensee="ACME",
            )
            response = client.get("/v1/source")
            assert response.status_code == 404
            assert "commercial license" in response.json()["detail"]

    def test_ready_reports_missing_scorer_and_unloaded_nli(self):
        cfg = self._fast_config()
        app = create_app(cfg)

        with TestClient(app) as client:
            client.app.state._state["scorer"] = None
            missing = client.get("/v1/ready")
            cfg.use_nli = True
            client.app.state._state["scorer"] = SimpleNamespace(
                _nli=SimpleNamespace(model_available=False)
            )
            unloaded = client.get("/v1/ready")

        assert missing.status_code == 503
        assert missing.json()["reason"] == "scorer not initialised"
        assert unloaded.status_code == 503
        assert unloaded.json()["reason"] == "NLI model not loaded"

    def test_feedback_store_success_disagreement_and_calibration(self):
        class FakeFeedbackStore:
            def __init__(self):
                self.reports = []

            def report(self, **kwargs):
                self.reports.append(kwargs)

            def count(self, domain=None):
                assert domain == "medical"
                return len(self.reports)

            def close(self):
                self.closed = True

        app = create_app(self._fast_config())
        with TestClient(app) as client:
            missing = client.post(
                "/v1/feedback",
                json={
                    "prompt": "p",
                    "response": "r",
                    "guardrail_approved": True,
                    "human_approved": False,
                    "domain": "medical",
                    "review_id": "r1",
                },
            )
            client.app.state._state["feedback_store"] = FakeFeedbackStore()
            accepted = client.post(
                "/v1/feedback",
                headers={"X-Tenant-ID": "tenant-a"},
                json={
                    "prompt": "p",
                    "response": "r",
                    "guardrail_approved": True,
                    "human_approved": False,
                    "guardrail_score": 0.8,
                    "domain": "medical",
                    "review_id": "r1",
                },
            )
            bad_calibration = client.get("/v1/feedback/calibration?min_corrections=0")

        assert missing.status_code == 503
        assert accepted.status_code == 200
        body = accepted.json()
        assert body["accepted"] is True
        assert body["disagreement"] is True
        assert body["correction_count"] == 1
        assert body["tenant_id"] == "tenant-a"
        assert bad_calibration.status_code == 400

    def test_verify_endpoint_no_scorer_sanitizer_and_no_context_paths(self):
        app = create_app(self._fast_config())

        class BlockingSanitizer:
            def check(self, text):
                del text
                return SimpleNamespace(blocked=True, reason="blocked")

        with TestClient(app) as client:
            client.app.state._state["scorer"] = None
            no_scorer = client.post(
                "/v1/verify",
                json={"prompt": "p", "response": "r"},
            )
            client.app.state._state["scorer"] = SimpleNamespace(
                ground_truth_store=SimpleNamespace(
                    retrieve_context=lambda *args, **kwargs: ""
                ),
                _nli=None,
            )
            no_context = client.post(
                "/v1/verify",
                json={"prompt": "p", "response": "r"},
            )
            client.app.state._state["sanitizer"] = BlockingSanitizer()
            blocked = client.post(
                "/v1/verify",
                json={"prompt": "p", "response": "r"},
            )

        assert no_scorer.status_code == 503
        assert no_context.status_code == 200
        assert (
            no_context.json()["reason"] == "No relevant context found in knowledge base"
        )
        assert blocked.status_code == 400
        assert "blocked" in blocked.json()["detail"]

    def test_process_redaction_and_internal_error_paths(self):
        from director_ai.core.types import CoherenceScore, ReviewResult

        class FakeRedactor:
            enabled = True

            def __call__(self, text):
                return text.replace("secret", "[redacted]")

            def redact(self, text):
                return text.replace("secret", "[redacted]")

        class FakeAgent:
            async def aprocess(self, prompt, tenant_id=""):
                assert prompt == "Tell me [redacted]"
                assert tenant_id == "tenant-a"
                return ReviewResult(
                    output="answer with secret",
                    coherence=CoherenceScore(
                        score=0.91,
                        approved=True,
                        h_logical=0.1,
                        h_factual=0.2,
                    ),
                    halted=False,
                    candidates_evaluated=1,
                    fallback_used=True,
                )

        class FailingAgent:
            async def aprocess(self, prompt, tenant_id=""):
                del prompt, tenant_id
                raise RuntimeError("processor failed")

        app = create_app(self._fast_config())
        with TestClient(app) as client:
            client.app.state._state["redactor"] = FakeRedactor()
            client.app.state._state["agent"] = FakeAgent()
            ok = client.post(
                "/v1/process",
                headers={"X-Tenant-ID": "tenant-a"},
                json={"prompt": "Tell me secret"},
            )
            client.app.state._state["agent"] = FailingAgent()
            failed = client.post("/v1/process", json={"prompt": "boom"})

        assert ok.status_code == 200
        body = ok.json()
        assert body["output"] == "answer with [redacted]"
        assert body["coherence"] == 0.91
        assert body["fallback_used"] is True
        assert failed.status_code == 500

    def test_batch_review_process_redaction_and_error_paths(self):
        from director_ai.core.types import CoherenceScore, ReviewResult

        class FakeRedactor:
            enabled = True

            def redact(self, text):
                return text.replace("secret", "[redacted]")

        class FakeBatchResult:
            def __init__(self, results, errors=()):
                self.results = results
                self.total = len(results) + len(errors)
                self.succeeded = len(results)
                self.failed = len(errors)
                self.errors = list(errors)

        class FakeBatcher:
            async def review_batch_async(self, pairs, tenant_id=""):
                assert pairs == [("p [redacted]", "r [redacted]")]
                assert tenant_id == "tenant-a"
                score = CoherenceScore(
                    score=0.7,
                    approved=True,
                    h_logical=0.1,
                    h_factual=0.2,
                )
                return FakeBatchResult([(True, score)])

            async def process_batch_async(self, prompts, tenant_id=""):
                assert prompts == ["p [redacted]"]
                assert tenant_id == "tenant-a"
                score = CoherenceScore(
                    score=0.8,
                    approved=True,
                    h_logical=0.1,
                    h_factual=0.1,
                )
                return FakeBatchResult(
                    [
                        ReviewResult(
                            output="secret output",
                            coherence=score,
                            halted=False,
                            candidates_evaluated=1,
                        )
                    ],
                    errors=[(2, "late failure")],
                )

        class FailingBatcher:
            async def review_batch_async(self, pairs, tenant_id=""):
                del pairs, tenant_id
                raise ValueError("bad batch")

            async def process_batch_async(self, prompts, tenant_id=""):
                del prompts, tenant_id
                raise RuntimeError("batch failed")

        app = create_app(self._fast_config())
        with TestClient(app) as client:
            mismatch = client.post(
                "/v1/batch",
                json={"task": "review", "prompts": ["p"], "responses": []},
            )
            client.app.state._state["redactor"] = FakeRedactor()
            client.app.state._state["batch"] = FakeBatcher()
            review_ok = client.post(
                "/v1/batch",
                headers={"X-Tenant-ID": "tenant-a"},
                json={
                    "task": "review",
                    "prompts": ["p secret"],
                    "responses": ["r secret"],
                },
            )
            process_ok = client.post(
                "/v1/batch",
                headers={"X-Tenant-ID": "tenant-a"},
                json={"task": "process", "prompts": ["p secret"]},
            )
            client.app.state._state["batch"] = FailingBatcher()
            value_error = client.post(
                "/v1/batch",
                json={"task": "review", "prompts": ["p"], "responses": ["r"]},
            )
            internal = client.post(
                "/v1/batch",
                json={"task": "process", "prompts": ["p"]},
            )

        assert mismatch.status_code == 422
        assert review_ok.status_code == 200
        assert review_ok.json()["results"][0]["score"] == 0.7
        assert process_ok.status_code == 200
        assert process_ok.json()["results"][0]["output"] == "[redacted] output"
        assert process_ok.json()["errors"] == [{"index": 2, "error": "late failure"}]
        assert value_error.status_code == 422
        assert internal.status_code == 500

    def test_stats_store_summary_hourly_and_prometheus_summary(self):
        class FakeStats:
            def summary(self):
                return {
                    "total": 3,
                    "approved": 2,
                    "rejected": 1,
                    "halted": 0,
                    "avg_score": 0.75,
                    "avg_latency_ms": 12.5,
                }

            def hourly_breakdown(self, days=7):
                assert days == 2
                return [{"hour": "2026-06-01T00:00:00Z", "total": 1}]

        app = create_app(self._fast_config())
        with TestClient(app) as client:
            client.app.state._state["stats"] = FakeStats()
            stats = client.get("/v1/stats")
            hourly = client.get("/v1/stats/hourly?days=2")

        assert stats.status_code == 200
        assert stats.json()["total"] == 3
        assert hourly.status_code == 200
        assert hourly.json()["data"][0]["total"] == 1
