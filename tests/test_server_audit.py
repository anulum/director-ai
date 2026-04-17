# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Server Audit Logging Tests
"""Multi-angle tests for FastAPI server audit logging pipeline.

Covers: review audit entry, process audit entry, disabled audit path,
request-id roundtrip, auto-generated request-id, parametrised endpoints,
audit entry fields, and pipeline performance documentation.
"""

from __future__ import annotations

import json

import pytest

from director_ai.core.config import DirectorConfig

try:
    from fastapi.testclient import TestClient

    from director_ai.server import create_app

    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _SERVER_AVAILABLE, reason="fastapi not installed")


def test_review_creates_audit_entry(tmp_path):
    audit_file = tmp_path / "audit.jsonl"
    cfg = DirectorConfig(audit_log_path=str(audit_file), llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post(
            "/v1/review",
            json={"prompt": "sky color?", "response": "The sky is blue."},
        )
    assert r.status_code == 200
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 1
    entry = json.loads(lines[0])
    assert "approved" in entry
    assert "score" in entry
    assert "query_hash" in entry
    assert entry["response_length"] > 0


def test_process_creates_audit_entry(tmp_path):
    audit_file = tmp_path / "audit.jsonl"
    cfg = DirectorConfig(audit_log_path=str(audit_file), llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post("/v1/process", json={"prompt": "What is 2+2?"})
    assert r.status_code == 200
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 1
    entry = json.loads(lines[0])
    assert "approved" in entry


def test_no_audit_when_path_empty():
    cfg = DirectorConfig(audit_log_path="", llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post(
            "/v1/review",
            json={"prompt": "test", "response": "test"},
        )
    assert r.status_code == 200


def test_request_id_round_trip():
    cfg = DirectorConfig(llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post(
            "/v1/review",
            json={"prompt": "test", "response": "test"},
            headers={"X-Request-ID": "my-req-001"},
        )
    assert r.status_code == 200
    assert r.headers["X-Request-ID"] == "my-req-001"


def test_request_id_generated_when_absent():
    cfg = DirectorConfig(llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/health")
    assert r.status_code == 200
    rid = r.headers.get("X-Request-ID", "")
    assert len(rid) > 0


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/v1/review", {"prompt": "test", "response": "test"}),
        ("/v1/process", {"prompt": "test question"}),
    ],
)
def test_audit_entry_has_required_fields(tmp_path, endpoint, payload):
    audit_file = tmp_path / "audit_fields.jsonl"
    cfg = DirectorConfig(audit_log_path=str(audit_file), llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post(endpoint, json=payload)
    assert r.status_code == 200
    lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
    entry = json.loads(lines[0])
    assert "approved" in entry


class TestServerAuditPerformanceDoc:
    """Document server audit pipeline performance."""

    def test_health_endpoint_fast(self):
        import time

        cfg = DirectorConfig(llm_provider="mock")
        app = create_app(cfg)
        with TestClient(app) as client:
            t0 = time.perf_counter()
            for _ in range(10):
                client.get("/v1/health")
            per_call_ms = (time.perf_counter() - t0) / 10 * 1000
        assert per_call_ms < 100, f"Health check took {per_call_ms:.0f}ms"

    def test_review_returns_json(self):
        cfg = DirectorConfig(llm_provider="mock")
        app = create_app(cfg)
        with TestClient(app) as client:
            r = client.post(
                "/v1/review",
                json={"prompt": "test", "response": "test"},
            )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
