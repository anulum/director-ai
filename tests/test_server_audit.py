# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Server Audit Logging Tests

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
