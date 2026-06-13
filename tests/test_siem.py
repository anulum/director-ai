# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SIEM sink tests (mocked transport, offline)

from __future__ import annotations

import json

import pytest

from director_ai.core.safety import siem
from director_ai.core.safety.audit import AuditEntry
from director_ai.core.safety.siem import (
    DatadogLogsSink,
    ElasticsearchSink,
    SplunkHECSink,
)


class _Resp:
    def __init__(self, status=200):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _capture(monkeypatch, *, status=200):
    calls: list[dict] = []

    def fake_post(url, **kw):
        calls.append({"url": url, **kw})
        return _Resp(status)

    monkeypatch.setattr(siem.requests, "post", fake_post)
    return calls


def _entry(tenant="acme"):
    return AuditEntry(
        timestamp="2026-06-13T00:00:00Z",
        query_hash="abcd1234",
        response_length=42,
        approved=True,
        score=0.91,
        tenant_id=tenant,
    )


def test_splunk_write_endpoint_headers_payload(monkeypatch):
    calls = _capture(monkeypatch)
    SplunkHECSink("https://splunk.example.com:8088/", "tok123", index="main").write(
        _entry()
    )
    c = calls[-1]
    assert c["url"].endswith("/services/collector/event")
    assert c["headers"]["Authorization"] == "Splunk tok123"
    assert c["json"]["sourcetype"] == "director-ai:audit"
    assert c["json"]["index"] == "main"
    assert c["json"]["event"]["query_hash"] == "abcd1234"


def test_splunk_batch_concatenates(monkeypatch):
    calls = _capture(monkeypatch)
    SplunkHECSink("https://s:8088", "t").write_batch([_entry(), _entry("beta")])
    body = calls[-1]["data"]
    # two concatenated event objects, both parseable
    assert body.count('"event"') == 2


def test_datadog_write(monkeypatch):
    calls = _capture(monkeypatch)
    DatadogLogsSink("dd-key", service="guard").write(_entry("t1"))
    c = calls[-1]
    assert c["url"] == "https://http-intake.logs.datadoghq.com/api/v2/logs"
    assert c["headers"]["DD-API-KEY"] == "dd-key"
    rec = c["json"][0]
    assert rec["service"] == "guard"
    assert rec["ddtags"] == "tenant:t1"
    assert rec["message"]["query_hash"] == "abcd1234"


def test_elasticsearch_write_and_bulk(monkeypatch):
    calls = _capture(monkeypatch)
    es = ElasticsearchSink("https://es:9200", "audit", api_key="k")
    es.write(_entry())
    assert calls[-1]["url"] == "https://es:9200/audit/_doc"
    assert calls[-1]["headers"]["Authorization"] == "ApiKey k"
    es.write_batch([_entry(), _entry()])
    bulk = calls[-1]
    assert bulk["url"] == "https://es:9200/_bulk"
    lines = [ln for ln in bulk["data"].strip().split("\n") if ln]
    assert len(lines) == 4  # action + doc per entry
    assert json.loads(lines[0]) == {"index": {"_index": "audit"}}


def test_raise_for_status_propagates(monkeypatch):
    _capture(monkeypatch, status=503)
    with pytest.raises(RuntimeError):
        SplunkHECSink("https://s:8088", "t").write(_entry())


@pytest.mark.parametrize(
    "ctor",
    [
        lambda: SplunkHECSink("", "t"),
        lambda: SplunkHECSink("u", ""),
        lambda: DatadogLogsSink(""),
        lambda: ElasticsearchSink(""),
    ],
)
def test_missing_credentials_rejected(ctor):
    with pytest.raises(ValueError):
        ctor()


def test_empty_batch_is_noop(monkeypatch):
    calls = _capture(monkeypatch)
    SplunkHECSink("https://s:8088", "t").write_batch([])
    DatadogLogsSink("k").write_batch([])
    ElasticsearchSink("https://es:9200").write_batch([])
    assert calls == []


def test_integration_audit_logger_forwards_to_sink(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    from director_ai.core.safety.audit import AuditLogger

    logger = AuditLogger(path=str(tmp_path / "audit.jsonl"))
    logger.add_sink(ElasticsearchSink("https://es:9200", "audit"))
    logger.log_review("what is the secret password?", "I can't share that.", False, 0.2)
    # the SIEM received the entry, and it is tenant-safe (no raw query text)
    assert len(calls) == 1
    doc = calls[-1]["json"]
    assert "secret password" not in json.dumps(doc)
    assert "query_hash" in doc
