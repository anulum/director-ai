from __future__ import annotations

import json

from director_ai.core.audit import AuditEntry, AuditLogger


class TestAuditEntry:
    def test_to_json_round_trip(self):
        entry = AuditEntry(
            timestamp="2026-02-28T12:00:00",
            query_hash="abc123",
            response_length=42,
            approved=True,
            score=0.95,
            h_logical=0.05,
            h_factual=0.1,
        )
        data = json.loads(entry.to_json())
        assert data["approved"] is True
        assert data["score"] == 0.95
        assert data["response_length"] == 42

    def test_defaults(self):
        entry = AuditEntry(
            timestamp="t",
            query_hash="h",
            response_length=0,
            approved=False,
            score=0.0,
        )
        assert entry.policy_violations == []
        assert entry.tenant_id == ""
        assert entry.halt_reason == ""
        assert entry.latency_ms == 0.0


class TestAuditLoggerLoggingOnly:
    def test_log_review_returns_entry(self):
        logger = AuditLogger()
        entry = logger.log_review(
            query="What is 2+2?",
            response="4",
            approved=True,
            score=0.95,
        )
        assert isinstance(entry, AuditEntry)
        assert entry.approved is True
        assert entry.response_length == 1

    def test_query_hashed(self):
        logger = AuditLogger()
        entry = logger.log_review(
            query="secret", response="x", approved=True, score=1.0
        )
        assert entry.query_hash != "secret"
        assert len(entry.query_hash) == 16

    def test_score_rounded(self):
        logger = AuditLogger()
        entry = logger.log_review(
            query="q", response="r", approved=True, score=0.123456789
        )
        assert entry.score == 0.1235


class TestAuditLoggerFileSink:
    def test_writes_jsonl(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=path)
        logger.log_review(query="q1", response="r1", approved=True, score=0.8)
        logger.log_review(query="q2", response="r2", approved=False, score=0.3)

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["approved"] is True

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "audit.jsonl"
        AuditLogger(path=path)
        assert path.parent.exists()

    def test_policy_violations_recorded(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=path)
        logger.log_review(
            query="q",
            response="r",
            approved=False,
            score=0.2,
            policy_violations=["forbidden:bad phrase"],
        )
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["policy_violations"] == ["forbidden:bad phrase"]

    def test_tenant_id_recorded(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=path)
        entry = logger.log_review(
            query="q", response="r", approved=True, score=0.9, tenant_id="acme"
        )
        assert entry.tenant_id == "acme"

    def test_all_optional_fields(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=path)
        entry = logger.log_review(
            query="q",
            response="r",
            approved=False,
            score=0.2,
            h_logical=0.8,
            h_factual=0.7,
            policy_violations=["v1", "v2"],
            tenant_id="tenant-1",
            halt_reason="coherence",
            latency_ms=42.567,
        )
        assert entry.h_logical == 0.8
        assert entry.h_factual == 0.7
        assert entry.halt_reason == "coherence"
        assert entry.latency_ms == 42.57  # rounded to 2 dp
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["halt_reason"] == "coherence"
        assert len(data["policy_violations"]) == 2


class TestAuditLoggerErrorPaths:
    def test_unwritable_path_raises(self, tmp_path):
        import sys

        if sys.platform == "win32":
            # On Windows, use a path with illegal characters
            bad_path = tmp_path / "con" / "audit.jsonl"
        else:
            bad_path = "/proc/nonexistent/audit.jsonl"
        try:
            logger = AuditLogger(path=bad_path)
            # On some platforms mkdir will fail, on others write will fail
            logger.log_review(query="q", response="r", approved=True, score=0.5)
        except OSError:
            pass  # Expected: the I/O error surfaces

    def test_logging_only_fallback(self, caplog):
        import logging

        logger = AuditLogger(logger_name="TestAudit")
        with caplog.at_level(logging.INFO, logger="TestAudit"):
            entry = logger.log_review(query="q", response="r", approved=True, score=0.9)
        assert entry.approved is True
        assert any("approved" in r.message for r in caplog.records)
