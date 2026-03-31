# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for compliance audit log pipeline (STRONG)."""

from __future__ import annotations

import time

from director_ai.compliance.audit_log import AuditEntry, AuditLog


def _make_entry(**kwargs) -> AuditEntry:
    defaults = {
        "prompt": "What is X?",
        "response": "X is Y.",
        "model": "gpt-4o",
        "provider": "openai",
        "score": 0.85,
        "approved": True,
        "verdict_confidence": 0.9,
        "task_type": "qa",
        "domain": "",
        "latency_ms": 15.0,
        "timestamp": time.time(),
    }
    defaults.update(kwargs)
    return AuditEntry(**defaults)


class TestAuditLogBasic:
    def test_empty_log(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        assert log.count() == 0
        assert log.query() == []
        log.close()

    def test_log_and_query(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        log.log(_make_entry())
        assert log.count() == 1
        entries = log.query()
        assert len(entries) == 1
        assert entries[0].model == "gpt-4o"
        assert entries[0].approved is True
        log.close()

    def test_multiple_entries(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        for i in range(10):
            log.log(_make_entry(score=i / 10))
        assert log.count() == 10
        log.close()

    def test_double_close(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        log.close()
        log.close()


class TestAuditLogFilters:
    def test_filter_by_model(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        log.log(_make_entry(model="gpt-4o"))
        log.log(_make_entry(model="claude-4"))
        log.log(_make_entry(model="gpt-4o"))
        assert log.count(model="gpt-4o") == 2
        assert log.count(model="claude-4") == 1
        entries = log.query(model="claude-4")
        assert len(entries) == 1
        assert entries[0].model == "claude-4"
        log.close()

    def test_filter_by_time(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        now = time.time()
        log.log(_make_entry(timestamp=now - 3600))
        log.log(_make_entry(timestamp=now - 1800))
        log.log(_make_entry(timestamp=now))
        entries = log.query(since=now - 2000)
        assert len(entries) == 2
        log.close()

    def test_filter_by_domain(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        log.log(_make_entry(domain="medical"))
        log.log(_make_entry(domain="finance"))
        entries = log.query(domain="medical")
        assert len(entries) == 1
        log.close()

    def test_limit(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        for _i in range(20):
            log.log(_make_entry())
        entries = log.query(limit=5)
        assert len(entries) == 5
        log.close()

    def test_human_override(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        log.log(_make_entry(human_override=True))
        log.log(_make_entry(human_override=False))
        log.log(_make_entry(human_override=None))
        entries = log.query()
        overrides = [e for e in entries if e.human_override is not None]
        assert len(overrides) == 2
        log.close()
