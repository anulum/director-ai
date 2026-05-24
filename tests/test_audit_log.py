# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Audit log contracts
"""Behavioural tests for compliance audit log persistence."""

import time

from director_ai.compliance.audit_log import AuditEntry, AuditLog


def _entry(prompt: str = "What is 2+2?") -> AuditEntry:
    return AuditEntry(
        prompt=prompt,
        response="4",
        model="test-model",
        provider="test-provider",
        score=0.95,
        approved=True,
        verdict_confidence=0.99,
        task_type="qa",
        domain="math",
        latency_ms=1.0,
        timestamp=time.time(),
    )


def test_audit_log_persists_and_queries_entries(tmp_path) -> None:
    log = AuditLog(str(tmp_path / "audit.db"))

    log.log(_entry())
    entries = log.query(limit=10)

    assert len(entries) == 1
    assert entries[0].prompt == "What is 2+2?"
    assert entries[0].approved is True


def test_audit_log_count_tracks_recorded_entries(tmp_path) -> None:
    log = AuditLog(str(tmp_path / "count.db"))

    assert log.count() == 0
    log.log(_entry("first"))
    log.log(_entry("second"))

    assert log.count() == 2


def test_audit_log_empty_query_is_stable(tmp_path) -> None:
    log = AuditLog(str(tmp_path / "empty.db"))

    assert log.query(limit=10) == []
