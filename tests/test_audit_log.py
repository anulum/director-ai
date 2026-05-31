# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Audit log contracts
"""Behavioural tests for compliance audit log persistence."""

from __future__ import annotations

import time

from director_ai.compliance.audit_log import AuditEntry, AuditLog


def _entry(
    prompt: str = "What is 2+2?",
    *,
    model: str = "test-model",
    domain: str = "math",
    tenant_id: str = "",
    timestamp: float | None = None,
    human_override: bool | None = None,
) -> AuditEntry:
    return AuditEntry(
        prompt=prompt,
        response="4",
        model=model,
        provider="test-provider",
        score=0.95,
        approved=True,
        verdict_confidence=0.99,
        task_type="qa",
        domain=domain,
        latency_ms=1.0,
        timestamp=time.time() if timestamp is None else timestamp,
        tenant_id=tenant_id,
        human_override=human_override,
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


def test_audit_log_query_filters_by_time_model_domain_and_tenant(tmp_path) -> None:
    log = AuditLog(str(tmp_path / "filters.db"))
    log.log(
        _entry(
            "old tenant",
            model="alpha",
            domain="math",
            tenant_id="tenant-a",
            timestamp=10.0,
        )
    )
    log.log(
        _entry(
            "selected",
            model="beta",
            domain="science",
            tenant_id="tenant-b",
            timestamp=20.0,
            human_override=True,
        )
    )
    log.log(
        _entry(
            "new wrong tenant",
            model="beta",
            domain="science",
            tenant_id="tenant-c",
            timestamp=30.0,
        )
    )

    entries = log.query(
        since=15.0,
        until=25.0,
        model="beta",
        domain="science",
        tenant_id="tenant-b",
        limit=5,
    )
    unbounded = log.query(model="beta", domain="science")

    assert [entry.prompt for entry in entries] == ["selected"]
    assert entries[0].tenant_id == "tenant-b"
    assert entries[0].human_override is True
    assert [entry.prompt for entry in unbounded] == ["new wrong tenant", "selected"]


def test_audit_log_count_filters_by_since_and_model(tmp_path) -> None:
    log = AuditLog(str(tmp_path / "count_filters.db"))
    log.log(_entry("old alpha", model="alpha", timestamp=10.0))
    log.log(_entry("new alpha", model="alpha", timestamp=20.0))
    log.log(_entry("new beta", model="beta", timestamp=30.0))

    assert log.count(since=15.0, model="alpha") == 1
    assert log.count(since=15.0) == 2
    assert log.count(model="beta") == 1


def test_audit_log_closed_connection_paths_are_noops(tmp_path) -> None:
    log = AuditLog(str(tmp_path / "closed.db"))
    log.close()

    log.log(_entry())

    assert log.query() == []
    assert log.count() == 0
    log.close()
