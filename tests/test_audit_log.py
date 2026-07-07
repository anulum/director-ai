# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Audit log contracts
"""Behavioural tests for compliance audit log persistence."""

from __future__ import annotations

import sqlite3
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


# --- tamper-evident hash chain (SEC-2) -------------------------------------


def test_audit_log_seals_entries_into_verifiable_chain(tmp_path) -> None:
    path = str(tmp_path / "seal.db")
    log = AuditLog(path, hmac_secret="s")
    log.log(_entry("a"))
    log.log(_entry("b"))

    assert log.verify_chain() == (True, None)

    con = sqlite3.connect(path)
    rows = con.execute(
        "SELECT prev_hash, entry_hash, chain_tag FROM audit_log ORDER BY id"
    ).fetchall()
    con.close()
    assert rows[0][0] == "0" * 64  # genesis parent
    assert len(rows[0][1]) == 64 and len(rows[0][2]) == 64
    assert rows[1][0] == rows[0][1]  # second.prev_hash == first.entry_hash


def test_audit_log_empty_chain_verifies(tmp_path) -> None:
    log = AuditLog(str(tmp_path / "empty_chain.db"), hmac_secret="s")
    assert log.verify_chain() == (True, None)


def test_audit_log_verify_detects_payload_tamper(tmp_path) -> None:
    path = str(tmp_path / "tamper.db")
    log = AuditLog(path, hmac_secret="s")
    log.log(_entry("original"))
    log.log(_entry("second"))

    con = sqlite3.connect(path)
    con.execute("UPDATE audit_log SET prompt = 'forged' WHERE id = 1")
    con.commit()
    con.close()

    assert log.verify_chain() == (False, 1)


def test_audit_log_verify_detects_deleted_row(tmp_path) -> None:
    path = str(tmp_path / "delete.db")
    log = AuditLog(path, hmac_secret="s")
    for prompt in ("a", "b", "c"):
        log.log(_entry(prompt))

    con = sqlite3.connect(path)
    con.execute("DELETE FROM audit_log WHERE id = 2")
    con.commit()
    con.close()

    # id 3's prev_hash no longer matches the running chain head after 2 is gone.
    assert log.verify_chain() == (False, 3)


def test_audit_log_verify_detects_forged_tag(tmp_path) -> None:
    path = str(tmp_path / "tag.db")
    log = AuditLog(path, hmac_secret="s")
    log.log(_entry("a"))

    con = sqlite3.connect(path)
    con.execute("UPDATE audit_log SET chain_tag = 'deadbeef' WHERE id = 1")
    con.commit()
    con.close()

    assert log.verify_chain() == (False, 1)


def test_audit_log_wrong_secret_fails_tag_verification(tmp_path) -> None:
    path = str(tmp_path / "wrongkey.db")
    writer = AuditLog(path, hmac_secret="right")
    writer.log(_entry("a"))
    writer.close()

    # The content hash and linkage still check out, but the HMAC tag needs the
    # original secret — a reader with the wrong key cannot forge or verify it.
    reader = AuditLog(path, hmac_secret="wrong")
    assert reader.verify_chain() == (False, 1)


def test_audit_log_chain_continues_across_restart(tmp_path) -> None:
    path = str(tmp_path / "restart.db")
    first = AuditLog(path, hmac_secret="s")
    first.log(_entry("a"))
    first.log(_entry("b"))
    first.close()

    reopened = AuditLog(path, hmac_secret="s")
    reopened.log(_entry("c"))

    assert reopened.count() == 3
    assert reopened.verify_chain() == (True, None)


def test_audit_log_seals_new_rows_on_legacy_unsealed_database(tmp_path) -> None:
    path = str(tmp_path / "legacy.db")
    # A pre-seal database: the original schema with no chain columns.
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE audit_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT NOT NULL, "
        "response TEXT NOT NULL, model TEXT NOT NULL DEFAULT '', "
        "provider TEXT NOT NULL DEFAULT '', score REAL NOT NULL, "
        "approved INTEGER NOT NULL, verdict_confidence REAL NOT NULL DEFAULT 0.0, "
        "task_type TEXT NOT NULL DEFAULT '', domain TEXT NOT NULL DEFAULT '', "
        "latency_ms REAL NOT NULL DEFAULT 0.0, tenant_id TEXT NOT NULL DEFAULT '', "
        "human_override INTEGER, timestamp REAL NOT NULL)"
    )
    con.execute(
        "INSERT INTO audit_log (prompt, response, score, approved, timestamp) "
        "VALUES ('legacy', 'x', 0.5, 1, 5.0)"
    )
    con.commit()
    con.close()

    # Opening migrates the schema (ALTER TABLE adds the chain columns) and seals
    # subsequent rows; the legacy NULL-hash row is skipped by verify_chain.
    log = AuditLog(path, hmac_secret="s")
    log.log(_entry("sealed"))

    assert log.count() == 2
    assert log.verify_chain() == (True, None)
