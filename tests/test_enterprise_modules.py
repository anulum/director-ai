# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for enterprise modules pipeline.

Covers: PostgresAuditSink, Redis wrappers, tenant isolation, pipeline
integration with server, and performance documentation.
"""

from __future__ import annotations

import json

import pytest

from director_ai.core.audit import AuditEntry
from director_ai.enterprise.audit_pg import SCHEMA_VERSION, PostgresAuditSink


def _make_entry(**overrides) -> AuditEntry:
    defaults = dict(
        timestamp="2026-03-07T12:00:00Z",
        query_hash="abc123",
        response_length=42,
        approved=True,
        score=0.95,
        h_logical=0.1,
        h_factual=0.05,
        policy_violations=[],
        tenant_id="tenant_a",
        halt_reason="",
        latency_ms=12.5,
    )
    defaults.update(overrides)
    return AuditEntry(**defaults)


class TestPostgresAuditSinkSchema:
    def test_sqlite_creates_table_and_indexes(self):
        sink = PostgresAuditSink("sqlite://")
        cur = sink._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert "director_audit_logs" in tables
        assert "_schema_version" in tables
        cur.close()

    def test_schema_version_reaches_current(self):
        sink = PostgresAuditSink("sqlite://")
        cur = sink._conn.cursor()
        cur.execute("SELECT MAX(version) FROM _schema_version")
        assert cur.fetchone()[0] == SCHEMA_VERSION
        cur.close()

    def test_idempotent_migration(self):
        sink = PostgresAuditSink("sqlite://")
        sink._migrate()
        sink._migrate()
        cur = sink._conn.cursor()
        cur.execute("SELECT MAX(version) FROM _schema_version")
        assert cur.fetchone()[0] == SCHEMA_VERSION
        cur.close()

    @pytest.mark.parametrize(
        "table_name",
        [
            "audit_logs; DROP TABLE users;--",
            "audit logs",
            "audit.logs",
            "1audit_logs",
            "audit_logs/*comment*/",
            'audit_logs"quoted',
        ],
    )
    def test_rejects_injection_looking_table_names_before_connect(
        self,
        monkeypatch,
        table_name,
    ):
        def forbidden_connect(self):
            raise AssertionError("invalid table name reached connection setup")

        monkeypatch.setattr(PostgresAuditSink, "_connect", forbidden_connect)

        with pytest.raises(ValueError, match="Invalid table_name"):
            PostgresAuditSink("sqlite://", table_name=table_name)

    def test_custom_safe_table_name_is_quoted_and_usable(self):
        sink = PostgresAuditSink("sqlite://", table_name="tenant_audit_2026")
        sink.write(_make_entry(tenant_id="t1"))

        assert sink.count("t1") == 1


class TestPostgresAuditSinkWrite:
    def test_write_single_entry(self):
        sink = PostgresAuditSink("sqlite://")
        entry = _make_entry()
        sink.write(entry)
        assert sink.count() == 1

    def test_write_batch(self):
        sink = PostgresAuditSink("sqlite://")
        entries = [_make_entry(query_hash=f"h{i}") for i in range(5)]
        written = sink.write_batch(entries)
        assert written == 5
        assert sink.count() == 5

    def test_write_batch_empty(self):
        sink = PostgresAuditSink("sqlite://")
        assert sink.write_batch([]) == 0

    def test_write_preserves_fields(self):
        sink = PostgresAuditSink("sqlite://")
        entry = _make_entry(
            tenant_id="t1",
            approved=False,
            policy_violations=["v1", "v2"],
            halt_reason="threshold",
        )
        sink.write(entry)
        rows = sink.query(tenant_id="t1")
        assert len(rows) == 1
        row = rows[0]
        assert row["tenant_id"] == "t1"
        assert row["approved"] == 0  # SQLite stores as int
        assert json.loads(row["policy_violations"]) == ["v1", "v2"]
        assert row["halt_reason"] == "threshold"

    def test_write_preserves_kb_snapshot_fields(self):
        sink = PostgresAuditSink("sqlite://")
        entry = _make_entry(
            kb_snapshot_root="b" * 64,
            kb_snapshot_revision=7,
            kb_snapshot_record_count=5,
            kb_snapshot_retraction_count=2,
            kb_snapshot_replacement_count=1,
        )

        sink.write(entry)
        row = sink.query()[0]

        assert row["kb_snapshot_root"] == "b" * 64
        assert row["kb_snapshot_revision"] == 7
        assert row["kb_snapshot_record_count"] == 5
        assert row["kb_snapshot_retraction_count"] == 2
        assert row["kb_snapshot_replacement_count"] == 1

    def test_cursor_failure_does_not_mask_original_error_or_leak_connection(self):
        class BrokenConnection:
            rolled_back = False

            def cursor(self):
                raise RuntimeError("cursor unavailable")

            def rollback(self):
                self.rolled_back = True

        sink = PostgresAuditSink("sqlite://")
        conn = BrokenConnection()
        returned: list[object] = []
        sink._get_conn = lambda: conn  # type: ignore[method-assign]
        sink._put_conn = lambda value: returned.append(value)  # type: ignore[method-assign]

        sink.write(_make_entry())

        assert conn.rolled_back is True
        assert returned == [conn]


class TestPostgresAuditSinkQuery:
    def test_query_by_tenant(self):
        sink = PostgresAuditSink("sqlite://")
        sink.write(_make_entry(tenant_id="a"))
        sink.write(_make_entry(tenant_id="b"))
        sink.write(_make_entry(tenant_id="a"))
        assert len(sink.query(tenant_id="a")) == 2
        assert len(sink.query(tenant_id="b")) == 1

    def test_query_by_approved(self):
        sink = PostgresAuditSink("sqlite://")
        sink.write(_make_entry(approved=True))
        sink.write(_make_entry(approved=False))
        sink.write(_make_entry(approved=True))
        assert len(sink.query(approved=True)) == 2
        assert len(sink.query(approved=False)) == 1

    def test_query_limit(self):
        sink = PostgresAuditSink("sqlite://")
        for i in range(10):
            sink.write(_make_entry(query_hash=f"h{i}"))
        assert len(sink.query(limit=3)) == 3

    def test_count_by_tenant(self):
        sink = PostgresAuditSink("sqlite://")
        sink.write(_make_entry(tenant_id="x"))
        sink.write(_make_entry(tenant_id="y"))
        sink.write(_make_entry(tenant_id="x"))
        assert sink.count(tenant_id="x") == 2
        assert sink.count(tenant_id="y") == 1
        assert sink.count() == 3

    def test_no_connection_returns_empty(self):
        sink = PostgresAuditSink("sqlite://")
        sink._conn = None
        assert sink.query() == []
        assert sink.count() == 0
        sink.write(_make_entry())  # no-op, no crash

    def test_query_and_count_cursor_failures_return_connection(self):
        class BrokenConnection:
            def cursor(self):
                raise RuntimeError("cursor unavailable")

        sink = PostgresAuditSink("sqlite://")
        conn = BrokenConnection()
        returned: list[object] = []
        sink._get_conn = lambda: conn  # type: ignore[method-assign]
        sink._put_conn = lambda value: returned.append(value)  # type: ignore[method-assign]

        assert sink.query() == []
        assert sink.count() == 0
        assert returned == [conn, conn]


class TestRedisGroundTruthStore:
    """Tests for RedisGroundTruthStore using fakeredis."""

    def _make_store(self):
        try:
            import fakeredis
        except ImportError:
            import pytest

            pytest.skip("fakeredis not installed")

        from director_ai.enterprise.redis import RedisGroundTruthStore

        store = RedisGroundTruthStore.__new__(RedisGroundTruthStore)
        store.prefix = "test:facts:"
        store.redis_url = "redis://fake"
        store.facts = {}
        store.logger = __import__("logging").getLogger("test")
        store.client = fakeredis.FakeRedis(decode_responses=True)
        return store

    def test_add_and_retrieve(self):
        store = self._make_store()
        store.add("sky", "The sky is blue.", tenant_id="t1")
        result = store.retrieve_context("sky color", tenant_id="t1")
        assert result is not None
        assert "blue" in result

    def test_tenant_isolation(self):
        store = self._make_store()
        store.add("fact", "value_a", tenant_id="a")
        store.add("fact", "value_b", tenant_id="b")
        assert store.count(tenant_id="a") == 1
        assert store.count(tenant_id="b") == 1
        ctx_a = store.retrieve_context("fact", tenant_id="a")
        ctx_b = store.retrieve_context("fact", tenant_id="b")
        assert ctx_a == "value_a"
        assert ctx_b == "value_b"

    def test_add_many(self):
        store = self._make_store()
        added = store.add_many({"k1": "v1", "k2": "v2", "k3": "v3"}, tenant_id="t")
        assert added == 3
        assert store.count(tenant_id="t") == 3

    def test_add_many_empty(self):
        store = self._make_store()
        assert store.add_many({}, tenant_id="t") == 0

    def test_retrieve_no_match(self):
        store = self._make_store()
        store.add("sky", "blue", tenant_id="t1")
        assert store.retrieve_context("xyz123", tenant_id="t1") is None


class TestRedisScoreCache:
    """Tests for RedisScoreCache using fakeredis."""

    def _make_cache(self):
        try:
            import fakeredis
        except ImportError:
            import pytest

            pytest.skip("fakeredis not installed")

        from director_ai.enterprise.redis import RedisScoreCache

        cache = RedisScoreCache.__new__(RedisScoreCache)
        cache.prefix = "test:cache:"
        cache.redis_url = "redis://fake"
        cache._ttl = 300.0
        cache._max_size = 1024
        cache._store = {}
        cache._lock = __import__("threading").Lock()
        cache._generation = 0
        cache.hits = 0
        cache.misses = 0
        cache.client = fakeredis.FakeRedis(decode_responses=True)
        return cache

    def test_put_and_get(self):
        cache = self._make_cache()
        cache.put("hello world", "prefix", 0.9, 0.1, 0.05)
        entry = cache.get("hello world", "prefix")
        assert entry is not None
        assert entry.score == 0.9

    def test_miss_returns_none(self):
        cache = self._make_cache()
        assert cache.get("nonexistent", "p") is None
        assert cache.misses == 1

    def test_generation_mismatch_evicts(self):
        cache = self._make_cache()
        cache.put("q", "p", 0.8, 0.1, 0.1)
        cache._generation = 1
        assert cache.get("q", "p") is None

    def test_clear(self):
        cache = self._make_cache()
        cache.put("q1", "p", 0.5, 0.1, 0.1)
        cache.put("q2", "p", 0.6, 0.1, 0.1)
        assert cache.size >= 2
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0
