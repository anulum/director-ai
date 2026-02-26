from __future__ import annotations

import pytest

from director_ai.core.stats import StatsStore


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "test_stats.db"
    s = StatsStore(db_path=db)
    yield s
    s.close()


class TestStatsStore:
    def test_empty_summary(self, store):
        s = store.summary()
        assert s["total"] == 0
        assert s["approved"] == 0
        assert s["avg_score"] is None

    def test_record_and_summary(self, store):
        store.record_review(approved=True, score=0.85, h_logical=0.1, h_factual=0.15)
        store.record_review(
            approved=False, score=0.45, h_logical=0.6, h_factual=0.4, halted=True
        )
        s = store.summary()
        assert s["total"] == 2
        assert s["approved"] == 1
        assert s["rejected"] == 1
        assert s["halted"] == 1
        assert s["avg_score"] == pytest.approx(0.65, abs=0.01)

    def test_summary_since_filter(self, store):
        store.record_review(approved=True, score=0.9)
        store.record_review(approved=False, score=0.3)
        s = store.summary(since=0)
        assert s["total"] == 2

    def test_hourly_breakdown(self, store):
        store.record_review(approved=True, score=0.8)
        store.record_review(approved=True, score=0.7)
        breakdown = store.hourly_breakdown(days=1)
        assert len(breakdown) >= 1
        assert breakdown[0]["total"] == 2

    def test_record_with_latency(self, store):
        store.record_review(approved=True, score=0.9, latency_ms=42.5)
        s = store.summary()
        assert s["avg_latency_ms"] == pytest.approx(42.5, abs=0.1)

    def test_db_created_at_path(self, tmp_path):
        db = tmp_path / "sub" / "stats.db"
        s = StatsStore(db_path=db)
        s.record_review(approved=True, score=0.5)
        s.close()
        assert db.exists()

    def test_multiple_records(self, store):
        for i in range(10):
            store.record_review(approved=i % 2 == 0, score=0.5 + i * 0.05)
        s = store.summary()
        assert s["total"] == 10
        assert s["approved"] == 5
