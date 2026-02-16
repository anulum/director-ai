# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Metrics & Observability Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import time

import pytest

from director_ai.core.metrics import MetricsCollector


@pytest.fixture
def collector():
    """Fresh MetricsCollector for each test."""
    return MetricsCollector()


class TestCounters:
    """Counter metric tests."""

    def test_increment(self, collector):
        collector.inc("reviews_total")
        collector.inc("reviews_total")
        m = collector.get_metrics()
        assert m["counters"]["reviews_total"]["total"] == 2.0

    def test_increment_by_amount(self, collector):
        collector.inc("reviews_total", 5.0)
        m = collector.get_metrics()
        assert m["counters"]["reviews_total"]["total"] == 5.0

    def test_labeled_counter(self, collector):
        collector.inc("halts_total", label="hard_limit")
        collector.inc("halts_total", label="hard_limit")
        collector.inc("halts_total", label="trend")
        m = collector.get_metrics()
        assert m["counters"]["halts_total"]["labels"]["hard_limit"] == 2.0
        assert m["counters"]["halts_total"]["labels"]["trend"] == 1.0
        assert m["counters"]["halts_total"]["total"] == 3.0

    def test_auto_create_counter(self, collector):
        collector.inc("new_counter")
        m = collector.get_metrics()
        assert "new_counter" in m["counters"]


class TestHistograms:
    """Histogram metric tests."""

    def test_observe(self, collector):
        collector.observe("coherence_score", 0.85)
        collector.observe("coherence_score", 0.90)
        m = collector.get_metrics()
        h = m["histograms"]["coherence_score"]
        assert h["count"] == 2
        assert abs(h["mean"] - 0.875) < 1e-6

    def test_quantile(self, collector):
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            collector.observe("coherence_score", v)
        m = collector.get_metrics()
        assert m["histograms"]["coherence_score"]["p50"] == 0.5

    def test_empty_histogram(self, collector):
        m = collector.get_metrics()
        h = m["histograms"]["coherence_score"]
        assert h["count"] == 0
        assert h["mean"] == 0.0

    def test_auto_create_histogram(self, collector):
        collector.observe("custom_histogram", 42.0)
        m = collector.get_metrics()
        assert "custom_histogram" in m["histograms"]


class TestGauges:
    """Gauge metric tests."""

    def test_set(self, collector):
        collector.gauge_set("active_requests", 5.0)
        m = collector.get_metrics()
        assert m["gauges"]["active_requests"] == 5.0

    def test_inc_dec(self, collector):
        collector.gauge_inc("active_requests")
        collector.gauge_inc("active_requests")
        collector.gauge_dec("active_requests")
        m = collector.get_metrics()
        assert m["gauges"]["active_requests"] == 1.0

    def test_auto_create_gauge(self, collector):
        collector.gauge_set("custom_gauge", 3.14)
        m = collector.get_metrics()
        assert m["gauges"]["custom_gauge"] == 3.14


class TestTimer:
    """Timer context manager tests."""

    def test_timer_records_duration(self, collector):
        with collector.timer("review_duration_seconds"):
            time.sleep(0.05)  # 50ms — safe for Windows timer resolution
        m = collector.get_metrics()
        h = m["histograms"]["review_duration_seconds"]
        assert h["count"] == 1
        assert h["mean"] >= 0.01  # At least 10ms


class TestPrometheusFormat:
    """Prometheus text exposition format tests."""

    def test_basic_format(self, collector):
        collector.inc("reviews_total", 3.0)
        collector.gauge_set("active_requests", 2.0)
        text = collector.prometheus_format()
        assert "director_ai_reviews_total 3.0" in text
        assert "director_ai_active_requests 2.0" in text

    def test_labeled_format(self, collector):
        collector.inc("halts_total", label="hard_limit")
        text = collector.prometheus_format()
        assert 'director_ai_halts_total{reason="hard_limit"} 1.0' in text

    def test_histogram_buckets(self, collector):
        collector.observe("coherence_score", 0.5)
        text = collector.prometheus_format()
        assert "director_ai_coherence_score_count 1" in text
        assert "director_ai_coherence_score_sum 0.5" in text
        assert "director_ai_coherence_score_bucket" in text


class TestReset:
    """Reset functionality test."""

    def test_reset_clears_all(self, collector):
        collector.inc("reviews_total", 10.0)
        collector.observe("coherence_score", 0.9)
        collector.gauge_set("active_requests", 5.0)
        collector.reset()
        m = collector.get_metrics()
        assert m["counters"]["reviews_total"]["total"] == 0.0
        assert m["histograms"]["coherence_score"]["count"] == 0
        assert m["gauges"]["active_requests"] == 0.0
