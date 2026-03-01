# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Metrics & Observability
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Prometheus-style metrics collection for production monitoring.

Usage::

    from director_ai.core.metrics import metrics

    metrics.inc("reviews_total")
    metrics.observe("coherence_score", 0.87)
    metrics.observe("review_duration_seconds", 0.142)
    print(metrics.prometheus_format())
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Counter:
    """Monotonically increasing counter."""

    value: float = 0.0
    labels: dict[str, float] = field(default_factory=dict)
    multi_labels: dict[str, float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, label: str = "") -> None:
        if label:
            self.labels[label] = self.labels.get(label, 0.0) + amount
        else:
            self.value += amount

    def inc_labeled(self, labels: dict[str, str], amount: float = 1.0) -> None:
        key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        self.multi_labels[key] = self.multi_labels.get(key, 0.0) + amount

    def total(self) -> float:
        return self.value + sum(self.labels.values()) + sum(self.multi_labels.values())


COHERENCE_SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
REVIEW_DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
BATCH_SIZE_BUCKETS = (1, 5, 10, 25, 50, 100, 500, 1000)
NLI_INFERENCE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
FACTUAL_RETRIEVAL_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
CHUNKED_NLI_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
NLI_CHUNK_COUNT_BUCKETS = (1, 2, 3, 4, 5, 8, 10, 15, 20)
HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
HISTOGRAM_MAX_SAMPLES = 100_000


@dataclass
class _Histogram:
    """Histogram with configurable bucket boundaries."""

    buckets: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    _values: list[float] = field(default_factory=list)
    max_samples: int = HISTOGRAM_MAX_SAMPLES

    def observe(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self.max_samples:
            # Keep the last half to preserve recent distribution
            self._values = self._values[-(self.max_samples // 2) :]

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def total(self) -> float:
        return sum(self._values) if self._values else 0.0

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def quantile(self, q: float) -> float:
        """Compute quantile (0.0-1.0). Returns 0 if empty."""
        if not self._values:
            return 0.0
        s = sorted(self._values)
        idx = int(q * (len(s) - 1))
        return s[idx]

    def bucket_counts(self) -> dict[str, int]:
        """Return cumulative bucket counts."""
        result = {}
        for b in self.buckets:
            result[f"le_{b}"] = sum(1 for v in self._values if v <= b)
        result["le_+Inf"] = len(self._values)
        return result


@dataclass
class _Gauge:
    """Point-in-time gauge value."""

    value: float = 0.0

    def set(self, value: float) -> None:
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


class MetricsCollector:
    """Thread-safe metrics collector with Prometheus-compatible output."""

    _METRIC_HELP: dict[str, str] = {
        "reviews_total": "Total review requests processed",
        "reviews_approved": "Reviews that passed coherence threshold",
        "reviews_rejected": "Reviews that failed coherence threshold",
        "halts_total": "Safety kernel halt events",
        "http_requests_total": "HTTP requests by method/endpoint/status",
        "coherence_score": "Coherence score distribution",
        "review_duration_seconds": "End-to-end review latency",
        "batch_size": "Batch request sizes",
        "nli_inference_seconds": "Single NLI inference latency",
        "factual_retrieval_seconds": "RAG retrieval latency",
        "chunked_nli_seconds": "Chunked NLI scoring latency",
        "nli_premise_chunks": "Premise chunk count per scoring call",
        "nli_hypothesis_chunks": "Hypothesis chunk count per scoring call",
        "http_request_duration_seconds": "HTTP request duration",
        "active_requests": "In-flight requests",
        "nli_model_loaded": "1 if NLI model is loaded",
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._counters: dict[str, _Counter] = {
            "reviews_total": _Counter(),
            "reviews_approved": _Counter(),
            "reviews_rejected": _Counter(),
            "halts_total": _Counter(),
        }
        self._histograms: dict[str, _Histogram] = {
            "coherence_score": _Histogram(buckets=COHERENCE_SCORE_BUCKETS),
            "review_duration_seconds": _Histogram(buckets=REVIEW_DURATION_BUCKETS),
            "batch_size": _Histogram(buckets=BATCH_SIZE_BUCKETS),
            "nli_inference_seconds": _Histogram(buckets=NLI_INFERENCE_BUCKETS),
            "factual_retrieval_seconds": _Histogram(buckets=FACTUAL_RETRIEVAL_BUCKETS),
            "chunked_nli_seconds": _Histogram(buckets=CHUNKED_NLI_BUCKETS),
            "nli_premise_chunks": _Histogram(buckets=NLI_CHUNK_COUNT_BUCKETS),
            "nli_hypothesis_chunks": _Histogram(buckets=NLI_CHUNK_COUNT_BUCKETS),
            "http_request_duration_seconds": _Histogram(buckets=HTTP_DURATION_BUCKETS),
        }
        self._gauges: dict[str, _Gauge] = {
            "active_requests": _Gauge(),
            "nli_model_loaded": _Gauge(),
        }

    def inc(self, name: str, amount: float = 1.0, label: str = "") -> None:
        """Increment a counter."""
        if not self.enabled:
            return
        with self._lock:
            if name not in self._counters:
                self._counters[name] = _Counter()
            self._counters[name].inc(amount, label)

    def inc_labeled(
        self,
        name: str,
        labels: dict[str, str],
        amount: float = 1.0,
    ) -> None:
        """Increment a counter with arbitrary label dict."""
        if not self.enabled:
            return
        with self._lock:
            if name not in self._counters:
                self._counters[name] = _Counter()
            self._counters[name].inc_labeled(labels, amount)

    def observe(self, name: str, value: float) -> None:
        """Record a histogram observation."""
        if not self.enabled:
            return
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = _Histogram()
            self._histograms[name].observe(value)

    def gauge_set(self, name: str, value: float) -> None:
        """Set a gauge value."""
        if not self.enabled:
            return
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = _Gauge()
            self._gauges[name].set(value)

    def gauge_inc(self, name: str, amount: float = 1.0) -> None:
        """Increment a gauge."""
        if not self.enabled:
            return
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = _Gauge()
            self._gauges[name].inc(amount)

    def gauge_dec(self, name: str, amount: float = 1.0) -> None:
        """Decrement a gauge."""
        if not self.enabled:
            return
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = _Gauge()
            self._gauges[name].dec(amount)

    def timer(self, histogram_name: str) -> _Timer:
        """Context manager that records elapsed time to a histogram."""
        return _Timer(self, histogram_name)

    def get_metrics(self) -> dict:
        """Return all metrics as a plain dict."""
        with self._lock:
            result: dict = {"counters": {}, "histograms": {}, "gauges": {}}
            for name, c in self._counters.items():
                result["counters"][name] = {
                    "total": c.total(),
                    "labels": dict(c.labels) if c.labels else {},
                    "multi_labels": dict(c.multi_labels) if c.multi_labels else {},
                }
            for name, h in self._histograms.items():
                result["histograms"][name] = {
                    "count": h.count,
                    "total": h.total,
                    "mean": h.mean,
                    "p50": h.quantile(0.5),
                    "p90": h.quantile(0.9),
                    "p99": h.quantile(0.99),
                }
            for name, g in self._gauges.items():
                result["gauges"][name] = g.value
            return result

    def prometheus_format(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for name, c in self._counters.items():
                fqn = f"director_ai_{name}"
                desc = self._METRIC_HELP.get(name, name)
                lines.append(f"# HELP {fqn} {desc}")
                lines.append(f"# TYPE {fqn} counter")
                if c.labels:
                    for label, val in c.labels.items():
                        lines.append(f'{fqn}{{reason="{label}"}} {val}')
                if c.multi_labels:
                    for label_str, val in c.multi_labels.items():
                        lines.append(f"{fqn}{{{label_str}}} {val}")
                if not c.labels and not c.multi_labels:
                    lines.append(f"{fqn} {c.value}")
            for name, h in self._histograms.items():
                fqn = f"director_ai_{name}"
                desc = self._METRIC_HELP.get(name, name)
                lines.append(f"# HELP {fqn} {desc}")
                lines.append(f"# TYPE {fqn} histogram")
                for bucket_name, count in h.bucket_counts().items():
                    le = bucket_name.replace("le_", "")
                    lines.append(f'{fqn}_bucket{{le="{le}"}} {count}')
                lines.append(f"{fqn}_count {h.count}")
                lines.append(f"{fqn}_sum {h.total}")
            for name, g in self._gauges.items():
                fqn = f"director_ai_{name}"
                desc = self._METRIC_HELP.get(name, name)
                lines.append(f"# HELP {fqn} {desc}")
                lines.append(f"# TYPE {fqn} gauge")
                lines.append(f"{fqn} {g.value}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            for c in self._counters.values():
                c.value = 0.0
                c.labels.clear()
                c.multi_labels.clear()
            for h in self._histograms.values():
                h._values.clear()
            for g in self._gauges.values():
                g.value = 0.0


class _Timer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str) -> None:
        self._collector = collector
        self._name = name
        self._start = 0.0

    def __enter__(self) -> _Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        elapsed = time.monotonic() - self._start
        self._collector.observe(self._name, elapsed)


# Module-level singleton
metrics = MetricsCollector()
