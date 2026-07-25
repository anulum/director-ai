# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — latency SLO qualification

"""Qualify one live Director-AI deployment against a declared latency SLO.

The qualifier deliberately makes a narrow claim: the measured deployment met
the supplied target for the supplied concurrency and deterministic workload.
It is not a hardware-independent runtime guarantee. Evidence contains aggregate
measurements and workload identity only; API keys and response bodies are never
serialised.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import requests

_SCHEMA_VERSION = "director-ai.latency-slo.v1"
_WORKLOAD = (
    (
        "Which deployment control is required?",
        "Every production deployment requires a documented readiness check.",
    ),
    (
        "What is the capital of France?",
        "Paris is the capital of France.",
    ),
    (
        "Does water boil at 50 degrees Celsius at sea level?",
        "No. At standard pressure, water boils at about 100 degrees Celsius.",
    ),
    (
        "What does DNA carry?",
        "DNA carries genetic information.",
    ),
)


@dataclass(frozen=True)
class LatencySLOConfig:
    """Inputs defining one reproducible deployment qualification run."""

    server_url: str = "http://127.0.0.1:8080"
    request_count: int = 100
    warmup_count: int = 10
    concurrency: int = 8
    timeout_ms: float = 5_000.0
    target_p95_ms: float = 500.0
    max_error_rate: float = 0.01
    tenant_id: str = ""
    api_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """Reject ambiguous, unsafe, or mathematically invalid inputs."""
        parsed = urlparse(self.server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("server_url must be an absolute http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("server_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("server_url must not contain a query or fragment")
        if (
            isinstance(self.request_count, bool)
            or not 1 <= self.request_count <= 1_000_000
        ):
            raise ValueError("request_count must be between 1 and 1000000")
        if isinstance(self.warmup_count, bool) or not 0 <= self.warmup_count <= 100_000:
            raise ValueError("warmup_count must be between 0 and 100000")
        if isinstance(self.concurrency, bool) or not 1 <= self.concurrency <= 1_024:
            raise ValueError("concurrency must be between 1 and 1024")
        if not math.isfinite(self.timeout_ms) or self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if not math.isfinite(self.target_p95_ms) or self.target_p95_ms <= 0:
            raise ValueError("target_p95_ms must be positive")
        if (
            not math.isfinite(self.max_error_rate)
            or not 0.0 <= self.max_error_rate <= 1.0
        ):
            raise ValueError("max_error_rate must be between 0 and 1")

    @property
    def base_url(self) -> str:
        """Return the normalised deployment base URL."""
        return self.server_url.rstrip("/")


@dataclass(frozen=True)
class _RequestMeasurement:
    latency_ms: float
    success: bool
    failure_category: str = ""


def _headers(config: LatencySLOConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if config.tenant_id:
        headers["X-Tenant-ID"] = config.tenant_id
    return headers


def _workload_hash() -> str:
    payload = json.dumps(_WORKLOAD, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return a linearly interpolated percentile, or ``None`` for no samples."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _request_once(config: LatencySLOConfig, index: int) -> _RequestMeasurement:
    prompt, response = _WORKLOAD[index % len(_WORKLOAD)]
    started = time.perf_counter()
    try:
        with requests.Session() as session:
            result = session.post(
                f"{config.base_url}/v1/review",
                headers=_headers(config),
                json={"prompt": prompt, "response": response},
                timeout=config.timeout_ms / 1_000.0,
                allow_redirects=False,
            )
        latency_ms = (time.perf_counter() - started) * 1_000.0
        if not 200 <= result.status_code < 300:
            return _RequestMeasurement(latency_ms, False, f"http_{result.status_code}")
        try:
            body = result.json()
        except ValueError:
            return _RequestMeasurement(latency_ms, False, "invalid_json")
        if not isinstance(body, dict) or not {
            "approved",
            "coherence",
        }.issubset(body):
            return _RequestMeasurement(latency_ms, False, "invalid_response")
        return _RequestMeasurement(latency_ms, True)
    except requests.Timeout:
        return _RequestMeasurement(
            (time.perf_counter() - started) * 1_000.0,
            False,
            "timeout",
        )
    except requests.RequestException:
        return _RequestMeasurement(
            (time.perf_counter() - started) * 1_000.0,
            False,
            "transport_error",
        )


def _execute_requests(
    config: LatencySLOConfig,
    count: int,
) -> tuple[list[_RequestMeasurement], float]:
    if count == 0:
        return [], 0.0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(config.concurrency, count)) as pool:
        futures = [pool.submit(_request_once, config, index) for index in range(count)]
        measurements = [future.result() for future in as_completed(futures)]
    return measurements, time.perf_counter() - started


def _probe_readiness(config: LatencySLOConfig) -> tuple[bool, str]:
    try:
        with requests.Session() as session:
            response = session.get(
                f"{config.base_url}/v1/ready",
                headers=_headers(config),
                timeout=config.timeout_ms / 1_000.0,
                allow_redirects=False,
            )
    except requests.Timeout:
        return False, "timeout"
    except requests.RequestException:
        return False, "transport_error"
    if not 200 <= response.status_code < 300:
        return False, f"http_{response.status_code}"
    try:
        body = response.json()
    except ValueError:
        return False, "invalid_json"
    if not isinstance(body, dict) or body.get("ready") is not True:
        return False, "not_ready"
    return True, ""


def _round_optional(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _failure_counts(
    measurements: list[_RequestMeasurement],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for measurement in measurements:
        if not measurement.success:
            category = measurement.failure_category or "unknown"
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def run_latency_slo(config: LatencySLOConfig) -> dict[str, Any]:
    """Run the qualification and return a secret-safe integrity packet."""
    import director_ai

    ready, readiness_failure = _probe_readiness(config)
    warmup: list[_RequestMeasurement] = []
    measured: list[_RequestMeasurement] = []
    duration_s = 0.0
    if ready:
        warmup, _ = _execute_requests(config, config.warmup_count)
        measured, duration_s = _execute_requests(config, config.request_count)

    successful_latencies = [item.latency_ms for item in measured if item.success]
    failed = len(measured) - len(successful_latencies)
    error_rate = failed / len(measured) if measured else 1.0
    p50_ms = _percentile(successful_latencies, 50.0)
    p95_ms = _percentile(successful_latencies, 95.0)
    p99_ms = _percentile(successful_latencies, 99.0)

    failures: list[str] = []
    warmup_failures = sum(not item.success for item in warmup)
    if not ready:
        failures.append(f"readiness:{readiness_failure}")
    if warmup_failures:
        failures.append("warmup_failures")
    if p95_ms is None:
        failures.append("no_successful_measurements")
    elif p95_ms > config.target_p95_ms:
        failures.append("p95_target_exceeded")
    if error_rate > config.max_error_rate:
        failures.append("error_rate_exceeded")

    content: dict[str, Any] = {
        "scope": {
            "claim": "deployment-specific operating-point qualification",
            "server_url": config.base_url,
            "endpoint": "/v1/review",
            "request_count": config.request_count,
            "warmup_count": config.warmup_count,
            "concurrency": config.concurrency,
            "timeout_ms": config.timeout_ms,
            "tenant_header_used": bool(config.tenant_id),
            "authentication_used": bool(config.api_key),
        },
        "targets": {
            "p95_ms_lte": config.target_p95_ms,
            "error_rate_lte": config.max_error_rate,
        },
        "workload": {
            "name": "director-ai-benign-review-v1",
            "pair_count": len(_WORKLOAD),
            "sha256": _workload_hash(),
            "bodies_recorded": False,
        },
        "readiness": {
            "passed": ready,
            "failure_category": readiness_failure or None,
        },
        "measurements": {
            "total": len(measured),
            "successful": len(successful_latencies),
            "failed": failed,
            "failure_categories": _failure_counts(measured),
            "warmup_failures": warmup_failures,
            "duration_s": round(duration_s, 3),
            "requests_per_second": round(
                len(measured) / duration_s if duration_s else 0.0,
                3,
            ),
            "error_rate": round(error_rate, 6),
            "latency_p50_ms": _round_optional(p50_ms),
            "latency_p95_ms": _round_optional(p95_ms),
            "latency_p99_ms": _round_optional(p99_ms),
            "percentile_method": "linear_interpolation",
        },
        "qualification": {
            "passed": not failures,
            "failure_reasons": failures,
        },
        "runtime": {
            "director_ai": director_ai.__version__,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "system_release": platform.release(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "limitations": [
            "Result applies only to this deployment, workload, and operating point.",
            "Network path and concurrent host load are part of the measurement.",
            "Qualification does not prove a universal or hard real-time guarantee.",
        ],
    }
    packet: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "content": content,
    }
    canonical = json.dumps(packet, sort_keys=True, separators=(",", ":"))
    packet["integrity"] = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    return packet


def verify_latency_slo_evidence(packet: dict[str, Any]) -> tuple[bool, str]:
    """Verify schema and content digest for a latency qualification packet."""
    if packet.get("schema_version") != _SCHEMA_VERSION:
        return False, "unsupported schema"
    integrity = packet.get("integrity")
    if not isinstance(packet.get("content"), dict) or not isinstance(integrity, dict):
        return False, "missing content or integrity"
    signed = {key: value for key, value in packet.items() if key != "integrity"}
    canonical = json.dumps(signed, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if integrity.get("algorithm") != "sha256" or integrity.get("digest") != expected:
        return False, "digest mismatch"
    return True, "verified"
