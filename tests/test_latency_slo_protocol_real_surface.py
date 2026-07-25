# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — latency SLO protocol real-surface tests
"""Real localhost protocol edge coverage for latency SLO qualification."""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Self, cast

import pytest

from director_ai import cli
from director_ai.core.observability.latency_slo import (
    LatencySLOConfig,
    run_latency_slo,
    verify_latency_slo_evidence,
)


@dataclass(frozen=True, slots=True)
class _ProtocolReply:
    """One explicit localhost protocol response."""

    status: int = 200
    body: bytes = b'{"approved":true,"coherence":1.0}'
    delay_s: float = 0.0
    disconnect: bool = False


@dataclass(slots=True)
class _ProtocolState:
    """State and observations for a real localhost HTTP boundary."""

    readiness: _ProtocolReply = field(
        default_factory=lambda: _ProtocolReply(body=b'{"ready":true}')
    )
    reviews: list[_ProtocolReply] = field(default_factory=list)
    review_count: int = 0
    review_headers: list[dict[str, str]] = field(default_factory=list)

    def next_review(self) -> _ProtocolReply:
        """Return the next planned review response, repeating the last one."""
        if not self.reviews:
            return _ProtocolReply()
        index = min(self.review_count, len(self.reviews) - 1)
        self.review_count += 1
        return self.reviews[index]


class _QualificationHTTPServer(ThreadingHTTPServer):
    """Threaded protocol server carrying typed qualification state."""

    daemon_threads = True
    state: _ProtocolState

    def __init__(self, state: _ProtocolState) -> None:
        super().__init__(("127.0.0.1", 0), _QualificationHandler)
        self.state = state


class _QualificationHandler(BaseHTTPRequestHandler):
    """Serve controlled readiness and review responses over real TCP."""

    protocol_version = "HTTP/1.1"

    def _reply(self, reply: _ProtocolReply) -> None:
        if reply.delay_s:
            time.sleep(reply.delay_s)
        if reply.disconnect:
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        self.send_response(reply.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply.body)))
        self.end_headers()
        try:
            self.wfile.write(reply.body)
        except (BrokenPipeError, ConnectionResetError):
            # Expected when exercising a bounded client timeout.
            return

    def do_GET(self) -> None:
        """Serve the readiness contract."""
        state = cast(_QualificationHTTPServer, self.server).state
        self._reply(state.readiness)

    def do_POST(self) -> None:
        """Record safe headers and serve the next review contract."""
        length = int(self.headers.get("Content-Length", "0") or "0")
        self.rfile.read(length)
        state = cast(_QualificationHTTPServer, self.server).state
        state.review_headers.append(dict(self.headers.items()))
        self._reply(state.next_review())

    def log_message(self, format: str, *args: object) -> None:
        """Suppress localhost protocol-server access logs."""


class _LiveProtocolServer:
    """Context manager for the controlled localhost HTTP boundary."""

    def __init__(self, state: _ProtocolState) -> None:
        self.state = state
        self._server = _QualificationHTTPServer(state)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = cast(tuple[str, int], self._server.socket.getsockname())
        return f"http://{host}:{port}"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.mark.parametrize(
    ("reply", "category"),
    [
        (_ProtocolReply(status=503, body=b'{"ready":false}'), "http_503"),
        (_ProtocolReply(body=b"not-json"), "invalid_json"),
        (_ProtocolReply(body=b'{"ready":false}'), "not_ready"),
        (_ProtocolReply(body=b"[]"), "not_ready"),
        (_ProtocolReply(body=b'{"ready":true}', delay_s=0.2), "timeout"),
    ],
)
def test_readiness_protocol_failures_are_classified_without_response_leakage(
    reply: _ProtocolReply,
    category: str,
) -> None:
    """Each real readiness failure should produce a stable safe category."""
    state = _ProtocolState(readiness=reply)
    with _LiveProtocolServer(state) as server:
        packet = run_latency_slo(
            LatencySLOConfig(
                server_url=server.base_url,
                request_count=1,
                warmup_count=0,
                concurrency=1,
                timeout_ms=50.0,
                target_p95_ms=500.0,
                max_error_rate=0.0,
            )
        )

    assert packet["content"]["readiness"] == {
        "passed": False,
        "failure_category": category,
    }
    assert packet["content"]["measurements"]["total"] == 0
    assert category in packet["content"]["qualification"]["failure_reasons"][0]


@pytest.mark.parametrize(
    ("reply", "category"),
    [
        (_ProtocolReply(status=429), "http_429"),
        (_ProtocolReply(body=b"not-json"), "invalid_json"),
        (_ProtocolReply(body=b'{"approved":true}'), "invalid_response"),
        (_ProtocolReply(body=b"[]"), "invalid_response"),
        (_ProtocolReply(delay_s=0.2), "timeout"),
        (_ProtocolReply(disconnect=True), "transport_error"),
    ],
)
def test_review_protocol_failures_count_towards_the_error_rate(
    reply: _ProtocolReply,
    category: str,
) -> None:
    """Malformed, failed, slow, and dropped reviews must fail qualification."""
    state = _ProtocolState(reviews=[reply])
    with _LiveProtocolServer(state) as server:
        packet = run_latency_slo(
            LatencySLOConfig(
                server_url=server.base_url,
                request_count=1,
                warmup_count=0,
                concurrency=1,
                timeout_ms=50.0,
                target_p95_ms=500.0,
                max_error_rate=0.0,
            )
        )

    measurements = packet["content"]["measurements"]
    assert measurements["successful"] == 0
    assert measurements["failed"] == 1
    assert measurements["failure_categories"] == {category: 1}
    assert packet["content"]["qualification"]["passed"] is False


def test_warmup_instability_and_measured_target_miss_both_fail_the_gate() -> None:
    """Warmup errors and slow successful measurements are independent failures."""
    state = _ProtocolState(
        reviews=[
            _ProtocolReply(status=503),
            _ProtocolReply(delay_s=0.02),
        ]
    )
    with _LiveProtocolServer(state) as server:
        packet = run_latency_slo(
            LatencySLOConfig(
                server_url=server.base_url,
                request_count=1,
                warmup_count=1,
                concurrency=1,
                timeout_ms=1_000.0,
                target_p95_ms=0.1,
                max_error_rate=0.0,
            )
        )

    measurements = packet["content"]["measurements"]
    failures = packet["content"]["qualification"]["failure_reasons"]
    assert measurements["warmup_failures"] == 1
    assert measurements["successful"] == 1
    assert "warmup_failures" in failures
    assert "p95_target_exceeded" in failures


def test_real_protocol_receives_auth_and_tenant_headers_without_evidence_values() -> (
    None
):
    """Credentials should reach the endpoint but only booleans reach evidence."""
    state = _ProtocolState()
    with _LiveProtocolServer(state) as server:
        packet = run_latency_slo(
            LatencySLOConfig(
                server_url=server.base_url,
                request_count=1,
                warmup_count=0,
                concurrency=1,
                timeout_ms=1_000.0,
                target_p95_ms=500.0,
                max_error_rate=0.0,
                tenant_id="tenant-private-value",
                api_key="credential-private-value",
            )
        )

    headers = state.review_headers[0]
    assert headers["Authorization"] == "Bearer credential-private-value"
    assert headers["X-Tenant-ID"] == "tenant-private-value"
    serialised = json.dumps(packet)
    assert "credential-private-value" not in serialised
    assert "tenant-private-value" not in serialised
    assert packet["content"]["qualification"]["passed"] is True


def test_evidence_verifier_rejects_schema_shape_and_algorithm_tampering() -> None:
    """Integrity verification should fail closed across top-level mutations."""
    state = _ProtocolState()
    with _LiveProtocolServer(state) as server:
        packet = run_latency_slo(
            LatencySLOConfig(
                server_url=server.base_url,
                request_count=1,
                warmup_count=0,
                concurrency=1,
                target_p95_ms=500.0,
                max_error_rate=0.0,
            )
        )

    wrong_schema = json.loads(json.dumps(packet))
    wrong_schema["schema_version"] = "unknown"
    assert verify_latency_slo_evidence(wrong_schema) == (
        False,
        "unsupported schema",
    )

    missing_content = json.loads(json.dumps(packet))
    missing_content.pop("content")
    assert verify_latency_slo_evidence(missing_content) == (
        False,
        "missing content or integrity",
    )

    malformed_integrity = json.loads(json.dumps(packet))
    malformed_integrity["integrity"] = "sha256"
    assert verify_latency_slo_evidence(malformed_integrity) == (
        False,
        "missing content or integrity",
    )

    wrong_algorithm = json.loads(json.dumps(packet))
    wrong_algorithm["integrity"]["algorithm"] = "sha512"
    assert verify_latency_slo_evidence(wrong_algorithm) == (
        False,
        "digest mismatch",
    )


def test_installed_cli_rejects_an_invalid_qualification_claim() -> None:
    """CLI validation should return argparse's configuration-error status."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["latency-slo", "--requests", "0"])

    assert exc_info.value.code == 2
