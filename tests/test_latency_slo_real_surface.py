# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — latency SLO real-surface tests
"""Real TCP/FastAPI coverage for the latency SLO qualification gate."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Self

import pytest
import uvicorn

from director_ai import cli
from director_ai.core.config import DirectorConfig
from director_ai.core.observability.latency_slo import (
    LatencySLOConfig,
    run_latency_slo,
    verify_latency_slo_evidence,
)
from director_ai.server import create_app


class _LiveDirectorServer:
    """Run the production FastAPI factory through Uvicorn on a real TCP socket."""

    def __init__(self) -> None:
        # ``from_profile`` reapplies the explicit profile values after generic
        # mode normalization, which is the supported way to request a no-model
        # rules deployment.
        config = DirectorConfig.from_profile("rules")
        config.review_queue_enabled = True
        config.review_queue_max_batch = 8
        config.review_queue_flush_timeout_ms = 2.0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(128)
        self._socket.setblocking(False)
        uvicorn_config = uvicorn.Config(
            create_app(config),
            log_level="error",
            lifespan="on",
        )
        self._server = uvicorn.Server(uvicorn_config)
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [self._socket]},
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        host, port = self._socket.getsockname()
        return f"http://{host}:{port}"

    def __enter__(self) -> Self:
        self._thread.start()
        # The first scorer import can initialise optional ML runtimes even for
        # the rules profile, so allow a bounded cold-start window.
        deadline = time.monotonic() + 60.0
        while not self._server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self._server.started:
            self._server.should_exit = True
            self._thread.join(timeout=10)
            self._socket.close()
            raise RuntimeError("Uvicorn did not start")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
        self._socket.close()


def _qualification_config(server_url: str, *, api_key: str = "") -> LatencySLOConfig:
    return LatencySLOConfig(
        server_url=server_url,
        request_count=12,
        warmup_count=4,
        concurrency=4,
        timeout_ms=5_000.0,
        target_p95_ms=5_000.0,
        max_error_rate=0.0,
        tenant_id="qualification-tenant",
        api_key=api_key,
    )


def test_live_deployment_qualification_emits_verified_secret_safe_evidence() -> None:
    """The gate should qualify a real ReviewQueue-backed HTTP deployment."""
    secret = "must-not-appear-in-evidence"
    config: LatencySLOConfig
    with _LiveDirectorServer() as server:
        config = _qualification_config(server.base_url, api_key=secret)
        packet = run_latency_slo(config)

    content = packet["content"]
    measurements = content["measurements"]
    assert content["readiness"] == {"passed": True, "failure_category": None}
    assert measurements["total"] == 12
    assert measurements["successful"] == 12
    assert measurements["failed"] == 0
    assert measurements["latency_p95_ms"] is not None
    assert content["qualification"] == {"passed": True, "failure_reasons": []}
    assert content["workload"]["bodies_recorded"] is False
    assert secret not in json.dumps(packet)
    assert secret not in repr(config)
    assert verify_latency_slo_evidence(packet) == (True, "verified")
    packet["generated_at"] = "tampered"
    assert verify_latency_slo_evidence(packet) == (False, "digest mismatch")


def test_installed_cli_dispatch_writes_live_qualification_packet(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Top-level CLI dispatch should exercise the live server and write evidence."""
    output = tmp_path / "qualification.json"
    with _LiveDirectorServer() as server:
        cli.main(
            [
                "latency-slo",
                "--server",
                server.base_url,
                "--requests",
                "8",
                "--warmup",
                "2",
                "--concurrency",
                "4",
                "--target-p95-ms",
                "5000",
                "--max-error-rate",
                "0",
                "--output",
                str(output),
            ]
        )

    packet = json.loads(output.read_text(encoding="utf-8"))
    assert packet["content"]["qualification"]["passed"] is True
    assert packet["content"]["measurements"]["total"] == 8
    assert verify_latency_slo_evidence(packet) == (True, "verified")
    terminal = capsys.readouterr().out
    assert "Latency SLO: QUALIFIED" in terminal
    assert str(output) in terminal


def test_unreachable_deployment_fails_closed_with_integrity_evidence() -> None:
    """Readiness failure should produce a non-qualifying, verifiable packet."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    host, port = probe.getsockname()
    probe.close()

    packet = run_latency_slo(
        LatencySLOConfig(
            server_url=f"http://{host}:{port}",
            request_count=1,
            warmup_count=0,
            concurrency=1,
            timeout_ms=100.0,
            target_p95_ms=500.0,
            max_error_rate=0.0,
        )
    )

    qualification = packet["content"]["qualification"]
    assert qualification["passed"] is False
    assert "no_successful_measurements" in qualification["failure_reasons"]
    assert "error_rate_exceeded" in qualification["failure_reasons"]
    assert packet["content"]["readiness"]["passed"] is False
    assert verify_latency_slo_evidence(packet) == (True, "verified")


def test_installed_cli_returns_two_and_writes_evidence_on_target_miss(
    tmp_path: Path,
) -> None:
    """A failed release gate should remain inspectable and return status 2."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    host, port = probe.getsockname()
    probe.close()
    output = tmp_path / "not-qualified.json"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "latency-slo",
                "--server",
                f"http://{host}:{port}",
                "--requests",
                "1",
                "--warmup",
                "0",
                "--concurrency",
                "1",
                "--timeout-ms",
                "100",
                "--output",
                str(output),
            ]
        )

    assert exc_info.value.code == 2
    packet = json.loads(output.read_text(encoding="utf-8"))
    assert packet["content"]["qualification"]["passed"] is False
    assert verify_latency_slo_evidence(packet) == (True, "verified")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("server_url", "file:///tmp/server.sock"),
        ("server_url", "http://user:secret@127.0.0.1:8080"),
        ("server_url", "http://127.0.0.1:8080?token=secret"),
        ("request_count", 0),
        ("request_count", True),
        ("request_count", 1_000_001),
        ("warmup_count", -1),
        ("warmup_count", True),
        ("warmup_count", 100_001),
        ("concurrency", 0),
        ("concurrency", True),
        ("concurrency", 1_025),
        ("timeout_ms", 0.0),
        ("timeout_ms", float("nan")),
        ("target_p95_ms", 0.0),
        ("target_p95_ms", float("inf")),
        ("max_error_rate", -0.1),
        ("max_error_rate", 1.1),
        ("max_error_rate", float("nan")),
    ],
)
def test_qualification_configuration_rejects_ambiguous_operating_points(
    field: str,
    value: object,
) -> None:
    """A qualification claim must have a complete, valid operating point."""
    values: dict[str, object] = {"server_url": "http://127.0.0.1:8080"}
    values[field] = value
    with pytest.raises(ValueError):
        LatencySLOConfig(**values)
