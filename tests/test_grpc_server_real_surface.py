# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - gRPC server real-surface tests
"""Real gRPC channel coverage for the public DirectorService server."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

grpc = pytest.importorskip("grpc", reason="grpcio required for gRPC server tests")
director_pb2 = pytest.importorskip(
    "director_ai.director_pb2",
    reason="generated DirectorService protobuf messages required",
)
director_pb2_grpc = pytest.importorskip(
    "director_ai.director_pb2_grpc",
    reason="generated DirectorService gRPC stubs required",
)

from director_ai.core.config import DirectorConfig  # noqa: E402
from director_ai.grpc_server import create_grpc_server  # noqa: E402


def _free_local_port() -> int:
    """Reserve and release an ephemeral loopback port for a local gRPC server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_self_signed_localhost_cert(tmp_path: Path) -> tuple[Path, Path]:
    """Write a short-lived localhost certificate and private key."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65_537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(minutes=10))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path = tmp_path / "localhost.crt"
    key_path = tmp_path / "localhost.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@contextmanager
def _running_server(
    config: DirectorConfig,
    *,
    tls_cert_path: str | None = None,
    tls_key_path: str | None = None,
    host: str = "127.0.0.1",
) -> Iterator[str]:
    """Start the production gRPC server and yield its loopback address."""
    port = _free_local_port()
    server = create_grpc_server(
        config=config,
        max_workers=2,
        port=port,
        tls_cert_path=tls_cert_path,
        tls_key_path=tls_key_path,
    )
    server.start()
    try:
        yield f"{host}:{port}"
    finally:
        server.stop(0).wait(timeout=5)


def _review_request(text: str) -> object:
    """Return a prompt/response protobuf request with matching text."""
    return director_pb2.ReviewRequest(prompt=text, response=text)


def test_review_round_trips_over_real_grpc_channel() -> None:
    """Review should return a real protobuf response over a gRPC channel."""
    config = DirectorConfig(use_nli=False, rate_limit_rpm=0)

    with _running_server(config) as address, grpc.insecure_channel(address) as channel:
        stub = director_pb2_grpc.DirectorServiceStub(channel)
        response = stub.Review(
            director_pb2.ReviewRequest(
                prompt="The sky is blue.",
                response="The sky is blue.",
                session_id="real-grpc-review",
            ),
            timeout=5,
            metadata=(("x-tenant-id", "tenant.alpha"),),
        )

    assert isinstance(response.approved, bool)
    assert 0.0 <= response.coherence <= 1.0
    assert 0.0 <= response.h_logical <= 1.0
    assert 0.0 <= response.h_factual <= 1.0


def test_review_batch_and_batch_limit_use_real_rpc_status() -> None:
    """ReviewBatch should round-trip valid batches and reject oversized ones."""
    config = DirectorConfig(use_nli=False, rate_limit_rpm=0)

    with _running_server(config) as address, grpc.insecure_channel(address) as channel:
        stub = director_pb2_grpc.DirectorServiceStub(channel)
        valid_response = stub.ReviewBatch(
            director_pb2.BatchReviewRequest(
                requests=[
                    _review_request(text)
                    for text in ("Paris is in France.", "Water is wet.")
                ]
            ),
            timeout=5,
            metadata=(("x-tenant-id", "tenant.alpha"),),
        )
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.ReviewBatch(
                director_pb2.BatchReviewRequest(
                    requests=[
                        _review_request(f"Prompt {index}.") for index in range(1001)
                    ]
                ),
                timeout=5,
                metadata=(("x-tenant-id", "tenant.alpha"),),
            )

    assert len(valid_response.responses) == 2
    assert all(
        0.0 <= response.coherence <= 1.0 for response in valid_response.responses
    )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "batch too large" in excinfo.value.details()


def test_api_key_authentication_is_enforced_over_real_grpc_channel() -> None:
    """Configured API keys should reject unauthenticated RPCs and allow valid ones."""
    config = DirectorConfig(
        use_nli=False,
        api_keys=["correct-key"],
        api_key_tenant_map='{"correct-key": "tenant.auth"}',
        rate_limit_rpm=0,
    )

    with _running_server(config) as address, grpc.insecure_channel(address) as channel:
        stub = director_pb2_grpc.DirectorServiceStub(channel)
        request = _review_request("Saturn has rings.")
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.Review(request, timeout=5)

        authenticated = stub.Review(
            request,
            timeout=5,
            metadata=(("x-api-key", "correct-key"),),
        )

    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert "invalid API key" in excinfo.value.details()
    assert 0.0 <= authenticated.coherence <= 1.0


def test_tls_review_round_trips_over_real_secure_grpc_channel(tmp_path: Path) -> None:
    """TLS configuration should serve DirectorService over a secure channel."""
    cert_path, key_path = _write_self_signed_localhost_cert(tmp_path)
    config = DirectorConfig(use_nli=False, rate_limit_rpm=0)
    credentials = grpc.ssl_channel_credentials(root_certificates=cert_path.read_bytes())
    with (
        _running_server(
            config,
            tls_cert_path=str(cert_path),
            tls_key_path=str(key_path),
            host="localhost",
        ) as address,
        grpc.secure_channel(address, credentials) as channel,
    ):
        stub = director_pb2_grpc.DirectorServiceStub(channel)
        response = stub.Review(_review_request("Secure gRPC is enabled."), timeout=5)

    assert 0.0 <= response.coherence <= 1.0


def test_process_round_trips_generated_response_over_real_grpc_channel() -> None:
    """Process should generate and score a response through the real gRPC server."""
    config = DirectorConfig(use_nli=False, max_candidates=1, rate_limit_rpm=0)

    with _running_server(config) as address, grpc.insecure_channel(address) as channel:
        stub = director_pb2_grpc.DirectorServiceStub(channel)
        response = stub.Process(
            director_pb2.ProcessRequest(prompt="What colour is the sky?"),
            timeout=5,
            metadata=(("x-tenant-id", "tenant.process"),),
        )

    assert response.output
    assert 0.0 <= response.coherence <= 1.0
    assert response.candidates_evaluated == 1
    assert isinstance(response.halted, bool)
    assert isinstance(response.warning, bool)
    assert response.fallback_used is False


def test_stream_tokens_round_trips_real_streaming_rpc() -> None:
    """StreamTokens should return ordered token events from a real streaming RPC."""
    config = DirectorConfig(use_nli=False, hard_limit=0.0, rate_limit_rpm=0)

    with _running_server(config) as address, grpc.insecure_channel(address) as channel:
        stub = director_pb2_grpc.DirectorServiceStub(channel)
        events = list(
            stub.StreamTokens(
                director_pb2.StreamRequest(prompt="What colour is the sky?"),
                timeout=5,
                metadata=(("x-tenant-id", "tenant.stream"),),
            )
        )

    assert [event.index for event in events] == list(range(len(events)))
    assert [event.token for event in events][:3] == ["Based", "on", "my"]
    assert all(0.0 <= event.coherence <= 1.0 for event in events)
    assert all(event.halted is False for event in events)
    assert all(event.halt_reason == "" for event in events)


def test_stream_tokens_authentication_abort_uses_stream_rpc_status() -> None:
    """Unauthenticated streaming calls should fail with UNAUTHENTICATED."""
    config = DirectorConfig(use_nli=False, api_keys=["stream-key"], rate_limit_rpm=0)

    with _running_server(config) as address, grpc.insecure_channel(address) as channel:
        stub = director_pb2_grpc.DirectorServiceStub(channel)
        with pytest.raises(grpc.RpcError) as excinfo:
            list(
                stub.StreamTokens(
                    director_pb2.StreamRequest(prompt="What colour is the sky?"),
                    timeout=5,
                )
            )

    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert "invalid API key" in excinfo.value.details()


def test_tenant_rate_limit_is_enforced_over_real_grpc_channel() -> None:
    """Tenant rate limits should produce RESOURCE_EXHAUSTED from real RPC calls."""
    config = DirectorConfig(use_nli=False, rate_limit_rpm=1)

    with _running_server(config) as address, grpc.insecure_channel(address) as channel:
        stub = director_pb2_grpc.DirectorServiceStub(channel)
        request = _review_request("The moon orbits Earth.")
        first_response = stub.Review(
            request,
            timeout=5,
            metadata=(("x-tenant-id", "tenant.rate"),),
        )
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.Review(
                request,
                timeout=5,
                metadata=(("x-tenant-id", "tenant.rate"),),
            )

    assert 0.0 <= first_response.coherence <= 1.0
    assert excinfo.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert "rate limit exceeded" in excinfo.value.details()
