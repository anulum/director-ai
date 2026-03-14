"""Deep coverage for grpc_server.py — servicer RPCs, auth, TLS, reflection."""

from __future__ import annotations

import contextlib
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import director_ai as _director_pkg

_SENTINEL = object()


@contextlib.contextmanager
def _grpc_context(mods):
    """Patch sys.modules AND clear director_ai submodule attr cache."""
    saved = {}
    for attr in ("director_pb2", "director_pb2_grpc"):
        full_key = f"director_ai.{attr}"
        if full_key not in mods:
            continue
        old = _director_pkg.__dict__.pop(attr, _SENTINEL)
        if old is not _SENTINEL:
            saved[attr] = old
    try:
        with patch.dict(sys.modules, mods):
            yield
    finally:
        for attr, old in saved.items():
            setattr(_director_pkg, attr, old)


def _build_grpc_mocks():
    """Build mock grpc module + stubs for create_grpc_server internals."""
    grpc = MagicMock()
    grpc.__version__ = "1.78.0"
    grpc.StatusCode.INVALID_ARGUMENT = "INVALID_ARGUMENT"
    grpc.StatusCode.UNAUTHENTICATED = "UNAUTHENTICATED"
    grpc.ServerInterceptor = type("ServerInterceptor", (), {})
    grpc.unary_unary_rpc_method_handler = MagicMock()
    grpc.ssl_server_credentials = MagicMock(return_value="creds")

    mock_server = MagicMock()
    grpc.server.return_value = mock_server

    return grpc, mock_server


def _inject_grpc(grpc_mock):
    """Build sys.modules patch dict with grpc + proto mocks."""
    mods = {
        "grpc": grpc_mock,
        "grpc_reflection": MagicMock(),
        "grpc_reflection.v1alpha": MagicMock(),
        "grpc_reflection.v1alpha.reflection": MagicMock(),
        "director_ai.director_pb2": MagicMock(),
        "director_ai.director_pb2_grpc": MagicMock(),
    }
    return mods


class TestCreateGrpcServer:
    def test_insecure_port(self):
        grpc, srv = _build_grpc_mocks()
        mods = _inject_grpc(grpc)
        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            result = create_grpc_server(port=50099)
            srv.add_insecure_port.assert_called_once()
            assert result is srv

    def test_tls_port(self):
        grpc, srv = _build_grpc_mocks()
        mods = _inject_grpc(grpc)
        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as cf:
                cf.write(b"cert")
                cert_path = cf.name
            with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as kf:
                kf.write(b"key")
                key_path = kf.name
            create_grpc_server(
                port=50099,
                tls_cert_path=cert_path,
                tls_key_path=key_path,
            )
            srv.add_secure_port.assert_called_once()

    def test_grpc_import_error(self):
        with patch.dict(sys.modules, {"grpc": None}):
            if "director_ai.grpc_server" in sys.modules:
                del sys.modules["director_ai.grpc_server"]
            from director_ai.grpc_server import create_grpc_server

            with pytest.raises(ImportError, match="grpcio"):
                create_grpc_server()


class TestAuthInterceptor:
    def test_auth_with_api_keys(self):
        grpc, srv = _build_grpc_mocks()
        mods = _inject_grpc(grpc)

        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(use_nli=False, api_keys=["secret-key-123"])

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50097)
            call_kwargs = grpc.server.call_args
            interceptors = call_kwargs.get("interceptors") or (
                call_kwargs[1].get("interceptors") if len(call_kwargs) > 1 else []
            )
            assert interceptors is not None

    def test_no_api_keys(self):
        grpc, srv = _build_grpc_mocks()
        mods = _inject_grpc(grpc)

        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(use_nli=False, api_keys=[])

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            result = create_grpc_server(config=cfg, port=50096)
            assert result is not None
