"""Coverage tests for grpc_server.py — servicer methods via mocked grpc."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from director_ai.core.config import DirectorConfig


def _make_grpc_mock():
    """Build a minimal grpc mock sufficient for create_grpc_server."""
    grpc = MagicMock()
    grpc.__version__ = "1.78.0"
    grpc.StatusCode.INVALID_ARGUMENT = "INVALID_ARGUMENT"
    grpc.StatusCode.UNAUTHENTICATED = "UNAUTHENTICATED"
    grpc.server.return_value = MagicMock()
    grpc.ssl_server_credentials.return_value = MagicMock()
    grpc.unary_unary_rpc_method_handler.return_value = MagicMock()
    return grpc


class TestCreateGrpcServerNoProto:
    def test_create_server_without_proto(self):
        grpc_mock = _make_grpc_mock()
        with patch.dict(
            sys.modules,
            {
                "grpc": grpc_mock,
                "grpc._cython": MagicMock(),
                "grpc._cython.cygrpc": MagicMock(),
                "concurrent": MagicMock(),
                "concurrent.futures": MagicMock(),
                "grpc_reflection": None,
                "grpc_reflection.v1alpha": None,
                "grpc_reflection.v1alpha.reflection": None,
            },
        ):
            import importlib

            import director_ai.grpc_server as gs_mod

            importlib.reload(gs_mod)

            cfg = DirectorConfig(use_nli=False)
            server = gs_mod.create_grpc_server(cfg)
            assert server is not None


class TestServicerMethods:
    def test_create_server_returns_server(self):
        grpc_mock = _make_grpc_mock()
        with patch.dict(
            sys.modules,
            {
                "grpc": grpc_mock,
                "grpc._cython": MagicMock(),
                "grpc._cython.cygrpc": MagicMock(),
                "concurrent": MagicMock(),
                "concurrent.futures": MagicMock(),
                "grpc_reflection": None,
                "grpc_reflection.v1alpha": None,
                "grpc_reflection.v1alpha.reflection": None,
            },
        ):
            import importlib

            import director_ai.grpc_server as gs_mod

            importlib.reload(gs_mod)

            cfg = DirectorConfig(use_nli=False)
            server = gs_mod.create_grpc_server(cfg)
            assert server is not None
            grpc_mock.server.return_value.add_insecure_port.assert_called_once()
