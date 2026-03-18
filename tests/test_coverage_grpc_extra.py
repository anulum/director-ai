# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for grpc_server.py â€” servicer methods via mocked grpc."""

from __future__ import annotations

import contextlib
import sys
from unittest.mock import MagicMock, patch

import director_ai as _director_pkg
from director_ai.core.config import DirectorConfig

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


def _make_mods(grpc_mock):
    """Build sys.modules dict with grpc + proto mocks."""
    return {
        "grpc": grpc_mock,
        "grpc._cython": MagicMock(),
        "grpc._cython.cygrpc": MagicMock(),
        "concurrent": MagicMock(),
        "concurrent.futures": MagicMock(),
        "grpc_reflection": None,
        "grpc_reflection.v1alpha": None,
        "grpc_reflection.v1alpha.reflection": None,
        "director_ai.director_pb2": MagicMock(),
        "director_ai.director_pb2_grpc": MagicMock(),
    }


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
        mods = _make_mods(grpc_mock)
        with _grpc_context(mods):
            cfg = DirectorConfig(use_nli=False)
            from director_ai.grpc_server import create_grpc_server

            server = create_grpc_server(cfg)
            assert server is not None


class TestServicerMethods:
    def test_create_server_returns_server(self):
        grpc_mock = _make_grpc_mock()
        mods = _make_mods(grpc_mock)
        with _grpc_context(mods):
            cfg = DirectorConfig(use_nli=False)
            from director_ai.grpc_server import create_grpc_server

            server = create_grpc_server(cfg)
            assert server is not None
            grpc_mock.server.return_value.add_insecure_port.assert_called_once()
