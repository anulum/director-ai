"""Coverage tests for grpc_server.py."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


class TestNsHelper:
    def test_ns(self):
        from director_ai.grpc_server import _ns

        obj = _ns(a=1, b="two")
        assert obj.a == 1
        assert obj.b == "two"


class TestCreateGrpcServer:
    def test_grpc_import_error(self):
        with (
            patch.dict(sys.modules, {"grpc": None}),
            pytest.raises(ImportError, match="grpcio"),
        ):
            from director_ai.core.config import DirectorConfig
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(DirectorConfig())
