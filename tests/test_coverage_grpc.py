"""Coverage tests for grpc_server.py."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


class TestCreateGrpcServer:
    def test_grpc_import_error(self):
        with (
            patch.dict(sys.modules, {"grpc": None}),
            pytest.raises(ImportError, match="grpcio"),
        ):
            from director_ai.core.config import DirectorConfig
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(DirectorConfig())
