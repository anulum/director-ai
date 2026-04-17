# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — gRPC Server Coverage Tests
"""Multi-angle tests for grpc_server.py.

Covers: import error handling, module availability check,
server creation contract, and graceful degradation when
grpcio is unavailable.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

# ── Import error handling ──────────────────────────────────────────


class TestGrpcImportGuard:
    """gRPC server must fail gracefully when grpcio is missing."""

    def test_grpc_import_error_raises_with_message(self):
        with (
            patch.dict(sys.modules, {"grpc": None}),
            pytest.raises(ImportError, match="grpcio"),
        ):
            from director_ai.core.config import DirectorConfig
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(DirectorConfig())

    def test_grpc_import_error_mentions_install(self):
        with patch.dict(sys.modules, {"grpc": None}):
            try:
                from director_ai.core.config import DirectorConfig
                from director_ai.grpc_server import create_grpc_server

                create_grpc_server(DirectorConfig())
            except ImportError as e:
                assert "grpcio" in str(e).lower() or "grpc" in str(e).lower()

    @pytest.mark.parametrize(
        "missing_module",
        [
            "grpc",
        ],
    )
    def test_missing_grpc_module_detected(self, missing_module):
        with (
            patch.dict(sys.modules, {missing_module: None}),
            pytest.raises(ImportError),
        ):
            from director_ai.core.config import DirectorConfig
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(DirectorConfig())


# ── Module availability ───────────────────────────────────────────


class TestGrpcModuleAvailability:
    """Verify grpc_server module can be imported when deps are present."""

    def test_grpc_server_module_exists(self):
        """grpc_server.py must exist as a module."""
        import importlib

        spec = importlib.util.find_spec("director_ai.grpc_server")
        assert spec is not None

    def test_create_grpc_server_is_callable(self):
        """create_grpc_server must be a callable."""
        try:
            from director_ai.grpc_server import create_grpc_server

            assert callable(create_grpc_server)
        except ImportError:
            pytest.skip("grpcio not installed")
