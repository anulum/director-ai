# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for cli.py â€” serve execution, ingest, config edges."""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock, patch

import pytest

from director_ai.cli import main

_HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
_skip_no_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


@_skip_no_server
class TestCliServeExec:
    def test_serve_http_runs_uvicorn(self):
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            main(["serve", "--transport", "http", "--port", "9999", "--workers", "1"])
            mock_uvicorn.run.assert_called_once()

    def test_serve_default(self):
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            main(["serve"])
            mock_uvicorn.run.assert_called_once()

    def test_serve_with_profile(self):
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            main(["serve", "--profile", "fast"])
            mock_uvicorn.run.assert_called_once()

    def test_serve_grpc_transport(self):
        with patch("director_ai.grpc_server.create_grpc_server") as mock_grpc:
            mock_server = MagicMock()
            mock_grpc.return_value = mock_server
            main(["serve", "--transport", "grpc", "--port", "9876"])


class TestCliConfig:
    def test_config_list(self, capsys):
        main(["config", "--list"])
        out = capsys.readouterr().out
        assert len(out) > 10

    def test_config_show_profile(self, capsys):
        main(["config", "--profile", "fast"])
        out = capsys.readouterr().out
        assert len(out) > 0


class TestCliIngestEdges:
    def test_ingest_no_file(self):
        with pytest.raises(SystemExit):
            main(["ingest"])

    def test_ingest_missing_file(self):
        with pytest.raises(SystemExit):
            main(["ingest", "/nonexistent/file.json"])
