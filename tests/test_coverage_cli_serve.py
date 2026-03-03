"""Coverage tests for cli.py — serve execution, ingest, config edges."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from director_ai.cli import main


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
