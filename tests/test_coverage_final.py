"""Final coverage tests — remaining reachable gaps across cli, server, backends."""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from director_ai.cli import main
from director_ai.core.config import DirectorConfig

_HAS_FASTAPI = __import__("importlib").util.find_spec("fastapi") is not None
_skip_no_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


class TestCliIngestProcessing:
    def test_ingest_json_file(self, capsys, tmp_path):
        data = [
            {"text": "The sky is blue."},
            {"content": "Water is wet."},
        ]
        jf = tmp_path / "facts.jsonl"
        jf.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")
        main(["ingest", str(jf)])
        out = capsys.readouterr().out
        assert "ingested" in out.lower() or "fact" in out.lower() or len(out) > 0

    def test_ingest_text_file(self, capsys, tmp_path):
        tf = tmp_path / "doc.txt"
        tf.write_text("The sky is blue.\n\nWater is wet.", encoding="utf-8")
        main(["ingest", str(tf)])
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_ingest_directory(self, capsys, tmp_path):
        (tmp_path / "a.txt").write_text("Fact A.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Fact B.", encoding="utf-8")
        main(["ingest", str(tmp_path)])
        out = capsys.readouterr().out
        assert len(out) > 0


class TestCliStressTestRuns:
    def test_stress_test_runs(self, capsys):
        main(["stress-test", "--streams", "5", "--tokens-per-stream", "3"])
        out = capsys.readouterr().out
        assert "Streams:" in out

    def test_stress_test_json(self, capsys):
        main(["stress-test", "--streams", "5", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "streams_per_second" in data


class TestCliEvalWithMocks:
    def test_eval_runs(self, capsys):
        mock_run_all = ModuleType("benchmarks.run_all")
        mock_run_all._run_suite = MagicMock(return_value={"accuracy": 0.9})
        mock_run_all._print_comparison_table = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "benchmarks": ModuleType("benchmarks"),
                "benchmarks.run_all": mock_run_all,
            },
        ):
            main(["eval"])
            mock_run_all._run_suite.assert_called_once()

    def test_eval_with_output(self, capsys, tmp_path):
        mock_run_all = ModuleType("benchmarks.run_all")
        mock_run_all._run_suite = MagicMock(return_value={"accuracy": 0.9})
        mock_run_all._print_comparison_table = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "benchmarks": ModuleType("benchmarks"),
                "benchmarks.run_all": mock_run_all,
            },
        ):
            out_file = str(tmp_path / "eval.json")
            main(["eval", "--output", out_file])
            assert os.path.exists(out_file)

    def test_eval_with_dataset(self, capsys):
        mock_run_all = ModuleType("benchmarks.run_all")
        mock_run_all._run_suite = MagicMock(return_value={"accuracy": 0.9})
        mock_run_all._print_comparison_table = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "benchmarks": ModuleType("benchmarks"),
                "benchmarks.run_all": mock_run_all,
            },
        ):
            main(["eval", "--dataset", "aggrefact", "--max-samples", "10"])


class TestCliServeMultiWorker:
    def test_serve_multi_worker(self):
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            main(["serve", "--workers", "2", "--port", "9998"])
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[1].get("workers", 1) >= 2 or (
                len(call_kwargs[0]) > 0  # factory path mode
            )


@_skip_no_server
class TestServerOtelBranch:
    def test_otel_enabled(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False, otel_enabled=True)
        app = __import__("director_ai.server", fromlist=["create_app"]).create_app(
            config=cfg,
        )
        with TestClient(app) as c:
            resp = c.get("/v1/health")
            assert resp.status_code == 200


@_skip_no_server
class TestServerReviewWithTenant:
    def test_review_with_tenant_header(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False, tenant_routing=True)
        app = __import__("director_ai.server", fromlist=["create_app"]).create_app(
            config=cfg,
        )
        with TestClient(app) as c:
            resp = c.post(
                "/v1/review",
                json={"prompt": "sky?", "response": "The sky is blue."},
                headers={"X-Tenant-ID": "tenant-1"},
            )
            assert resp.status_code == 200
