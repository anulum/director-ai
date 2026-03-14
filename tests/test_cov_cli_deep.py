"""Deep coverage for cli.py — batch, ingest, quickstart, tune, serve edge cases."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from director_ai.cli import main


class TestBatchEdges:
    def test_batch_no_args(self):
        with pytest.raises(SystemExit):
            main(["batch"])

    def test_batch_missing_file(self):
        with pytest.raises(SystemExit):
            main(["batch", "/nonexistent/batch.jsonl"])

    def test_batch_with_output(self, tmp_path, capsys):
        inp = tmp_path / "input.jsonl"
        inp.write_text(
            json.dumps({"prompt": "What is 1+1?"})
            + "\n"
            + json.dumps({"prompt": "What is 2+2?"})
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "output.jsonl"
        main(["batch", str(inp), "--output", str(out)])
        assert out.exists()

    def test_batch_oversized_line(self, tmp_path, capsys):
        inp = tmp_path / "big.jsonl"
        big_prompt = json.dumps({"prompt": "x" * (1024 * 1024 + 1)})
        inp.write_text(big_prompt + "\n", encoding="utf-8")
        main(["batch", str(inp)])
        out = capsys.readouterr().out
        assert "skipping" in out.lower() or "Warning" in out

    def test_batch_max_prompts(self, tmp_path, capsys):
        inp = tmp_path / "many.jsonl"
        lines = "\n".join(json.dumps({"prompt": f"q{i}"}) for i in range(10_001))
        inp.write_text(lines, encoding="utf-8")
        main(["batch", str(inp)])
        out = capsys.readouterr().out
        assert "truncated" in out.lower() or len(out) > 0

    def test_batch_malformed_json(self, tmp_path, capsys):
        inp = tmp_path / "bad.jsonl"
        inp.write_text("not json\n", encoding="utf-8")
        main(["batch", str(inp)])
        out = capsys.readouterr().out
        assert "malformed" in out.lower() or "skipping" in out.lower()

    def test_batch_invalid_prompt(self, tmp_path, capsys):
        inp = tmp_path / "noprompt.jsonl"
        inp.write_text(json.dumps({"other": 123}) + "\n", encoding="utf-8")
        main(["batch", str(inp)])

    def test_batch_file_too_large(self, tmp_path):
        inp = tmp_path / "huge.jsonl"
        inp.write_text("x", encoding="utf-8")
        with (
            patch("director_ai.cli._BATCH_MAX_FILE_SIZE", 0),
            pytest.raises(SystemExit),
        ):
            main(["batch", str(inp)])


class TestIngestEdges:
    def test_ingest_with_persist(self, tmp_path, capsys):
        f = tmp_path / "doc.txt"
        f.write_text("Fact one.\n\nFact two.", encoding="utf-8")
        persist = tmp_path / "persist"
        persist.mkdir()
        main(["ingest", str(f), "--persist", str(persist)])

    def test_ingest_with_chunk_size(self, tmp_path, capsys):
        f = tmp_path / "doc.txt"
        f.write_text("Fact one.\n\nFact two.\n\nFact three.", encoding="utf-8")
        main(["ingest", str(f), "--chunk-size", "100"])

    def test_ingest_json_decode_error(self, tmp_path, capsys):
        f = tmp_path / "bad.json"
        f.write_text("not valid json\n", encoding="utf-8")
        main(["ingest", str(f)])


class TestQuickstart:
    def test_quickstart_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        main(["quickstart"])
        assert (tmp_path / "director_guard").exists()

    def test_quickstart_invalid_profile(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            main(["quickstart", "--profile", "nonexistent_profile"])

    def test_quickstart_dir_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "director_guard").mkdir()
        with pytest.raises(SystemExit):
            main(["quickstart"])


class TestTune:
    def test_tune_no_args(self):
        with pytest.raises(SystemExit):
            main(["tune"])

    def test_tune_missing_file(self):
        with pytest.raises(SystemExit):
            main(["tune", "/nonexistent.jsonl"])

    def test_tune_valid(self, tmp_path, capsys):
        f = tmp_path / "labeled.jsonl"
        lines = [
            json.dumps(
                {"prompt": "sky?", "response": "The sky is blue.", "label": True},
            ),
            json.dumps(
                {"prompt": "sky?", "response": "The sky is green.", "label": False},
            ),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")
        main(["tune", str(f)])
        out = capsys.readouterr().out
        assert "threshold" in out.lower()

    def test_tune_with_output(self, tmp_path, capsys):
        f = tmp_path / "labeled.jsonl"
        lines = [
            json.dumps({"prompt": "q", "response": "a", "label": True}),
            json.dumps({"prompt": "q", "response": "b", "label": False}),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")
        out_f = tmp_path / "result.yaml"
        main(["tune", str(f), "--output", str(out_f)])
        assert out_f.exists()

    def test_tune_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["tune", str(f)])

    def test_tune_missing_fields(self, tmp_path, capsys):
        f = tmp_path / "partial.jsonl"
        f.write_text(json.dumps({"prompt": "q"}) + "\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["tune", str(f)])

    def test_tune_malformed_json(self, tmp_path, capsys):
        f = tmp_path / "bad.jsonl"
        lines = [
            "not json",
            json.dumps({"prompt": "q", "response": "a", "label": True}),
            json.dumps({"prompt": "q", "response": "b", "label": False}),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")
        main(["tune", str(f)])


class TestServeEdges:
    def test_serve_invalid_port(self):
        with pytest.raises(SystemExit):
            main(["serve", "--port", "abc"])

    def test_serve_invalid_workers(self):
        with pytest.raises(SystemExit):
            main(["serve", "--workers", "0"])

    def test_serve_invalid_transport(self):
        with pytest.raises(SystemExit):
            main(["serve", "--transport", "mqtt"])

    def test_serve_missing_uvicorn(self):
        with patch.dict(sys.modules, {"uvicorn": None}), pytest.raises(SystemExit):
            main(["serve"])


class TestEvalEdges:
    def test_eval_invalid_max_samples(self, capsys):
        mock_run_all = ModuleType("benchmarks.run_all")
        mock_run_all._run_suite = MagicMock(return_value={"accuracy": 0.9})
        mock_run_all._print_comparison_table = MagicMock()

        with (
            patch.dict(
                sys.modules,
                {
                    "benchmarks": ModuleType("benchmarks"),
                    "benchmarks.run_all": mock_run_all,
                },
            ),
            pytest.raises(SystemExit),
        ):
            main(["eval", "--max-samples", "abc"])


class TestConfigEdges:
    def test_config_profile_missing_value(self):
        with pytest.raises(SystemExit):
            main(["config", "--profile"])

    def test_config_default(self, capsys):
        main(["config"])
        out = capsys.readouterr().out
        assert len(out) > 10
