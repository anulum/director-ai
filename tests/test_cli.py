# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — CLI Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import json
import sys
import tempfile
import types

import pytest

from director_ai.cli import main


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_help_flag(self, capsys):
        main(["--help"])
        captured = capsys.readouterr()
        assert "Director-Class AI CLI" in captured.out
        assert "Commands:" in captured.out

    def test_no_args_shows_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "Director-Class AI CLI" in captured.out

    def test_unknown_command(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["foobar"])
        assert exc_info.value.code == 1


class TestVersionCommand:
    """Tests for 'director-ai version'."""

    def test_version(self, capsys):
        main(["version"])
        captured = capsys.readouterr()
        assert "director-ai" in captured.out
        # Version should be a semver string
        parts = captured.out.strip().split()[-1].split(".")
        assert len(parts) == 3


class TestReviewCommand:
    """Tests for 'director-ai review'."""

    def test_review_success(self, capsys):
        main(["review", "What is 2+2?", "4"])
        captured = capsys.readouterr()
        assert "Approved:" in captured.out
        assert "Coherence:" in captured.out

    def test_review_missing_args(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["review", "only-prompt"])
        assert exc_info.value.code == 1


class TestProcessCommand:
    """Tests for 'director-ai process'."""

    def test_process_success(self, capsys):
        main(["process", "What color is the sky?"])
        captured = capsys.readouterr()
        assert "Output:" in captured.out
        assert "Halted:" in captured.out

    def test_process_missing_prompt(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["process"])
        assert exc_info.value.code == 1


class TestBatchCommand:
    """Tests for 'director-ai batch'."""

    def test_batch_success(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "Q1"}) + "\n")
            f.write(json.dumps({"prompt": "Q2"}) + "\n")
            path = f.name

        main(["batch", path])
        captured = capsys.readouterr()
        assert "Total:" in captured.out
        assert "Success:" in captured.out

    def test_batch_with_output(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as inp:
            inp.write(json.dumps({"prompt": "Q1"}) + "\n")
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as out:
            output_path = out.name

        main(["batch", input_path, "--output", output_path])
        captured = capsys.readouterr()
        assert "Results written to" in captured.out

        with open(output_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "output" in data
        assert "halted" in data

    def test_batch_missing_file(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["batch"])
        assert exc_info.value.code == 1


class TestQuickstartCommand:
    """Tests for 'director-ai quickstart'."""

    def test_quickstart_creates_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        main(["quickstart"])
        d = tmp_path / "director_guard"
        assert d.is_dir()
        assert (d / "config.yaml").is_file()
        assert (d / "facts.txt").is_file()
        assert (d / "guard.py").is_file()
        assert (d / "README.md").is_file()

    def test_quickstart_with_profile(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        main(["quickstart", "--profile", "medical"])
        cfg_text = (tmp_path / "director_guard" / "config.yaml").read_text()
        assert "threshold: 0.75" in cfg_text
        assert "profile: medical" in cfg_text

    def test_quickstart_existing_dir_skips(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "director_guard").mkdir()
        with pytest.raises(SystemExit) as exc_info:
            main(["quickstart"])
        assert exc_info.value.code == 1

    def test_quickstart_invalid_profile(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            main(["quickstart", "--profile", "nonexistent"])
        assert exc_info.value.code == 1


class TestConfigCommand:
    """Tests for 'director-ai config'."""

    def test_config_default(self, capsys):
        main(["config"])
        captured = capsys.readouterr()
        assert "coherence_threshold" in captured.out

    def test_config_profile(self, capsys):
        main(["config", "--profile", "fast"])
        captured = capsys.readouterr()
        assert "coherence_threshold" in captured.out


class TestServeWorkers:
    """Tests for --workers flag on serve command."""

    def _mock_uvicorn(self, monkeypatch):
        calls: list[tuple] = []
        mock_uv = types.ModuleType("uvicorn")
        mock_uv.run = lambda *a, **kw: calls.append((a, kw))  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "uvicorn", mock_uv)
        return calls

    def test_multi_worker_uses_factory(self, monkeypatch):
        calls = self._mock_uvicorn(monkeypatch)
        main(["serve", "--workers", "4", "--port", "9877"])
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] == "director_ai.server:create_app"
        assert kwargs["factory"] is True
        assert kwargs["workers"] == 4
        assert kwargs["port"] == 9877

    def test_single_worker_no_factory(self, monkeypatch):
        calls = self._mock_uvicorn(monkeypatch)
        mock_app = object()
        mock_server = types.ModuleType("director_ai.server")
        mock_server.create_app = lambda config: mock_app  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "director_ai.server", mock_server)
        main(["serve", "--port", "9876"])
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] is mock_app
        assert "factory" not in kwargs
        assert "workers" not in kwargs

    def test_workers_invalid_value(self, monkeypatch):
        self._mock_uvicorn(monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            main(["serve", "--workers", "abc"])
        assert exc_info.value.code == 1

    def test_workers_zero_rejected(self, monkeypatch):
        self._mock_uvicorn(monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            main(["serve", "--workers", "0"])
        assert exc_info.value.code == 1

    def test_help_shows_stress_test(self, capsys):
        main(["--help"])
        captured = capsys.readouterr()
        assert "stress-test" in captured.out
