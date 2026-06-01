# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI Tests
"""Multi-angle tests for director-ai CLI entry point pipeline.

Covers: subcommands, arg parsing, profile loading, serve/review/process,
error handling, pipeline integration, and performance documentation.
"""

import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

import director_ai.cli as cli_module
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

    def test_invalid_command_token_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["../review"])
        captured = capsys.readouterr()
        assert exc_info.value.code == 1
        assert "Invalid command name" in captured.out


class TestVersionCommand:
    """Tests for 'director-ai version'."""

    def test_version(self, capsys):
        main(["version"])
        captured = capsys.readouterr()
        assert "director-ai" in captured.out
        first_line = captured.out.strip().splitlines()[0]
        parts = first_line.split()[-1].split(".")
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

    def _patch_batch_runtime(self, monkeypatch):
        seen: dict[str, tuple[str, ...]] = {}

        class FakeConfig:
            def build_store(self):
                return object()

            def build_scorer(self, **_kwargs):
                return object()

        class FakeAgent:
            def __init__(self, *_args, **_kwargs):
                pass

        class FakeBatchProcessor:
            def __init__(self, *_args, **_kwargs):
                pass

            def process_batch(self, prompts):
                seen["prompts"] = tuple(prompts)
                return types.SimpleNamespace(
                    total=len(prompts),
                    succeeded=len(prompts),
                    failed=0,
                    duration_seconds=0.01,
                    results=[(True, object()) for _prompt in prompts],
                )

        monkeypatch.setattr(
            "director_ai.core.config.DirectorConfig.from_env",
            lambda: FakeConfig(),
        )
        monkeypatch.setattr("director_ai.core.agent.CoherenceAgent", FakeAgent)
        monkeypatch.setattr(
            "director_ai.core.runtime.batch.BatchProcessor",
            FakeBatchProcessor,
        )
        return seen

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
            mode="w",
            suffix=".jsonl",
            delete=False,
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

    def test_batch_rejects_nonexistent_input_file(self, tmp_path, capsys):
        missing_path = tmp_path / "missing.jsonl"

        with pytest.raises(SystemExit) as exc_info:
            main(["batch", str(missing_path)])

        captured = capsys.readouterr()
        assert exc_info.value.code == 1
        assert "file not found" in captured.out

    def test_batch_ignores_blank_lines(self, tmp_path, monkeypatch, capsys):
        input_path = tmp_path / "prompts.jsonl"
        input_path.write_text("\n" + json.dumps({"prompt": "usable"}) + "\n")
        seen = self._patch_batch_runtime(monkeypatch)

        main(["batch", str(input_path)])

        captured = capsys.readouterr()
        assert "Total:" in captured.out
        assert seen["prompts"] == ("usable",)

    def test_batch_skips_malformed_json_and_keeps_valid_prompt(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        input_path = tmp_path / "prompts.jsonl"
        input_path.write_text("{not valid json}\n" + json.dumps({"prompt": "Q1"}) + "\n")
        seen = self._patch_batch_runtime(monkeypatch)

        main(["batch", str(input_path)])

        captured = capsys.readouterr()
        assert "skipping malformed JSON" in captured.out
        assert seen["prompts"] == ("Q1",)

    def test_batch_rejects_oversized_input_file(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        input_path = tmp_path / "prompts.jsonl"
        input_path.write_text(json.dumps({"prompt": "Q1"}) + "\n")
        monkeypatch.setattr(
            "os.path.getsize",
            lambda _path: cli_module._BATCH_MAX_FILE_SIZE + 1,
        )

        with pytest.raises(SystemExit) as exc_info:
            main(["batch", str(input_path)])

        captured = capsys.readouterr()
        assert exc_info.value.code == 1
        assert "file too large" in captured.out

    def test_batch_skips_oversized_line_and_processes_remaining_prompt(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        input_path = tmp_path / "prompts.jsonl"
        input_path.write_text(
            json.dumps({"prompt": "x" * 40}) + "\n"
            + json.dumps({"prompt": "short"}) + "\n",
        )
        monkeypatch.setattr(cli_module, "_BATCH_MAX_LINE_SIZE", 30)
        seen = self._patch_batch_runtime(monkeypatch)

        main(["batch", str(input_path)])

        captured = capsys.readouterr()
        assert "skipping line 1" in captured.out
        assert "Total:" in captured.out
        assert "Success:" in captured.out
        assert seen["prompts"] == ("short",)

    def test_batch_enforces_prompt_cap(self, tmp_path, monkeypatch, capsys):
        input_path = tmp_path / "prompts.jsonl"
        input_path.write_text(
            json.dumps({"prompt": "Q1"}) + "\n"
            + json.dumps({"prompt": "Q2"}) + "\n",
        )
        monkeypatch.setattr(cli_module, "_BATCH_MAX_PROMPTS", 1)
        seen = self._patch_batch_runtime(monkeypatch)

        main(["batch", str(input_path)])

        captured = capsys.readouterr()
        assert "truncated at 1 prompts" in captured.out
        assert "Total:" in captured.out
        assert "Success:" in captured.out
        assert seen["prompts"] == ("Q1",)

    def test_batch_skips_invalid_prompt_and_keeps_valid_prompt(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        input_path = tmp_path / "prompts.jsonl"
        input_path.write_text(
            json.dumps({"prompt": ""}) + "\n"
            + json.dumps({"prompt": "usable"}) + "\n",
        )
        seen = self._patch_batch_runtime(monkeypatch)

        main(["batch", str(input_path)])

        captured = capsys.readouterr()
        assert "skipping invalid prompt" in captured.out
        assert "Total:" in captured.out
        assert "Success:" in captured.out
        assert seen["prompts"] == ("usable",)

    def test_batch_output_skips_non_review_results(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        input_path = tmp_path / "prompts.jsonl"
        output_path = tmp_path / "results.jsonl"
        input_path.write_text(json.dumps({"prompt": "Q1"}) + "\n")
        seen = self._patch_batch_runtime(monkeypatch)

        main(["batch", str(input_path), "--output", str(output_path)])

        captured = capsys.readouterr()
        assert "Results written to" in captured.out
        assert seen["prompts"] == ("Q1",)
        assert output_path.read_text() == ""


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
        assert (d / ".env").is_file()
        assert (d / "docker-compose.yml").is_file()
        assert (d / "chroma").is_dir()
        assert (d / "models" / "factcg-onnx" / "README.md").is_file()
        onnx_readme = (d / "models" / "factcg-onnx" / "README.md").read_text()
        assert "director-ai export --format onnx" in onnx_readme
        assert "model.onnx" in onnx_readme

    def test_quickstart_compose_has_default_and_onnx_paths(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        main(["quickstart", "--profile", "lite"])
        d = tmp_path / "director_guard"
        compose_text = (d / "docker-compose.yml").read_text()
        env_text = (d / ".env").read_text()
        assert "director-proxy:" in compose_text
        assert "director-api:" in compose_text
        assert "director-proxy-onnx:" in compose_text
        assert "director-ai[server,vector]" in compose_text
        assert "director-ai[server,vector,nli,onnx]" in compose_text
        assert (
            "director-ai ingest /app/facts.txt --persist /data/chroma" in compose_text
        )
        assert "--config-env" in compose_text
        assert "DIRECTOR_VECTOR_BACKEND=chroma" in env_text
        assert "DIRECTOR_CHROMA_PERSIST_DIR=/data/chroma" in env_text
        assert "DIRECTOR_ONNX_PATH" not in env_text

    def test_quickstart_with_profile(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        main(["quickstart", "--profile", "medical"])
        cfg_text = (tmp_path / "director_guard" / "config.yaml").read_text()
        assert "threshold: 0.3" in cfg_text
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

    def test_quickstart_no_compose(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        main(["quickstart", "--no-compose"])
        d = tmp_path / "director_guard"
        assert not (d / "docker-compose.yml").exists()
        assert not (d / ".env").exists()
        assert "Docker Compose" not in (d / "README.md").read_text()

    def test_quickstart_run_invokes_docker_compose(self, tmp_path, monkeypatch):
        calls: list[tuple[list[str], object, bool]] = []

        def fake_run(command, cwd=None, check=False):
            calls.append((command, cwd, check))
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("director_ai.cli.shutil.which", lambda name: name)
        monkeypatch.setattr("director_ai.cli.subprocess.run", fake_run)
        main(["quickstart", "--run"])
        assert calls == [(["docker", "compose", "up"], Path("director_guard"), True)]

    def test_quickstart_run_requires_docker_compose(self, tmp_path, monkeypatch):
        def fake_run(command, cwd=None, check=False):
            raise FileNotFoundError

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("director_ai.cli.shutil.which", lambda _name: None)
        monkeypatch.setattr("director_ai.cli.subprocess.run", fake_run)
        with pytest.raises(SystemExit) as exc_info:
            main(["quickstart", "--run"])
        assert exc_info.value.code == 1

    def test_quickstart_run_reports_missing_compose_binary(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        def fake_run(command, cwd=None, check=False):
            raise FileNotFoundError

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("director_ai.cli.shutil.which", lambda _name: "docker")
        monkeypatch.setattr("director_ai.cli.subprocess.run", fake_run)

        with pytest.raises(SystemExit) as exc_info:
            main(["quickstart", "--run"])

        captured = capsys.readouterr()
        assert exc_info.value.code == 1
        assert "docker compose is required" in captured.out

    def test_quickstart_run_reports_compose_process_failure(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        def fake_run(command, cwd=None, check=False):
            raise subprocess.CalledProcessError(17, command)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("director_ai.cli.shutil.which", lambda _name: "docker")
        monkeypatch.setattr("director_ai.cli.subprocess.run", fake_run)

        with pytest.raises(SystemExit) as exc_info:
            main(["quickstart", "--run"])

        captured = capsys.readouterr()
        assert exc_info.value.code == 17
        assert "docker compose failed with exit code 17" in captured.out

    def test_quickstart_ignores_unknown_options(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        main(["quickstart", "--unknown-option", "--no-compose"])

        d = tmp_path / "director_guard"
        assert d.is_dir()
        assert not (d / "docker-compose.yml").exists()


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

    def test_config_profile_requires_value(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["config", "--profile"])
        assert exc_info.value.code == 1


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


class TestCLIConfigFromEnv:
    """Tests that CLI commands respect env var overrides."""

    def test_review_uses_env_threshold(self, capsys, monkeypatch):
        monkeypatch.setenv("DIRECTOR_COHERENCE_THRESHOLD", "0.99")
        monkeypatch.setenv("DIRECTOR_SOFT_LIMIT", "0.99")
        main(["review", "What is 2+2?", "4"])
        captured = capsys.readouterr()
        assert "Approved:" in captured.out

    def test_process_uses_env_config(self, capsys, monkeypatch):
        monkeypatch.setenv("DIRECTOR_COHERENCE_THRESHOLD", "0.01")
        main(["process", "What color is the sky?"])
        captured = capsys.readouterr()
        assert "Output:" in captured.out


class TestBenchCommand:
    """Tests for 'director-ai bench'."""

    def test_bench_subcommand_runs(self, capsys):
        pytest.importorskip("benchmarks", reason="benchmarks not on sys.path in CI")
        main(["bench", "--dataset", "regression"])
        captured = capsys.readouterr()
        assert "passed" in captured.out

    def test_bench_help_shown(self, capsys):
        main(["--help"])
        captured = capsys.readouterr()
        assert "bench" in captured.out

    def test_bench_invalid_dataset(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["bench", "--dataset", "nonexistent"])
        assert exc_info.value.code == 1
