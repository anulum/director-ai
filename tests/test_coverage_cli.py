"""Coverage tests for cli.py — all subcommands."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from director_ai.cli import main


class TestCliHelp:
    def test_no_args_prints_help(self, capsys):
        main([])
        assert "Commands:" in capsys.readouterr().out

    def test_help_flag(self, capsys):
        main(["--help"])
        assert "Commands:" in capsys.readouterr().out

    def test_unknown_command(self, capsys):
        with pytest.raises(SystemExit):
            main(["nonexistent"])


class TestCliVersion:
    def test_version(self, capsys):
        main(["version"])
        assert "director-ai" in capsys.readouterr().out


class TestCliReview:
    def test_review_needs_two_args(self):
        with pytest.raises(SystemExit):
            main(["review", "only_one"])

    def test_review_works(self, capsys):
        main(["review", "What color is the sky?", "The sky is blue."])
        out = capsys.readouterr().out
        assert "Approved" in out
        assert "Coherence" in out


class TestCliProcess:
    def test_process_needs_arg(self):
        with pytest.raises(SystemExit):
            main(["process"])

    def test_process_works(self, capsys):
        main(["process", "What color is the sky?"])
        out = capsys.readouterr().out
        assert "Output" in out


class TestCliBatch:
    def test_batch_needs_arg(self):
        with pytest.raises(SystemExit):
            main(["batch"])

    def test_batch_file_not_found(self):
        with pytest.raises(SystemExit):
            main(["batch", "/nonexistent/file.jsonl"])

    def test_batch_processes_jsonl(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "Is water wet?"}) + "\n")
            f.write(json.dumps({"prompt": "Is fire hot?"}) + "\n")
            path = f.name

        try:
            main(["batch", path])
            out = capsys.readouterr().out
            assert "Total:" in out
        finally:
            os.unlink(path)

    def test_batch_with_output(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "test"}) + "\n")
            inpath = f.name

        outpath = inpath + ".out"
        try:
            main(["batch", inpath, "--output", outpath])
            assert os.path.exists(outpath)
            out = capsys.readouterr().out
            assert "Results written" in out
        finally:
            os.unlink(inpath)
            if os.path.exists(outpath):
                os.unlink(outpath)

    def test_batch_skips_bad_json(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not json\n")
            f.write(json.dumps({"prompt": "valid"}) + "\n")
            path = f.name

        try:
            main(["batch", path])
            out = capsys.readouterr().out
            assert "Total:" in out
        finally:
            os.unlink(path)

    def test_batch_skips_invalid_prompt(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": ""}) + "\n")
            f.write(json.dumps({"prompt": "valid"}) + "\n")
            path = f.name

        try:
            main(["batch", path])
            out = capsys.readouterr().out
            assert "Total:" in out
        finally:
            os.unlink(path)

    def test_batch_file_too_large(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        from unittest.mock import patch

        with (
            patch("os.path.getsize", return_value=200_000_000),
            pytest.raises(SystemExit),
        ):
            main(["batch", path])

        os.unlink(path)


class TestCliQuickstart:
    def test_quickstart_creates_directory(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        main(["quickstart"])
        assert (tmp_path / "director_guard").exists()
        assert (tmp_path / "director_guard" / "guard.py").exists()
        assert (tmp_path / "director_guard" / "config.yaml").exists()

    def test_quickstart_with_profile(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        main(["quickstart", "--profile", "medical"])
        content = (tmp_path / "director_guard" / "config.yaml").read_text()
        assert "medical" in content

    def test_quickstart_invalid_profile(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            main(["quickstart", "--profile", "nonexistent"])

    def test_quickstart_dir_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "director_guard").mkdir()
        with pytest.raises(SystemExit):
            main(["quickstart"])


class TestCliIngest:
    def test_ingest_needs_arg(self):
        with pytest.raises(SystemExit):
            main(["ingest"])

    def test_ingest_path_not_found(self):
        with pytest.raises(SystemExit):
            main(["ingest", "/nonexistent/path"])

    def test_ingest_txt_file(self, capsys, tmp_path):
        txt = tmp_path / "facts.txt"
        txt.write_text("The sky is blue.\n\nWater is wet.\n")
        main(["ingest", str(txt)])
        out = capsys.readouterr().out
        assert "Ingested" in out

    def test_ingest_jsonl_file(self, capsys, tmp_path):
        jf = tmp_path / "facts.jsonl"
        jf.write_text(json.dumps({"text": "Earth orbits the Sun."}) + "\n")
        main(["ingest", str(jf)])
        out = capsys.readouterr().out
        assert "Ingested" in out

    def test_ingest_directory(self, capsys, tmp_path):
        (tmp_path / "a.txt").write_text("Fact A.\n\nFact B.\n")
        (tmp_path / "b.md").write_text("# Fact C\n\nFact D.\n")
        main(["ingest", str(tmp_path)])
        out = capsys.readouterr().out
        assert "Ingested" in out

    def test_ingest_empty_dir(self, capsys, tmp_path):
        with pytest.raises(SystemExit):
            main(["ingest", str(tmp_path)])

    def test_ingest_with_persist(self, capsys, tmp_path):
        txt = tmp_path / "facts.txt"
        txt.write_text("Fact.\n")
        persist = tmp_path / "db"
        main(["ingest", str(txt), "--persist", str(persist)])
        out = capsys.readouterr().out
        assert "Persisted" in out

    def test_ingest_with_chunk_size(self, capsys, tmp_path):
        txt = tmp_path / "facts.txt"
        txt.write_text("Sentence one.\n\nSentence two.\n\nSentence three.\n")
        main(["ingest", str(txt), "--chunk-size", "5"])
        out = capsys.readouterr().out
        assert "Ingested" in out


class TestCliTune:
    def test_tune_needs_arg(self):
        with pytest.raises(SystemExit):
            main(["tune"])

    def test_tune_file_not_found(self):
        with pytest.raises(SystemExit):
            main(["tune", "/nonexistent.jsonl"])

    def test_tune_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        with pytest.raises(SystemExit):
            main(["tune", str(f)])

    def test_tune_works(self, capsys, tmp_path):
        f = tmp_path / "labeled.jsonl"
        lines = []
        for i in range(20):
            lines.append(
                json.dumps(
                    {
                        "prompt": f"q{i}",
                        "response": f"a{i}",
                        "label": i % 2 == 0,
                    }
                )
            )
        f.write_text("\n".join(lines) + "\n")

        main(["tune", str(f)])
        out = capsys.readouterr().out
        assert "Best threshold" in out

    def test_tune_with_output(self, capsys, tmp_path):
        f = tmp_path / "labeled.jsonl"
        lines = [json.dumps({"prompt": "q", "response": "a", "label": True})]
        lines *= 10
        f.write_text("\n".join(lines) + "\n")

        out_yaml = tmp_path / "config.yaml"
        main(["tune", str(f), "--output", str(out_yaml)])
        assert out_yaml.exists()

    def test_tune_skips_incomplete(self, capsys, tmp_path):
        f = tmp_path / "partial.jsonl"
        f.write_text(
            json.dumps({"prompt": "q"})
            + "\n"
            + json.dumps({"prompt": "q", "response": "a", "label": True})
            + "\n"
        )
        main(["tune", str(f)])
        out = capsys.readouterr().out
        assert "Best threshold" in out


class TestCliConfig:
    def test_config_default(self, capsys):
        main(["config"])
        out = capsys.readouterr().out
        assert "coherence_threshold" in out

    def test_config_with_profile(self, capsys):
        main(["config", "--profile", "fast"])
        out = capsys.readouterr().out
        assert "coherence_threshold" in out


class TestCliDoctor:
    def test_doctor_runs(self, capsys):
        main(["doctor"])
        out = capsys.readouterr().out
        assert "checks passed" in out

    def test_doctor_output_includes_version(self, capsys):
        main(["doctor"])
        out = capsys.readouterr().out
        assert "director-ai" in out


class TestCliStressTest:
    def test_stress_test_default(self, capsys):
        main(["stress-test", "--streams", "5", "--tokens-per-stream", "10"])
        out = capsys.readouterr().out
        assert "Streams:" in out

    def test_stress_test_json(self, capsys):
        main(
            [
                "stress-test",
                "--streams",
                "5",
                "--tokens-per-stream",
                "10",
                "--concurrency",
                "2",
                "--json",
            ]
        )
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "streams_per_second" in data
