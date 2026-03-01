from __future__ import annotations

import json

from director_ai.cli import main


class TestStressTestCommand:
    """Tests for 'director-ai stress-test'."""

    def test_default_run(self, capsys):
        main(["stress-test", "--streams", "5", "--tokens-per-stream", "10"])
        captured = capsys.readouterr()
        assert "Streams:" in captured.out
        assert "Tokens/s:" in captured.out
        assert "Latency p50:" in captured.out

    def test_json_output(self, capsys):
        main(["stress-test", "--streams", "5", "--tokens-per-stream", "10", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["streams"] == 5
        assert data["tokens_per_stream"] == 10
        assert "tokens_per_second" in data
        assert "latency_p50" in data
        assert "latency_p95" in data
        assert "latency_p99" in data

    def test_concurrency_flag(self, capsys):
        main(
            [
                "stress-test",
                "--streams",
                "4",
                "--tokens-per-stream",
                "5",
                "--concurrency",
                "2",
                "--json",
            ]
        )
        data = json.loads(capsys.readouterr().out)
        assert data["concurrency"] == 2

    def test_halt_rate_is_float(self, capsys):
        main(["stress-test", "--streams", "3", "--tokens-per-stream", "5", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data["halt_rate"], float)
        assert 0.0 <= data["halt_rate"] <= 1.0

    def test_total_seconds_non_negative(self, capsys):
        main(["stress-test", "--streams", "3", "--tokens-per-stream", "5", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["total_seconds"] >= 0

    def test_streams_per_second_positive(self, capsys):
        main(["stress-test", "--streams", "3", "--tokens-per-stream", "5", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["streams_per_second"] > 0

    def test_latency_ordering(self, capsys):
        main(["stress-test", "--streams", "10", "--tokens-per-stream", "20", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["latency_p50"] <= data["latency_p95"]
        assert data["latency_p95"] <= data["latency_p99"]

    def test_text_output_format(self, capsys):
        main(["stress-test", "--streams", "3", "--tokens-per-stream", "5"])
        out = capsys.readouterr().out
        assert "Halt rate:" in out
        assert "Total time:" in out
        assert "ms" in out
