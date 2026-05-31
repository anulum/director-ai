# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Voice Demo Tests
"""Tests for the credential-free voice pipeline demo."""

from __future__ import annotations

import os
import subprocess
import sys

from director_ai.voice import DryRunTTSAdapter, run_voice_demo
from director_ai.voice.demo import (
    DEFAULT_FACTS,
    DEFAULT_PROMPT,
    VoiceDemoResult,
    _amain,
    main,
    scripted_tokens,
)


class TestVoiceDemo:
    async def test_run_voice_demo_uses_real_pipeline_with_dry_tts(self):
        result = await run_voice_demo(
            tokens=scripted_tokens("The Team plan costs CHF 19 per seat."),
            facts=DEFAULT_FACTS,
            prompt=DEFAULT_PROMPT,
            score_every=32,
            use_nli=False,
        )

        assert result.audio_chunks
        assert result.tts_texts == ["The Team plan costs CHF 19 per seat."]
        assert result.total_audio_bytes == sum(
            len(chunk) for chunk in result.audio_chunks
        )
        assert not result.halted
        assert result.halt_reason == ""

    async def test_run_voice_demo_reports_guarded_halt(self):
        result = await run_voice_demo(
            tokens=scripted_tokens("Unsupported WhatsApp approvals ship today."),
            facts={"support": "Slack notifications are supported."},
            prompt="Which integrations are supported?",
            score_every=1,
            hard_limit=0.99,
            use_nli=False,
            recovery="Let me verify that before speaking.",
        )

        assert result.halted
        assert result.halt_reason in {"hard_limit", "window_avg"}
        assert result.recovery_text == "Let me verify that before speaking."
        assert result.tts_texts[-1] == result.recovery_text

    async def test_dry_run_tts_adapter_closes_and_records_text(self):
        adapter = DryRunTTSAdapter(prefix=b"voice:")

        chunks = [chunk async for chunk in adapter.synthesise("Hello.")]
        await adapter.close()

        assert chunks == [b"voice:Hello."]
        assert adapter.texts == ["Hello."]
        assert adapter.closed

    async def test_run_voice_demo_accepts_sync_iterable_tokens(self):
        result = await run_voice_demo(
            tokens=["The Team plan costs CHF 19 per seat."],
            facts=DEFAULT_FACTS,
            prompt=DEFAULT_PROMPT,
            score_every=16,
            use_nli=False,
        )

        assert result.tts_texts == ["The Team plan costs CHF 19 per seat."]
        assert result.audio_chunks == [b"audio:The Team plan costs CHF 19 per seat."]
        assert not result.halted

    async def test_async_cli_prints_halt_reason_when_demo_halts(
        self, monkeypatch, capsys
    ):
        async def _fake_run_voice_demo(**kwargs):
            assert kwargs["score_every"] == 2
            assert kwargs["hard_limit"] == 0.9
            return VoiceDemoResult(
                audio_chunks=[b"audio:recovery"],
                tts_texts=["recovery"],
                halted=True,
                halt_reason="hard_limit",
                recovery_text="recovery",
            )

        monkeypatch.setattr(
            "director_ai.voice.demo.run_voice_demo",
            _fake_run_voice_demo,
        )

        exit_code = await _amain(
            [
                "--response",
                "Unsupported claim.",
                "--score-every",
                "2",
                "--hard-limit",
                "0.9",
            ]
        )

        output = capsys.readouterr().out
        assert exit_code == 0
        assert "status=halted" in output
        assert "halt_reason=hard_limit" in output
        assert "tts_calls=1" in output
        assert "audio_chunks=1" in output
        assert "audio_bytes=14" in output

    async def test_async_cli_omits_halt_reason_when_demo_completes(
        self, monkeypatch, capsys
    ):
        async def _fake_run_voice_demo(**kwargs):
            assert kwargs["score_every"] == 4
            return VoiceDemoResult(
                audio_chunks=[b"audio:ok"],
                tts_texts=["ok"],
                halted=False,
            )

        monkeypatch.setattr(
            "director_ai.voice.demo.run_voice_demo",
            _fake_run_voice_demo,
        )

        exit_code = await _amain([])

        output = capsys.readouterr().out
        assert exit_code == 0
        assert "status=completed" in output
        assert "halt_reason=" not in output
        assert "tts_calls=1" in output
        assert "audio_chunks=1" in output
        assert "audio_bytes=8" in output

    def test_main_runs_async_cli(self, monkeypatch):
        async def _fake_amain(argv):
            assert argv == ["--score-every", "8"]
            return 7

        monkeypatch.setattr("director_ai.voice.demo._amain", _fake_amain)

        assert main(["--score-every", "8"]) == 7

    def test_module_cli_runs_without_runtime_warning(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = ".:src"
        completed = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::RuntimeWarning",
                "-m",
                "director_ai.voice.demo",
                "--response",
                "The Team plan costs CHF 19 per seat.",
                "--score-every",
                "32",
            ],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert "status=completed" in completed.stdout
