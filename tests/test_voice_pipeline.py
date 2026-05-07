# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Voice Pipeline Tests
"""Multi-angle tests for voice_pipeline() integration.

Covers: text-to-speech pipeline, guard integration, hallucination detection
in voice output, coherence threshold enforcement, adapter routing,
pipeline wiring, and performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.integrations.voice import VoiceToken
from director_ai.voice import pipeline as pipeline_module
from director_ai.voice.adapters import TTSAdapter
from director_ai.voice.pipeline import voice_pipeline


class RecordingAdapter(TTSAdapter):
    """Records all text sent to TTS and yields deterministic audio."""

    def __init__(self):
        self.texts: list[str] = []
        self.closed = False

    async def synthesise(self, text: str):
        self.texts.append(text)
        yield b"audio:" + text.encode()

    async def close(self):
        self.closed = True


class FlushRecordingAdapter(RecordingAdapter):
    def __init__(self):
        super().__init__()
        self.flushed = False

    async def flush(self):
        self.flushed = True
        yield b"flush"


class ScriptedGuard:
    tokens: list[VoiceToken] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def feed_stream(self, token_source):
        for _item in token_source:
            pass
        for token in self.tokens:
            yield token


class TestVoicePipelineNormal:
    async def test_full_stream_produces_audio(self):
        tts = RecordingAdapter()
        tokens = ["Hello ", "world."]
        audio = [
            chunk
            async for chunk in voice_pipeline(
                iter(tokens),
                tts,
                facts={"greeting": "Hello world."},
                prompt="greeting",
                use_nli=False,
                score_every=1,
            )
        ]
        assert len(audio) > 0
        assert all(isinstance(c, bytes) for c in audio)
        assert tts.closed

    async def test_sentence_buffer_batches_at_period(self):
        tts = RecordingAdapter()
        tokens = ["Hello ", "world", ". ", "More ", "text."]
        _ = [
            chunk
            async for chunk in voice_pipeline(
                iter(tokens),
                tts,
                facts={"greeting": "Hello world. More text."},
                prompt="greeting",
                use_nli=False,
                score_every=1,
                sentence_buffer=True,
            )
        ]
        # First sentence ends at ". " — should be one TTS call
        # Second sentence ends at "text." — another TTS call
        assert len(tts.texts) >= 2
        assert "Hello " in tts.texts[0]

    async def test_no_sentence_buffer_sends_each_token(self):
        tts = RecordingAdapter()
        tokens = ["Hello ", "world."]
        _ = [
            chunk
            async for chunk in voice_pipeline(
                iter(tokens),
                tts,
                facts={"greeting": "Hello world."},
                prompt="greeting",
                use_nli=False,
                score_every=1,
                sentence_buffer=False,
            )
        ]
        assert len(tts.texts) == 2

    async def test_async_token_source(self):
        tts = RecordingAdapter()

        async def async_tokens():
            for t in ["Hi.", " Bye."]:
                yield t

        audio = [
            chunk
            async for chunk in voice_pipeline(
                async_tokens(),
                tts,
                facts={"greet": "Hi. Bye."},
                prompt="greet",
                use_nli=False,
                score_every=1,
            )
        ]
        assert len(audio) > 0
        assert tts.closed

    async def test_rejected_guard_tokens_are_not_synthesised(self, monkeypatch):
        class RejectThenApproveGuard(ScriptedGuard):
            tokens = [
                VoiceToken("unsafe ", 0, approved=False, coherence=0.1),
                VoiceToken("Safe.", 1, approved=True, coherence=1.0),
            ]

        monkeypatch.setattr(pipeline_module, "AsyncVoiceGuard", RejectThenApproveGuard)
        tts = FlushRecordingAdapter()

        audio = [
            chunk
            async for chunk in voice_pipeline(
                iter(["ignored"]),
                tts,
                sentence_buffer=True,
            )
        ]

        assert tts.texts == ["Safe."]
        assert audio == [b"audio:Safe.", b"flush"]
        assert tts.flushed
        assert tts.closed

    async def test_unfinished_sentence_is_flushed_at_stream_end(self, monkeypatch):
        class TrailingTextGuard(ScriptedGuard):
            tokens = [
                VoiceToken("Trailing approved text", 0, approved=True, coherence=1.0)
            ]

        monkeypatch.setattr(pipeline_module, "AsyncVoiceGuard", TrailingTextGuard)
        tts = FlushRecordingAdapter()

        audio = [
            chunk
            async for chunk in voice_pipeline(
                iter(["ignored"]),
                tts,
                sentence_buffer=True,
            )
        ]

        assert tts.texts == ["Trailing approved text"]
        assert audio == [b"audio:Trailing approved text", b"flush"]
        assert tts.closed


class TestVoicePipelineHalt:
    async def test_halt_produces_recovery_audio(self):
        tts = RecordingAdapter()
        tokens = ["a", "b", "c", "d", "e"]
        recovery = "Let me check."
        _ = [
            chunk
            async for chunk in voice_pipeline(
                iter(tokens),
                tts,
                use_nli=False,
                score_every=1,
                hard_limit=0.99,
                recovery=recovery,
            )
        ]
        recovery_sent = any(recovery in t for t in tts.texts)
        assert recovery_sent
        assert tts.closed

    async def test_on_halt_callback_fires(self):
        tts = RecordingAdapter()
        halt_tokens = []

        def on_halt(vtoken):
            halt_tokens.append(vtoken)

        tokens = ["a", "b", "c"]
        _ = [
            chunk
            async for chunk in voice_pipeline(
                iter(tokens),
                tts,
                use_nli=False,
                score_every=1,
                hard_limit=0.99,
                on_halt=on_halt,
            )
        ]
        assert len(halt_tokens) == 1
        assert halt_tokens[0].halted

    async def test_async_on_halt_callback(self):
        tts = RecordingAdapter()
        halt_tokens = []

        async def on_halt(vtoken):
            halt_tokens.append(vtoken)

        tokens = ["a", "b", "c"]
        _ = [
            chunk
            async for chunk in voice_pipeline(
                iter(tokens),
                tts,
                use_nli=False,
                score_every=1,
                hard_limit=0.99,
                on_halt=on_halt,
            )
        ]
        assert len(halt_tokens) == 1

    async def test_halt_flushes_approved_token_without_recovery(self, monkeypatch):
        class ApprovedHaltGuard(ScriptedGuard):
            tokens = [
                VoiceToken(
                    "Final approved sentence.",
                    0,
                    approved=True,
                    coherence=0.2,
                    halted=True,
                    halt_reason="threshold",
                    recovery_text="",
                )
            ]

        monkeypatch.setattr(pipeline_module, "AsyncVoiceGuard", ApprovedHaltGuard)
        tts = FlushRecordingAdapter()

        audio = [
            chunk
            async for chunk in voice_pipeline(
                iter(["ignored"]),
                tts,
                on_halt=None,
            )
        ]

        assert tts.texts == ["Final approved sentence."]
        assert audio == [b"audio:Final approved sentence.", b"flush"]
        assert tts.flushed
        assert tts.closed


class TestVoicePipelineEmpty:
    async def test_empty_stream(self):
        tts = RecordingAdapter()
        audio = [
            chunk
            async for chunk in voice_pipeline(
                iter([]),
                tts,
                use_nli=False,
            )
        ]
        assert audio == []
        assert tts.closed


class TestVoicePipelineParametrised:
    """Parametrised voice pipeline tests."""

    @pytest.mark.parametrize("score_every", [1, 2, 5])
    async def test_various_score_intervals(self, score_every):
        tts = RecordingAdapter()
        tokens = ["Hello ", "world", ". ", "More ", "text."]
        audio = [
            chunk
            async for chunk in voice_pipeline(
                iter(tokens),
                tts,
                use_nli=False,
                score_every=score_every,
            )
        ]
        assert len(audio) > 0
        assert tts.closed

    @pytest.mark.parametrize("sentence_buffer", [True, False])
    async def test_buffer_modes(self, sentence_buffer):
        tts = RecordingAdapter()
        tokens = ["A. ", "B."]
        _ = [
            chunk
            async for chunk in voice_pipeline(
                iter(tokens),
                tts,
                use_nli=False,
                score_every=1,
                sentence_buffer=sentence_buffer,
            )
        ]
        assert len(tts.texts) >= 1


class TestVoicePipelinePerformanceDoc:
    """Document voice pipeline performance characteristics."""

    async def test_audio_chunks_are_bytes(self):
        tts = RecordingAdapter()
        audio = [
            chunk
            async for chunk in voice_pipeline(
                iter(["Test."]),
                tts,
                use_nli=False,
                score_every=1,
            )
        ]
        for chunk in audio:
            assert isinstance(chunk, bytes)

    async def test_adapter_closed_after_pipeline(self):
        tts = RecordingAdapter()
        _ = [
            chunk
            async for chunk in voice_pipeline(
                iter(["Test."]),
                tts,
                use_nli=False,
            )
        ]
        assert tts.closed
