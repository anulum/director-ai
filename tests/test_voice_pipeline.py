# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Voice Pipeline Tests
"""Tests for voice_pipeline() integration."""

from __future__ import annotations

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
