# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Voice Adapter SDK-Aware Tests
"""Tests that import real TTS SDKs and mock only the HTTP/API layer.

These tests run only when the SDKs are installed (CI extras matrix).
They verify that adapter code correctly constructs SDK clients and
calls the right methods with the right parameters.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


class TestElevenLabsAdapterSDK:
    def test_get_client_creates_real_sdk_client(self):
        pytest = __import__("pytest")
        pytest.importorskip("elevenlabs")
        from director_ai.voice.adapters import ElevenLabsAdapter

        adapter = ElevenLabsAdapter(api_key="test-key-123")
        client = adapter._get_client()
        assert client is not None
        assert adapter._client is client
        # Second call returns cached client
        assert adapter._get_client() is client

    async def test_synthesise_calls_convert(self):
        pytest = __import__("pytest")
        pytest.importorskip("elevenlabs")
        from director_ai.voice.adapters import ElevenLabsAdapter

        adapter = ElevenLabsAdapter(
            voice_id="test-voice",
            model_id="test-model",
            api_key="test-key",
        )

        class FakeAudioStream:
            async def __aiter__(self):
                for chunk in [b"chunk1", b"chunk2"]:
                    yield chunk

        mock_convert = AsyncMock(return_value=FakeAudioStream())
        mock_client = MagicMock()
        mock_client.text_to_speech.convert = mock_convert
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Hello world")]
        mock_convert.assert_awaited_once_with(
            text="Hello world",
            voice_id="test-voice",
            model_id="test-model",
        )
        assert len(chunks) > 0

    async def test_synthesise_non_async_response(self):
        pytest = __import__("pytest")
        pytest.importorskip("elevenlabs")
        from director_ai.voice.adapters import ElevenLabsAdapter

        adapter = ElevenLabsAdapter(api_key="test-key")
        mock_client = MagicMock()
        mock_client.text_to_speech.convert = AsyncMock(return_value=b"raw-audio")
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Hi")]
        assert chunks == [b"raw-audio"]


class TestOpenAITTSAdapterSDK:
    def test_get_client_creates_real_sdk_client(self):
        pytest = __import__("pytest")
        pytest.importorskip("openai")
        from director_ai.voice.adapters import OpenAITTSAdapter

        adapter = OpenAITTSAdapter(api_key="sk-test-key")
        client = adapter._get_client()
        assert client is not None
        assert adapter._client is client

    async def test_synthesise_calls_speech_create(self):
        pytest = __import__("pytest")
        pytest.importorskip("openai")
        from director_ai.voice.adapters import OpenAITTSAdapter

        adapter = OpenAITTSAdapter(
            voice="alloy",
            model="tts-1-hd",
            api_key="sk-test",
            response_format="opus",
        )

        mock_response = MagicMock()

        async def fake_iter_bytes(size):
            for chunk in [b"audio1", b"audio2", b"audio3"]:
                yield chunk

        mock_response.iter_bytes = fake_iter_bytes
        mock_create = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.audio.speech.create = mock_create
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Test text")]
        mock_create.assert_awaited_once_with(
            model="tts-1-hd",
            voice="alloy",
            input="Test text",
            response_format="opus",
        )
        assert chunks == [b"audio1", b"audio2", b"audio3"]

    async def test_close_calls_client_close(self):
        pytest = __import__("pytest")
        pytest.importorskip("openai")
        from director_ai.voice.adapters import OpenAITTSAdapter

        adapter = OpenAITTSAdapter(api_key="sk-test")
        mock_client = AsyncMock()
        adapter._client = mock_client
        await adapter.close()
        mock_client.close.assert_awaited_once()
        assert adapter._client is None


class TestDeepgramAdapterSDK:
    def test_get_client_creates_real_sdk_client(self):
        pytest = __import__("pytest")
        pytest.importorskip("deepgram")
        from director_ai.voice.adapters import DeepgramAdapter

        adapter = DeepgramAdapter(api_key="dg-test-key")
        client = adapter._get_client()
        assert client is not None
        assert adapter._client is client

    async def test_synthesise_calls_stream_raw(self):
        pytest = __import__("pytest")
        pytest.importorskip("deepgram")
        from director_ai.voice.adapters import DeepgramAdapter

        adapter = DeepgramAdapter(model="aura-zeus-en", api_key="dg-test")

        class FakeAsyncResponse:
            async def __aiter__(self):
                for chunk in [b"dg1", b"dg2"]:
                    yield chunk

        mock_response = FakeAsyncResponse()
        mock_stream_raw = AsyncMock(return_value=mock_response)
        mock_v = MagicMock()
        mock_v.stream_raw = mock_stream_raw
        mock_asyncrest = MagicMock()
        mock_asyncrest.v.return_value = mock_v
        mock_client = MagicMock()
        mock_client.speak.asyncrest = mock_asyncrest
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Hello")]
        mock_asyncrest.v.assert_called_once_with("1")
        mock_stream_raw.assert_awaited_once_with(
            {"model": "aura-zeus-en", "text": "Hello"},
        )
        assert len(chunks) > 0

    async def test_synthesise_sync_stream_fallback(self):
        pytest = __import__("pytest")
        pytest.importorskip("deepgram")
        from director_ai.voice.adapters import DeepgramAdapter

        adapter = DeepgramAdapter(api_key="dg-test")
        mock_response = MagicMock(spec=[])
        mock_response.stream = [b"s1", b"s2"]
        mock_stream_raw = AsyncMock(return_value=mock_response)
        mock_v = MagicMock()
        mock_v.stream_raw = mock_stream_raw
        mock_asyncrest = MagicMock()
        mock_asyncrest.v.return_value = mock_v
        mock_client = MagicMock()
        mock_client.speak.asyncrest = mock_asyncrest
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Test")]
        assert chunks == [b"s1", b"s2"]

    async def test_synthesise_content_fallback(self):
        pytest = __import__("pytest")
        pytest.importorskip("deepgram")
        from director_ai.voice.adapters import DeepgramAdapter

        adapter = DeepgramAdapter(api_key="dg-test")
        mock_response = MagicMock(spec=[])
        mock_response.content = b"raw-content"
        mock_stream_raw = AsyncMock(return_value=mock_response)
        mock_v = MagicMock()
        mock_v.stream_raw = mock_stream_raw
        mock_asyncrest = MagicMock()
        mock_asyncrest.v.return_value = mock_v
        mock_client = MagicMock()
        mock_client.speak.asyncrest = mock_asyncrest
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Test")]
        assert chunks == [b"raw-content"]
