# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TTS Adapter Tests
"""Multi-angle tests for TTS adapter pipeline.

Covers: ElevenLabs, OpenAI, Deepgram adapters, async synthesis, close
lifecycle, error handling, pipeline integration with voice_pipeline(),
and performance documentation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from director_ai.core.exceptions import DependencyError
from director_ai.voice.adapters import (
    DeepgramAdapter,
    ElevenLabsAdapter,
    OpenAITTSAdapter,
    TTSAdapter,
)


class MockTTSAdapter(TTSAdapter):
    """Test adapter that yields deterministic bytes."""

    def __init__(self, chunk: bytes = b"audio"):
        self._chunk = chunk
        self.closed = False
        self.flushed = False

    async def synthesise(self, text: str) -> None:
        yield self._chunk
        yield self._chunk

    async def flush(self):
        self.flushed = True
        yield b"flush"

    async def close(self) -> None:
        self.closed = True


class TestMockTTSAdapter:
    async def test_synthesise_yields_bytes(self):
        adapter = MockTTSAdapter(chunk=b"test_audio")
        chunks = [c async for c in adapter.synthesise("Hello")]
        assert chunks == [b"test_audio", b"test_audio"]

    async def test_flush_yields_bytes(self):
        adapter = MockTTSAdapter()
        chunks = [c async for c in adapter.flush()]
        assert chunks == [b"flush"]
        assert adapter.flushed

    async def test_close(self):
        adapter = MockTTSAdapter()
        await adapter.close()
        assert adapter.closed


class TestTTSAdapterBaseDefaults:
    async def test_default_flush_yields_nothing(self):
        class MinimalAdapter(TTSAdapter):
            async def synthesise(self, text):
                yield b"data"

        adapter = MinimalAdapter()
        chunks = [c async for c in adapter.flush()]
        assert chunks == []

    async def test_default_close_is_noop(self):
        class MinimalAdapter(TTSAdapter):
            async def synthesise(self, text):
                yield b"data"

        adapter = MinimalAdapter()
        await adapter.close()


class TestElevenLabsAdapterMissing:
    async def test_raises_dependency_error_when_sdk_missing(self):
        adapter = ElevenLabsAdapter(voice_id="test")
        with patch.dict("sys.modules", {"elevenlabs": None, "elevenlabs.client": None}):
            adapter._client = None
            with pytest.raises(DependencyError, match="elevenlabs"):
                adapter._get_client()

    def test_stores_config(self):
        adapter = ElevenLabsAdapter(
            voice_id="abc",
            model_id="turbo",
            api_key="key123",
        )
        assert adapter._voice_id == "abc"
        assert adapter._model_id == "turbo"
        assert adapter._api_key == "key123"

    async def test_close_clears_client(self):
        adapter = ElevenLabsAdapter()
        adapter._client = "fake"
        await adapter.close()
        assert adapter._client is None


class TestOpenAITTSAdapterMissing:
    async def test_raises_dependency_error_when_sdk_missing(self):
        adapter = OpenAITTSAdapter(voice="alloy")
        with patch.dict("sys.modules", {"openai": None}):
            adapter._client = None
            with pytest.raises(DependencyError, match="openai"):
                adapter._get_client()

    def test_stores_config(self):
        adapter = OpenAITTSAdapter(
            voice="alloy",
            model="tts-1-hd",
            api_key="sk-test",
            response_format="opus",
        )
        assert adapter._voice == "alloy"
        assert adapter._model == "tts-1-hd"
        assert adapter._api_key == "sk-test"
        assert adapter._response_format == "opus"

    async def test_close_with_no_client(self):
        adapter = OpenAITTSAdapter()
        await adapter.close()
        assert adapter._client is None


class TestDeepgramAdapterMissing:
    async def test_raises_dependency_error_when_sdk_missing(self):
        adapter = DeepgramAdapter(model="aura-asteria-en")
        with patch.dict("sys.modules", {"deepgram": None}):
            adapter._client = None
            with pytest.raises(DependencyError, match="deepgram"):
                adapter._get_client()

    def test_stores_config(self):
        adapter = DeepgramAdapter(model="aura-zeus-en", api_key="dg-test")
        assert adapter._model == "aura-zeus-en"
        assert adapter._api_key == "dg-test"

    async def test_close_clears_client(self):
        adapter = DeepgramAdapter()
        adapter._client = "fake"
        await adapter.close()
        assert adapter._client is None


# ---------------------------------------------------------------------------
# Happy-path tests with mocked SDK clients
# ---------------------------------------------------------------------------


class TestElevenLabsAdapterHappyPath:
    """Test ElevenLabsAdapter with a mocked elevenlabs client."""

    async def test_get_client_creates_instance(self):
        from unittest.mock import MagicMock

        mock_module = MagicMock()
        mock_client_cls = MagicMock()
        mock_module.AsyncElevenLabs = mock_client_cls
        with patch.dict(
            "sys.modules", {"elevenlabs": MagicMock(), "elevenlabs.client": mock_module}
        ):
            adapter = ElevenLabsAdapter(voice_id="v1", api_key="key")
            adapter._client = None
            client = adapter._get_client()
            mock_client_cls.assert_called_once_with(api_key="key")
            assert client is mock_client_cls.return_value

    async def test_get_client_without_api_key(self):
        from unittest.mock import MagicMock

        mock_module = MagicMock()
        mock_client_cls = MagicMock()
        mock_module.AsyncElevenLabs = mock_client_cls
        with patch.dict(
            "sys.modules", {"elevenlabs": MagicMock(), "elevenlabs.client": mock_module}
        ):
            adapter = ElevenLabsAdapter(voice_id="v1")
            adapter._client = None
            adapter._get_client()
            mock_client_cls.assert_called_once_with()

    async def test_get_client_returns_cached(self):
        adapter = ElevenLabsAdapter()
        adapter._client = "cached"
        assert adapter._get_client() == "cached"

    async def test_synthesise_async_iterable(self):
        from unittest.mock import AsyncMock, MagicMock

        adapter = ElevenLabsAdapter(voice_id="v1", model_id="m1")

        async def fake_stream():
            yield b"chunk1"
            yield b"chunk2"

        mock_client = MagicMock()
        mock_client.text_to_speech.convert = AsyncMock(return_value=fake_stream())
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Hello")]
        assert chunks == [b"chunk1", b"chunk2"]
        mock_client.text_to_speech.convert.assert_called_once_with(
            text="Hello", voice_id="v1", model_id="m1"
        )

    async def test_synthesise_non_iterable(self):
        from unittest.mock import AsyncMock, MagicMock

        adapter = ElevenLabsAdapter(voice_id="v1")
        mock_client = MagicMock()
        mock_client.text_to_speech.convert = AsyncMock(return_value=b"raw_audio")
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Hi")]
        assert chunks == [b"raw_audio"]


class TestOpenAITTSAdapterHappyPath:
    """Test OpenAITTSAdapter with a mocked openai client."""

    async def test_get_client_with_api_key(self):
        from unittest.mock import MagicMock

        mock_openai = MagicMock()
        mock_cls = MagicMock()
        mock_openai.AsyncOpenAI = mock_cls
        with patch.dict("sys.modules", {"openai": mock_openai}):
            adapter = OpenAITTSAdapter(api_key="sk-test")
            adapter._client = None
            client = adapter._get_client()
            mock_cls.assert_called_once_with(api_key="sk-test")
            assert client is mock_cls.return_value

    async def test_get_client_without_api_key(self):
        from unittest.mock import MagicMock

        mock_openai = MagicMock()
        mock_cls = MagicMock()
        mock_openai.AsyncOpenAI = mock_cls
        with patch.dict("sys.modules", {"openai": mock_openai}):
            adapter = OpenAITTSAdapter()
            adapter._client = None
            adapter._get_client()
            mock_cls.assert_called_once_with()

    async def test_get_client_returns_cached(self):
        adapter = OpenAITTSAdapter()
        adapter._client = "cached"
        assert adapter._get_client() == "cached"

    async def test_synthesise_streams_bytes(self):
        from unittest.mock import AsyncMock, MagicMock

        adapter = OpenAITTSAdapter(voice="nova", model="tts-1", response_format="mp3")

        async def fake_iter(size):
            yield b"audio1"
            yield b"audio2"

        mock_response = MagicMock()
        mock_response.iter_bytes = fake_iter

        mock_client = MagicMock()
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Test text")]
        assert chunks == [b"audio1", b"audio2"]
        mock_client.audio.speech.create.assert_called_once_with(
            model="tts-1", voice="nova", input="Test text", response_format="mp3"
        )

    async def test_close_with_active_client(self):
        from unittest.mock import AsyncMock, MagicMock

        adapter = OpenAITTSAdapter()
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        adapter._client = mock_client
        await adapter.close()
        mock_client.close.assert_called_once()
        assert adapter._client is None


class TestDeepgramAdapterHappyPath:
    """Test DeepgramAdapter with a mocked deepgram client."""

    async def test_get_client_with_api_key(self):
        from unittest.mock import MagicMock

        mock_dg = MagicMock()
        mock_cls = MagicMock()
        mock_dg.DeepgramClient = mock_cls
        with patch.dict("sys.modules", {"deepgram": mock_dg}):
            adapter = DeepgramAdapter(api_key="dg-key")
            adapter._client = None
            client = adapter._get_client()
            mock_cls.assert_called_once_with(api_key="dg-key")
            assert client is mock_cls.return_value

    async def test_get_client_without_api_key(self):
        from unittest.mock import MagicMock

        mock_dg = MagicMock()
        mock_cls = MagicMock()
        mock_dg.DeepgramClient = mock_cls
        with patch.dict("sys.modules", {"deepgram": mock_dg}):
            adapter = DeepgramAdapter()
            adapter._client = None
            adapter._get_client()
            mock_cls.assert_called_once_with()

    async def test_get_client_returns_cached(self):
        adapter = DeepgramAdapter()
        adapter._client = "cached"
        assert adapter._get_client() == "cached"

    async def test_synthesise_async_iterable(self):
        from unittest.mock import AsyncMock, MagicMock

        adapter = DeepgramAdapter(model="aura-zeus-en")

        async def fake_stream():
            yield b"dg1"
            yield b"dg2"

        mock_response = fake_stream()
        mock_client = MagicMock()
        mock_v = MagicMock()
        mock_v.stream_raw = AsyncMock(return_value=mock_response)
        mock_client.speak.asyncrest.v.return_value = mock_v
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Hi")]
        assert chunks == [b"dg1", b"dg2"]

    async def test_synthesise_stream_attribute(self):
        from unittest.mock import AsyncMock, MagicMock

        adapter = DeepgramAdapter()
        mock_response = MagicMock(spec=[])
        mock_response.stream = [b"s1", b"s2"]
        mock_client = MagicMock()
        mock_v = MagicMock()
        mock_v.stream_raw = AsyncMock(return_value=mock_response)
        mock_client.speak.asyncrest.v.return_value = mock_v
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Stream")]
        assert chunks == [b"s1", b"s2"]

    async def test_synthesise_content_fallback(self):
        from unittest.mock import AsyncMock, MagicMock

        adapter = DeepgramAdapter()
        mock_response = MagicMock(spec=[])
        mock_response.content = b"raw_content"
        del mock_response.stream
        mock_client = MagicMock()
        mock_v = MagicMock()
        mock_v.stream_raw = AsyncMock(return_value=mock_response)
        mock_client.speak.asyncrest.v.return_value = mock_v
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Content")]
        assert chunks == [b"raw_content"]

    async def test_synthesise_bytes_fallback(self):
        from unittest.mock import AsyncMock, MagicMock

        adapter = DeepgramAdapter()
        mock_response = b"raw_bytes"
        mock_client = MagicMock()
        mock_v = MagicMock()
        mock_v.stream_raw = AsyncMock(return_value=mock_response)
        mock_client.speak.asyncrest.v.return_value = mock_v
        adapter._client = mock_client

        chunks = [c async for c in adapter.synthesise("Bytes")]
        assert chunks == [b"raw_bytes"]
