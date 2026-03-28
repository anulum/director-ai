# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TTS Adapter Tests
"""Tests for TTSAdapter ABC and provider adapters."""

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
