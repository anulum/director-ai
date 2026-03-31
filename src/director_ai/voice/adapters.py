# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TTS Adapters for Voice AI Pipeline
"""TTS adapter ABC and provider implementations.

Each adapter wraps a TTS SDK and converts approved text fragments into
streaming audio bytes. SDKs are lazy-imported — install only the one
you need::

    pip install elevenlabs     # for ElevenLabsAdapter
    pip install openai         # for OpenAITTSAdapter
    pip install deepgram-sdk   # for DeepgramAdapter
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from director_ai.core.exceptions import DependencyError

__all__ = [
    "TTSAdapter",
    "ElevenLabsAdapter",
    "OpenAITTSAdapter",
    "DeepgramAdapter",
]


class TTSAdapter(ABC):
    """Converts approved text into streaming audio bytes."""

    @abstractmethod
    async def synthesise(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio bytes for a text fragment."""
        ...  # pragma: no cover
        yield b""  # pragma: no cover

    async def flush(self) -> AsyncIterator[bytes]:
        """Flush any buffered audio. Override if your TTS buffers internally."""
        return
        yield b""  # makes this a valid async generator

    async def close(self) -> None:
        """Release resources. Override in subclasses that hold connections."""
        return None


class ElevenLabsAdapter(TTSAdapter):
    """ElevenLabs streaming TTS via their async SDK.

    Requires ``pip install elevenlabs``.

    Parameters
    ----------
    voice_id : str — ElevenLabs voice ID.
    model_id : str — TTS model (default: eleven_turbo_v2_5).
    api_key : str or None — API key (defaults to ELEVENLABS_API_KEY env var).

    """

    def __init__(
        self,
        *,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: str = "eleven_turbo_v2_5",
        api_key: str | None = None,
    ):
        self._voice_id = voice_id
        self._model_id = model_id
        self._api_key = api_key
        self._client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from elevenlabs.client import AsyncElevenLabs
        except ImportError:
            raise DependencyError(
                "ElevenLabsAdapter requires the elevenlabs package: "
                "pip install elevenlabs"
            ) from None
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        self._client = AsyncElevenLabs(**kwargs)
        return self._client

    async def synthesise(self, text: str) -> AsyncIterator[bytes]:
        client = self._get_client()
        audio_stream = await client.text_to_speech.convert(
            text=text,
            voice_id=self._voice_id,
            model_id=self._model_id,
        )
        if hasattr(audio_stream, "__aiter__"):
            async for chunk in audio_stream:
                yield chunk
        else:
            yield audio_stream

    async def close(self) -> None:
        self._client = None


class OpenAITTSAdapter(TTSAdapter):
    """OpenAI TTS streaming via their async SDK.

    Requires ``pip install openai``.

    Parameters
    ----------
    voice : str — TTS voice (default: nova).
    model : str — TTS model (default: tts-1).
    api_key : str or None — API key (defaults to OPENAI_API_KEY env var).
    response_format : str — audio format (default: mp3).

    """

    def __init__(
        self,
        *,
        voice: str = "nova",
        model: str = "tts-1",
        api_key: str | None = None,
        response_format: str = "mp3",
    ):
        self._voice = voice
        self._model = model
        self._api_key = api_key
        self._response_format = response_format
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise DependencyError(
                "OpenAITTSAdapter requires the openai package: pip install openai"
            ) from None
        self._client = (
            AsyncOpenAI(api_key=self._api_key) if self._api_key else AsyncOpenAI()
        )
        return self._client

    async def synthesise(self, text: str) -> AsyncIterator[bytes]:
        client = self._get_client()
        response = await client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format=self._response_format,
        )
        async for chunk in response.iter_bytes(1024):
            yield chunk

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


class DeepgramAdapter(TTSAdapter):
    """Deepgram TTS streaming via their async SDK.

    Requires ``pip install deepgram-sdk``.

    Parameters
    ----------
    model : str — TTS model (default: aura-asteria-en).
    api_key : str or None — API key (defaults to DEEPGRAM_API_KEY env var).

    """

    def __init__(
        self,
        *,
        model: str = "aura-asteria-en",
        api_key: str | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from deepgram import DeepgramClient
        except ImportError:
            raise DependencyError(
                "DeepgramAdapter requires the deepgram-sdk package: "
                "pip install deepgram-sdk"
            ) from None
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        self._client = DeepgramClient(**kwargs)
        return self._client

    async def synthesise(self, text: str) -> AsyncIterator[bytes]:
        client = self._get_client()
        options = {"model": self._model, "text": text}
        response = await client.speak.asyncrest.v("1").stream_raw(options)
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                yield chunk
        elif hasattr(response, "stream"):
            for chunk in response.stream:
                yield chunk
        else:
            yield response.content if hasattr(response, "content") else bytes(response)

    async def close(self) -> None:
        self._client = None
