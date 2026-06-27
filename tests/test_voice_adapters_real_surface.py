# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - voice adapter real-surface tests
"""Real SDK-construction coverage for voice TTS adapters."""

from __future__ import annotations

from director_ai.voice.adapters import (
    DeepgramAdapter,
    ElevenLabsAdapter,
    OpenAITTSAdapter,
)


def test_elevenlabs_adapter_constructs_real_async_sdk_client() -> None:
    """ElevenLabs adapter should cache the installed async SDK client object."""
    adapter = ElevenLabsAdapter(api_key="elevenlabs-local-test")

    client: object = adapter._get_client()

    assert client is adapter._get_client()
    assert type(client).__module__ == "elevenlabs.client"
    assert type(client).__name__ == "AsyncElevenLabs"


async def test_openai_tts_adapter_constructs_and_closes_real_async_client() -> None:
    """OpenAI TTS adapter should construct and close the installed async client."""
    adapter = OpenAITTSAdapter(api_key="sk-local-test")

    client: object = adapter._get_client()

    assert client is adapter._get_client()
    assert type(client).__module__ == "openai"
    assert type(client).__name__ == "AsyncOpenAI"

    await adapter.close()

    assert adapter._client is None


def test_deepgram_adapter_constructs_real_async_sdk_client() -> None:
    """Deepgram adapter should cache the installed async SDK client object."""
    adapter = DeepgramAdapter(api_key="deepgram-local-test")

    client: object = adapter._get_client()

    assert client is adapter._get_client()
    assert type(client).__module__ == "deepgram.client"
    assert type(client).__name__ == "AsyncDeepgramClient"
