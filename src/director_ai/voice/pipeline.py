# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Voice Pipeline (Guard + TTS)
"""Wires AsyncVoiceGuard and a TTSAdapter into one async audio stream.

Usage::

    from director_ai.voice import voice_pipeline, ElevenLabsAdapter

    tts = ElevenLabsAdapter(voice_id="JBFqnCBsd6RMkjVDRZzb")

    async for audio_chunk in voice_pipeline(llm_tokens, tts, facts=my_facts):
        await websocket.send_bytes(audio_chunk)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.integrations.voice import VoiceToken

from .adapters import TTSAdapter
from .guard import AsyncVoiceGuard

__all__ = ["voice_pipeline"]

_SENTENCE_ENDS = frozenset(".!?")


async def voice_pipeline(
    token_source: AsyncIterator[str] | Iterator[str],
    tts: TTSAdapter,
    *,
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    prompt: str = "",
    threshold: float = 0.3,
    hard_limit: float = 0.25,
    score_every: int = 4,
    soft_halt: bool = True,
    recovery: str = "I need to verify that information. One moment.",
    use_nli: bool = True,
    on_halt: Callable[[VoiceToken], Awaitable[None] | None] | None = None,
    sentence_buffer: bool = True,
) -> AsyncIterator[bytes]:
    """Stream guardrailed audio from an LLM token stream.

    Feeds tokens through :class:`AsyncVoiceGuard`, buffers approved text
    into sentence-sized chunks, synthesises via the given
    :class:`TTSAdapter`, and yields audio bytes as they arrive.

    On halt the pipeline flushes any buffered text, synthesises the
    recovery message, calls ``on_halt``, then stops.

    Parameters
    ----------
    token_source : async or sync iterable of str — LLM token stream.
    tts : TTSAdapter — audio synthesis backend.
    facts, store, prompt, threshold, hard_limit, score_every, soft_halt,
        recovery, use_nli : forwarded to AsyncVoiceGuard.
    on_halt : optional callback receiving the halting VoiceToken.
    sentence_buffer : if True (default), batch tokens into sentences
        before sending to TTS for natural prosody. If False, send each
        approved token immediately (lower latency, choppier audio).

    """
    guard = AsyncVoiceGuard(
        facts=facts,
        store=store,
        threshold=threshold,
        score_every=score_every,
        hard_limit=hard_limit,
        soft_halt=soft_halt,
        recovery=recovery,
        use_nli=use_nli,
        prompt=prompt,
    )

    text_buf: list[str] = []

    async def _flush_buffer():
        if not text_buf:
            return
        chunk = "".join(text_buf)
        text_buf.clear()
        async for audio in tts.synthesise(chunk):
            yield audio

    async for vtoken in guard.feed_stream(token_source):
        if vtoken.halted:
            # Flush remaining buffered text
            if vtoken.approved:
                text_buf.append(vtoken.token)
            async for audio in _flush_buffer():
                yield audio

            # Synthesise recovery message
            if vtoken.recovery_text:
                async for audio in tts.synthesise(vtoken.recovery_text):
                    yield audio

            # Fire callback
            if on_halt is not None:
                result = on_halt(vtoken)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result

            async for audio in tts.flush():
                yield audio
            await tts.close()
            return

        if not vtoken.approved:
            continue

        if sentence_buffer:
            text_buf.append(vtoken.token)
            stripped = vtoken.token.rstrip()
            if stripped and stripped[-1] in _SENTENCE_ENDS:
                async for audio in _flush_buffer():
                    yield audio
        else:
            async for audio in tts.synthesise(vtoken.token):
                yield audio

    # Stream ended without halt — flush remaining
    async for audio in _flush_buffer():
        yield audio
    async for audio in tts.flush():
        yield audio
    await tts.close()
