# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming Voice AI Demo
"""True streaming voice output with hallucination halting.

Streams an LLM response through AsyncVoiceGuard and pipes approved
text to ElevenLabs TTS in real-time. Audio chunks arrive as the LLM
generates — if a hallucination is detected mid-stream, the pipeline
halts and speaks a recovery message.

Requirements::

    pip install director-ai[nli] openai elevenlabs

Set environment variables::

    OPENAI_API_KEY=sk-...
    ELEVENLABS_API_KEY=...

Usage::

    python examples/voice_streaming_demo.py
"""

from __future__ import annotations

import asyncio

PRODUCT_FACTS = {
    "pricing": "CloudSync Team plan costs $19/user/month, up to 25 users.",
    "trial": "All paid plans include a 14-day free trial.",
    "support": "Phone support is available for Enterprise customers only.",
    "storage": "Team plan includes 100 GB storage. Business plan includes 1 TB.",
}

USER_QUESTION = "What does the Team plan include and how much does it cost?"


async def main():
    from openai import AsyncOpenAI

    from director_ai.voice import ElevenLabsAdapter, voice_pipeline

    client = AsyncOpenAI()
    tts = ElevenLabsAdapter(voice_id="JBFqnCBsd6RMkjVDRZzb")

    print(f"User: {USER_QUESTION}")
    print("Streaming guardrailed audio...\n")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a CloudSync product support agent. "
                    "Answer based on product documentation."
                ),
            },
            {"role": "user", "content": USER_QUESTION},
        ],
        stream=True,
    )

    async def llm_tokens():
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if token:
                print(token, end="", flush=True)
                yield token
        print()

    audio_chunks = []
    halted = False

    def on_halt(vtoken):
        nonlocal halted
        halted = True
        print(f"\n[HALT] reason={vtoken.halt_reason} coherence={vtoken.coherence:.3f}")

    async for audio_chunk in voice_pipeline(
        llm_tokens(),
        tts,
        facts=PRODUCT_FACTS,
        prompt=USER_QUESTION,
        threshold=0.3,
        hard_limit=0.25,
        score_every=4,
        soft_halt=True,
        recovery="Let me verify that information and get back to you.",
        on_halt=on_halt,
    ):
        audio_chunks.append(audio_chunk)

    total_bytes = sum(len(c) for c in audio_chunks)
    print(f"\nAudio: {len(audio_chunks)} chunks, {total_bytes:,} bytes total")
    if halted:
        print("(Response was halted due to detected hallucination)")


if __name__ == "__main__":
    asyncio.run(main())
