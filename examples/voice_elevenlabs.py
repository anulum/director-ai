# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — ElevenLabs Voice AI Example

"""Guardrailed voice output with ElevenLabs TTS.

Streams an LLM response through VoiceGuard, collects approved text,
and sends it to ElevenLabs for speech synthesis. If a hallucination
is detected mid-stream, the guard halts and the recovery message
is spoken instead.

Requirements::

    pip install director-ai[nli] openai elevenlabs

Set environment variables::

    OPENAI_API_KEY=sk-...
    ELEVENLABS_API_KEY=...

Usage::

    python examples/voice_elevenlabs.py
"""

from __future__ import annotations

# Product knowledge base — replace with your own facts
PRODUCT_FACTS = {
    "pricing": "CloudSync Team plan costs $19/user/month, up to 25 users.",
    "trial": "All paid plans include a 14-day free trial.",
    "support": "Phone support is available for Enterprise customers only.",
    "storage": "Team plan includes 100 GB storage. Business plan includes 1 TB.",
    "encryption": "AES-256 at rest, TLS 1.3 in transit.",
}

USER_QUESTION = "What does the Team plan include and how much does it cost?"


def main():
    from elevenlabs import stream as el_stream
    from elevenlabs.client import ElevenLabs
    from openai import OpenAI

    from director_ai import VoiceGuard

    client = OpenAI()
    tts = ElevenLabs()

    guard = VoiceGuard(
        facts=PRODUCT_FACTS,
        prompt=USER_QUESTION,
        threshold=0.3,
        hard_limit=0.25,
        score_every=4,
        soft_halt=True,
        recovery="Let me verify that information and get back to you.",
    )

    print(f"User: {USER_QUESTION}")
    print("Generating response...")

    response = client.chat.completions.create(
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

    approved_text = []
    halted = False

    for chunk in response:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue

        result = guard.feed(token)
        if result.halted:
            approved_text.append(f" {result.recovery_text}")
            halted = True
            print(
                f"\n[HALT] reason={result.halt_reason} coherence={result.coherence:.3f}"
            )
            break
        approved_text.append(token)

    final_text = "".join(approved_text)
    print(f"\nAssistant: {final_text}")

    if halted:
        print("(Response was halted due to detected hallucination)")

    # Synthesise speech from approved text only
    audio = tts.text_to_speech.convert(
        text=final_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George" voice
        model_id="eleven_turbo_v2_5",
    )
    el_stream(audio)


if __name__ == "__main__":
    main()
