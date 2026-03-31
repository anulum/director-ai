# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Twilio Voice Webhook Example

"""Guardrailed phone support bot with Twilio + Director-AI.

Implements a Flask webhook that:
1. Receives a Twilio voice call
2. Transcribes caller speech via Twilio <Gather>
3. Generates an LLM response
4. Runs VoiceGuard on the response
5. Speaks only approved text back via Twilio <Say>

Requirements::

    pip install director-ai[nli] flask openai twilio

Set environment variables::

    OPENAI_API_KEY=sk-...
    TWILIO_ACCOUNT_SID=AC...
    TWILIO_AUTH_TOKEN=...

Usage::

    python examples/voice_twilio_webhook.py
    # Expose with ngrok: ngrok http 5000
    # Point your Twilio phone number webhook to https://<ngrok-url>/voice

Architecture::

    Caller -> Twilio -> /voice (greeting + gather)
                     -> /handle-speech (transcription)
                     -> OpenAI GPT-4o-mini (LLM)
                     -> VoiceGuard (guardrail)
                     -> Twilio <Say> (approved text only)
"""

from __future__ import annotations

from flask import Flask, request

app = Flask(__name__)

# Product knowledge base
SUPPORT_FACTS = {
    "hours": "Support hours are Monday to Friday, 9 AM to 6 PM Eastern.",
    "refund": "Full refund within 30 days. No refunds after 30 days.",
    "pricing": "Basic plan $29/month. Pro plan $79/month. Enterprise custom pricing.",
    "trial": "14-day free trial for Basic and Pro plans.",
    "phone": "Phone support available for Pro and Enterprise customers only.",
    "sla": "Enterprise SLA guarantees 99.99% uptime.",
}

SYSTEM_PROMPT = (
    "You are a phone support agent for Acme Corp. "
    "Answer caller questions based on company policy. "
    "Keep answers concise — the caller is listening, not reading."
)


def _generate_and_guard(user_text: str) -> str:
    """Generate LLM response and run it through VoiceGuard."""
    from openai import OpenAI

    from director_ai import VoiceGuard

    client = OpenAI()
    guard = VoiceGuard(
        facts=SUPPORT_FACTS,
        prompt=user_text,
        threshold=0.3,
        hard_limit=0.25,
        score_every=4,
        soft_halt=True,
        recovery="I need to check on that. Let me transfer you to a specialist.",
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        stream=True,
    )

    tokens = []
    for chunk in response:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue
        result = guard.feed(token)
        if result.halted:
            tokens.append(f" {result.recovery_text}")
            break
        tokens.append(token)

    return "".join(tokens)


@app.route("/voice", methods=["POST"])
def voice_entry():
    """Twilio calls this when a call connects."""
    from twilio.twiml.voice_response import Gather, VoiceResponse

    resp = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/handle-speech",
        method="POST",
        speech_timeout="auto",
        language="en-US",
    )
    gather.say("Hello, this is Acme support. How can I help you?", voice="Polly.Joanna")
    resp.append(gather)
    resp.say("I didn't catch that. Please call again.", voice="Polly.Joanna")
    return str(resp), 200, {"Content-Type": "text/xml"}


@app.route("/handle-speech", methods=["POST"])
def handle_speech():
    """Twilio posts transcribed speech here."""
    from twilio.twiml.voice_response import Gather, VoiceResponse

    caller_text = request.form.get("SpeechResult", "")
    if not caller_text.strip():
        resp = VoiceResponse()
        resp.say("I didn't understand. Could you repeat that?", voice="Polly.Joanna")
        resp.redirect("/voice")
        return str(resp), 200, {"Content-Type": "text/xml"}

    # Run through LLM + VoiceGuard
    answer = _generate_and_guard(caller_text)

    resp = VoiceResponse()
    resp.say(answer, voice="Polly.Joanna")

    # Continue the conversation
    gather = Gather(
        input="speech",
        action="/handle-speech",
        method="POST",
        speech_timeout="auto",
        language="en-US",
    )
    gather.say("Is there anything else I can help with?", voice="Polly.Joanna")
    resp.append(gather)
    resp.say("Thank you for calling Acme support. Goodbye!", voice="Polly.Joanna")
    return str(resp), 200, {"Content-Type": "text/xml"}


if __name__ == "__main__":
    app.run(port=5000, debug=True)
