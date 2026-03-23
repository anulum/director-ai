# Voice AI Integration

Real-time hallucination prevention for text-to-speech pipelines. `VoiceGuard` sits between your LLM and TTS engine, scoring tokens as they arrive and halting the stream before hallucinated content reaches the speaker.

Once words are spoken, you can't unsay them. `VoiceGuard` ensures they don't get spoken in the first place.

## Quick Start

```python
from director_ai import VoiceGuard

guard = VoiceGuard(
    facts={"refund": "30-day refund policy", "hours": "9am-5pm EST"},
    prompt="What is the refund policy?",
)

for token in llm.stream("What is the refund policy?"):
    result = guard.feed(token)
    if result.halted:
        tts.stop()
        tts.speak(result.recovery_text)
        break
    tts.speak_chunk(result.token)
```

## How It Works

```
LLM tokens ──→ VoiceGuard.feed() ──→ TTS engine ──→ speaker
                    │
                    ├─ approved=True  → token forwarded to TTS
                    └─ halted=True    → stream stopped, recovery text spoken
```

`VoiceGuard` accumulates tokens and periodically scores the growing text against your knowledge base. Three halt mechanisms run simultaneously:

| Mechanism | Trigger | Behavior |
|-----------|---------|----------|
| **Hard halt** | Single score < `hard_limit` | Immediate stop, token rejected |
| **Window average** | Rolling average < `threshold` | Soft halt (finish sentence) or hard halt |
| **Already halted** | Previous halt fired | All subsequent tokens rejected |

Soft halt mode (default) waits for a sentence boundary (`.`, `!`, `?`) before signaling halt, so the speaker finishes a complete sentence instead of cutting off mid-word.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `facts` | `dict[str, str]` | `None` | Key-value fact pairs for grounding |
| `store` | `GroundTruthStore` | `None` | Pre-built store (overrides `facts`) |
| `prompt` | `str` | `""` | User prompt (used as NLI premise) |
| `threshold` | `float` | `0.3` | Window average coherence floor |
| `hard_limit` | `float` | `0.25` | Immediate halt threshold |
| `score_every` | `int` | `4` | Score every N-th token |
| `window_size` | `int` | `8` | Sliding window for average |
| `soft_halt` | `bool` | `True` | Wait for sentence end before halting |
| `recovery` | `str` | `"I need to verify..."` | Text spoken on halt |
| `use_nli` | `bool` | `True` | Enable NLI model scoring |

## VoiceToken

`feed()` returns a `VoiceToken` for every token:

| Field | Type | Description |
|-------|------|-------------|
| `token` | `str` | The input token |
| `index` | `int` | Position in stream |
| `approved` | `bool` | Whether to forward to TTS |
| `coherence` | `float` | Current coherence score |
| `halted` | `bool` | Whether halt was triggered |
| `halt_reason` | `str` | `hard_limit`, `window_avg`, `soft_halt_sentence_end`, or `already_halted` |
| `recovery_text` | `str` | Text to speak instead (set on halt) |

## Multi-Turn Conversations

Call `reset()` between conversation turns and `set_prompt()` for each new user message:

```python
guard = VoiceGuard(store=my_knowledge_base)

while True:
    user_text = asr.listen()
    guard.reset()
    guard.set_prompt(user_text)

    for token in llm.stream(user_text):
        result = guard.feed(token)
        if result.halted:
            tts.speak(result.recovery_text)
            break
        tts.speak_chunk(result.token)
```

## ElevenLabs Example

```python
from elevenlabs import stream as el_stream
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from director_ai import VoiceGuard

client = OpenAI()
tts = ElevenLabs()
guard = VoiceGuard(
    facts={"product": "Widget Pro costs $49, ships in 2 days"},
    prompt="Tell me about Widget Pro",
)

# Collect approved text, halt if needed
approved_text = []
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me about Widget Pro"}],
    stream=True,
)

for chunk in response:
    token = chunk.choices[0].delta.content or ""
    if not token:
        continue
    result = guard.feed(token)
    if result.halted:
        approved_text.append(f" {result.recovery_text}")
        break
    approved_text.append(token)

# Send approved text to TTS
audio = tts.text_to_speech.convert(
    text="".join(approved_text),
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_turbo_v2_5",
)
el_stream(audio)
```

## Latency Budget

Voice AI requires end-to-end latency under 500ms for natural conversation. `VoiceGuard` with `score_every=4` adds negligible overhead:

| Component | Latency | Notes |
|-----------|---------|-------|
| LLM generation | 50-200ms first token | Depends on provider |
| **VoiceGuard scoring** | **<1ms per check** | 0.5ms on L40S, heuristic <0.1ms |
| TTS synthesis | 50-150ms | Depends on provider |
| Audio streaming | ~100ms buffer | Network + playback |

With `score_every=4` and ~5 tokens/second, scoring runs once per ~800ms — well within budget. Increase `score_every` for lower overhead at the cost of later halt detection.

## Tuning for Voice

Voice conversations differ from text:

- **Lower `hard_limit` (0.20-0.25):** Voice responses are shorter and more conversational. Transient dips are common — don't halt on brief fluctuations.
- **`soft_halt=True` (default):** Cutting mid-word sounds unnatural. Always finish the sentence.
- **Short `recovery` text:** "Let me check on that" is better than a paragraph. The user is listening, not reading.
- **Higher `score_every` (4-8):** Voice generates ~5 tokens/second. Scoring every token wastes cycles on partial words.

## Full API

::: director_ai.integrations.voice.VoiceGuard

::: director_ai.integrations.voice.VoiceToken
