# Credential-Free Voice Demo

`director_ai.voice.demo` provides a deterministic dry-run voice pipeline for
local validation, CI, and onboarding. It uses the production async
`voice_pipeline()` plus `DryRunTTSAdapter`, so the guard, sentence buffering,
halt callback, recovery path, and adapter lifecycle are exercised without
external services or credentials.

```python
from director_ai.voice import run_voice_demo

result = await run_voice_demo(use_nli=False)

print(result.tts_texts)
print(result.audio_chunks)
print(result.total_audio_bytes)
```

Run it from the command line:

```bash
python -m director_ai.voice.demo
```

The dry-run adapter records every text fragment sent to synthesis and emits
deterministic bytes prefixed with `audio:`. This makes it suitable for
regression tests and for validating that a deployment path closes adapters on
all completion and failure paths before replacing the adapter with a real TTS
backend.

::: director_ai.voice.demo.DryRunTTSAdapter

::: director_ai.voice.demo.VoiceDemoResult

::: director_ai.voice.demo.run_voice_demo

::: director_ai.voice.demo.scripted_tokens
